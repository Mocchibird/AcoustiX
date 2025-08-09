import numpy as np
import os
import tensorflow as tf
from pattern import angle_transformation
import matplotlib.pyplot as plt
from rapidfuzz import process
import yaml
import json
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

import soundfile as sf


# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RadioMaterial, LambertianPattern
from sionna import SPEED_OF_LIGHT, DIELECTRIC_PERMITTIVITY_VACUUM


from ir_utils import compute_metric
from pattern import Pattern


def add_room_mesh(ax, ply_file, face_color="lightgrey", alpha=0.15):
    """
    Overlay a .ply mesh (triangles) on an existing Axes3D.

    Parameters
    ----------
    ax        : matplotlib 3-D axis that already contains your RX/TX points
    ply_file  : path to the .ply file (ASCII or binary)
    face_color: any Matplotlib color
    alpha     : 0 (transparent) .. 1 (opaque)
    """
    # 1) load mesh
    room = trimesh.load(ply_file, force='mesh')   # vertices, faces

    # 2) triangles → Poly3DCollection
    triangles = room.vertices[room.faces]         # (F, 3, 3) array
    coll = Poly3DCollection(triangles,
                            facecolor=face_color,
                            edgecolor="k",
                            linewidth=0.1,
                            alpha=alpha)
    ax.add_collection3d(coll)

    # 3) keep sensible axis limits
    mins, maxs = room.bounds
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)

def sanitize_position(pos):
    """Helper to convert position array into a filename-safe string"""
    return '_'.join([f"{x:.2f}" for x in pos])


def load_cfg(config_file):
    """ Load the simulation configurations
    """


    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    rt_config = config['rt_config']
    ir_config = config['ir_config']


    rx_pattern = Pattern(pattern_type=ir_config["rx_pattern"])
    tx_pattern = Pattern(pattern_type=ir_config["tx_pattern"])


    material_db_file = './acoustic_absorptions.json'


    with open(material_db_file, "r") as f:
        material_db = json.load(f)


    simu_config = {
        'rt_config': rt_config,
        'attn': ir_config['attn'],
        'fs': ir_config['fs'],
        'ir_len': ir_config['ir_len'],
        'speed': ir_config['speed'],
        'noise': ir_config['noise'],
        'rx_pattern': rx_pattern,
        'tx_pattern': tx_pattern,
        'material_db': material_db
    }
   
    return simu_config

def ir_kernel(x, frequency=100, decay=3, kernal_type='sinc'):
    """ Impulse response kernel function, can config as
    1. sinc function: sinx/x
    2. exponential cos decay function: cosx*exp(-x)
    """


    if kernal_type == 'sinc':
        kernel_signal =  np.where(x == 0, 1.0, np.sin(frequency * np.pi * x) / (decay * frequency * np.pi * x))
    elif kernal_type == 'exp_cosine':
        kernel_signal = np.cos(frequency*np.pi*x) / np.exp(decay * frequency * x)


    return kernel_signal

def calculate_reflection_coefficient(scene, reflect_coeff):
    imag_eta = (np.log(2 / (1-reflect_coeff) - 1) / 2) ** 2
    conductivity = imag_eta * 2 * np.pi * scene.frequency * DIELECTRIC_PERMITTIVITY_VACUUM    
   
    return conductivity

def assign_material(scene, material_db):
    """ Assign acoustic material to the objects in the scene
    """


    wall = RadioMaterial("wall",
                        relative_permittivity=1.0,
                        conductivity=10,
                        scattering_coefficient=0.6,
                        scattering_pattern=LambertianPattern())
    scene.add(wall)


    for sub_object in scene.objects:
       
        obj = scene.get(sub_object)
        match, score, _ = process.extractOne(obj.radio_material.name, material_db.keys())


        if match:
            material_data = material_db[match]
           
            reflectivity = 0
            for _, value in material_data.items():
                reflectivity += value
            reflectivity = reflectivity / len(material_data.items())


            conductivity = calculate_reflection_coefficient(scene, reflectivity)
            template_material = RadioMaterial(f"{obj.radio_material.name}",
                                            relative_permittivity=1.0,
                                            conductivity=conductivity+10,
                                            scattering_coefficient=0,
                                            scattering_pattern=LambertianPattern())


            try:
                scene.add(template_material)
                obj.radio_material = template_material
            except:
                pass
        else:
            obj.radio_material = wall
    
def config_scene(scene_path):
    """ Config scene and tx and rx arrays
    """


    scene = load_scene(scene_path)


    scene.frequency = 17.9 * 1e9
    scene.synthetic_array = True


    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.1, horizontal_spacing=0.1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.1, horizontal_spacing=0.1, pattern="iso", polarization="cross")


    return scene

def ir_simulation(scene_path, rx_pos, tx_pos, rx_ori, tx_ori, simu_config):
    """
    Computes propagation paths


    Input
    ------
    scene_path: path to ".xml"
        the path to the simulation .xml file


    rx_pos : numpy array [n_mics, 3]
        microphone positions


    tx_pos : numpy array [n_speaker, 3]
        speaker positions (by default, only one speaker is used)


    rx_ori : numpy array [n_speaker ,3]
        microphone orientation in unit vector


    tx_ori : numpy array [n_speaker, 3]
        speaker orientation in unit vector


    simu_config : dictionary
        ray tracing configurations for acoustic simulations


    Output
    ------
    ir_samples: numpy array [n_mics, ir_seqlen]
        simulated impulse response


    rx_pos: numpy array [n_mics, 3]
        rx positions


    rx_ori: numpy array [n_mics, 3]
        rx orientations
    """


    # ---------- Load basic configuration here ----------
    fs = simu_config['fs']
    ir_len = simu_config['ir_len']
    speed = simu_config['speed']
    noise = simu_config['noise']
    rt_config = simu_config['rt_config']


    attn_coeff = simu_config['attn']


    rx_pattern_func = simu_config['rx_pattern']
    tx_pattern_func = simu_config['tx_pattern']
    material_db = simu_config['material_db']


    BASIC_VOLUME = 1000


    # generate ir template
    x = np.linspace(-10, 10, ir_len*2)
    y = ir_kernel(x, frequency=40, decay=0.25)[ir_len:]
    sinc_template = y / np.max(y)
    sinc_template = np.concatenate((sinc_template, np.zeros(ir_len)))


    # ---------- Load scene ----------
    scene = config_scene(scene_path)


    # Create transmitter and receiver
    tx = Transmitter(name="tx", position=tx_pos)
    scene.add(tx)


    for i, pos in enumerate(rx_pos):
        rx = Receiver(name=f"rx+{i}",
                    position=pos,
                    orientation=[0,0,0])    
        scene.add(rx)


    # ---------- config the material properties here ----------
    assign_material(scene, material_db)
   
    # ---------- ray tracing part here ----------
    paths = scene.compute_paths(
        max_depth=rt_config['max_depth'],
        num_samples=rt_config['num_samples'],
        los=rt_config['los'],
        reflection=rt_config['reflection'],
        diffraction=rt_config['diffraction'],
        scattering=rt_config['scattering'],
        scat_keep_prob=rt_config['scat_prob'],
        scat_random_phases=False)
    paths.normalize_delays = False
   
    # get the wave propagation distance
    propagation_distance = np.array(paths.tau[0,:,0,:].numpy() * SPEED_OF_LIGHT)


    # get the ray frequency dependent amplitude
    amplitude_with_sign = paths.a[0,:,:,0,0,:,0].numpy()
    ray_amplitude = np.array(np.real(amplitude_with_sign))
    ray_amplitude = np.sqrt(np.sum(ray_amplitude**2, axis=1)) * np.sign(ray_amplitude[:,0,:])


    rx_azimuth = paths.phi_r[0,:, 0,:]
    rx_elevation = paths.theta_r[0,:, 0,:]


    tx_azimuth = paths.phi_t[0,:, 0,:]
    tx_elevation = paths.theta_t[0,:, 0,:]


    if paths.mask.numpy().shape[1]:
        pro_dis = []
        ray_amp = []
        rx_azs = []
        rx_els = []
        tx_azs = []
        tx_els = []


        for i in range(paths.mask.numpy().shape[1]):
            true_path = np.where(paths.mask.numpy()[0,i,0,:] == True)[0]
            pro_dis.append(propagation_distance[i,true_path])
            ray_amp.append(ray_amplitude[i,true_path])


            rx_azs.append(tf.gather(rx_azimuth[i], true_path))
            rx_els.append(tf.gather(rx_elevation[i], true_path))
            tx_azs.append(tf.gather(tx_azimuth[i], true_path))
            tx_els.append(tf.gather(tx_elevation[i], true_path))
           
    ir_samples = []
    valid_mask = []


    ## ---------- calculate the impulse resposne ----------
    for index, (propagation_dis, ray_energy, rx_az, rx_el, tx_az, tx_el) in enumerate(zip(pro_dis, ray_amp, rx_azs, rx_els, tx_azs, tx_els)):
        energy_ifft = np.zeros(ir_len) * 1j


        propagation_idx = propagation_dis / speed * fs
        valid_samples = (propagation_idx > 0) & (propagation_idx < ir_len)


        ray_energy = ray_energy[valid_samples] * BASIC_VOLUME
        propagation_idx = propagation_idx[valid_samples]


        rx_az = rx_az[valid_samples]
        rx_el = rx_el[valid_samples]
        tx_az = tx_az[valid_samples]
        tx_el = tx_el[valid_samples]


        oriented_rx_angle = angle_transformation(rx_ori[index], np.stack([rx_az, rx_el], axis=-1))
        oriented_tx_angle = angle_transformation(tx_ori,  np.stack([tx_az, tx_el], axis=-1))


        rx_gain = rx_pattern_func.get_pattern(oriented_rx_angle)
        tx_gain = tx_pattern_func.get_pattern(oriented_tx_angle)


        # sum up the energy of each path
        for i, energy in enumerate(ray_energy):
            shift_samples = propagation_idx[i]
            if i == 0:
                shift_samples = np.ceil(shift_samples)
                energy_los = energy
            else:
                energy = energy * np.random.choice([-1, 1])
                   
            energy_ifft += tx_gain[i] * rx_gain[i] * energy * np.exp(-1j*2*np.pi/ir_len*np.arange(0, ir_len)*shift_samples) * np.exp(-shift_samples * attn_coeff)


        time_signal = np.real(np.fft.ifft(energy_ifft * np.fft.fft(sinc_template[:ir_len]))) + noise * (np.random.randn(ir_len))
        energy_ifft = np.fft.fft(time_signal)


        if np.max(np.abs(time_signal)) < 50*noise: valid_mask.append(0)
        else: valid_mask.append(1)


        ir_samples.append(time_signal)
   
    ir_samples = np.array(ir_samples)
    valid_mask = np.array(valid_mask)


    valid_ir = np.where(valid_mask)[0]
    ir_samples = ir_samples[valid_ir]
    rx_pos = rx_pos[valid_ir]
    rx_ori = rx_ori[valid_ir]


    return ir_samples, rx_pos, rx_ori

def plot_figure(ir_samples, rx_pos, tx_pos, rx_ori, tx_ori, fs):
    for i in range(len(ir_samples)):
        sampled_ir = ir_samples[i]
        fft_signal = np.fft.fft(sampled_ir)

        energy, t60, C50, edt = compute_metric(sampled_ir, fs)
 
        plt.figure(figsize=(12,12))
        plt.suptitle(f"t60 = {t60}, C50 = {C50}, EDT = {edt}")


        plt.subplot(221)
        plt.title("Simulated impulse response")
        plt.plot(sampled_ir)


        plt.subplot(222)
        plt.title("IR energy decay trend")
        plt.plot(energy)


        plt.subplot(223)
        plt.title("Channel Impulse response")
        plt.plot(np.abs(fft_signal))


        plt.subplot(224)
        plt.title("Position of both speaker and microphone")
        plt.scatter(rx_pos[:,0], rx_pos[:,1])
        plt.scatter(rx_pos[i,0], rx_pos[i,1], c='r', s=100)
        plt.quiver(rx_pos[i, 0], rx_pos[i, 1], rx_ori[i,0], rx_ori[i,1], angles='xy', scale_units='xy', scale=1, color='r', width=0.005)
        plt.scatter(tx_pos[0], tx_pos[1], c='b',s=100)
        plt.quiver(tx_pos[0], tx_pos[1], tx_ori[0], tx_ori[1], angles='xy', scale_units='xy', scale=1, color='r', width=0.005)

        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.axis("equal")
        plt.grid(True)
        plt.show()

def save_ir(ir_samples, rx_pos, rx_ori, tx_pos, tx_ori, save_path, prefix, fs, 
           original_positions=None, roomname="EmptyRoom"):
    """Function to save the impulse response samples"""
    
    # Ensure directory exists
    rir_dir = os.path.join(save_path, 'RIR')
    os.makedirs(rir_dir, exist_ok=True)

    # Sanitize transmitter position for filename uniqueness
    tx_tag = sanitize_position(tx_pos)

    for i in range(ir_samples.shape[0]):
        ir = ir_samples[i, :]
        position_rx = rx_pos[i, :]
        orientation_rx = rx_ori[i, :]
        position_tx = tx_pos
        orientation_tx = tx_ori

        energy, t60, C50, edt = compute_metric(ir, fs)

        # Create unique filename
        file_base = f"ir_{tx_tag}_{str(int(prefix+i)).zfill(6)}"
        npz_path = os.path.join(rir_dir, f"{file_base}.npz")

        # Save IR and metadata
        np.savez(npz_path,
                 ir=ir,
                 position_rx=position_rx,
                 position_tx=position_tx,
                 orientation_rx=orientation_rx,
                 orientation_tx=orientation_tx,
                 energy=energy,
                 t60=t60,
                 C50=C50,
                 edt=edt)

        # =================== ORIGINAL PLOT ===================
        sampled_ir = ir
        fft_signal = np.fft.fft(sampled_ir)

        plt.figure(figsize=(12, 9))
        plt.suptitle(f"t60 = {t60:.2f}, C50 = {C50:.2f}, EDT = {edt:.2f}")

        plt.subplot(221)
        plt.title("Simulated impulse response")
        plt.plot(sampled_ir)

        plt.subplot(222)
        plt.title("IR energy decay trend")
        plt.plot(energy)

        plt.subplot(223)
        plt.title("Channel Impulse response (FFT Magnitude)")
        plt.plot(np.abs(fft_signal))

        ax = plt.subplot(224, projection='3d')
        ax.set_title("Room mesh + speaker + microphones")

        # Add room mesh
        add_room_mesh(ax, "./extract_scene/LRoom/meshes/LRoom.ply")

        # All RXs (small black dots)
        ax.scatter(rx_pos[:, 0], rx_pos[:, 1], rx_pos[:, 2], c="k", s=15)
        
        # Highlight current RX in red
        ax.scatter(rx_pos[i, 0], rx_pos[i, 1], rx_pos[i, 2], c="r", s=80)
        
        # TX in blue
        ax.scatter(tx_pos[0], tx_pos[1], tx_pos[2], c="b", s=120)
        
        # Orientation arrows
        ax.quiver(rx_pos[:, 0], rx_pos[:, 1], rx_pos[:, 2],
                  rx_ori[:, 0], rx_ori[:, 1], rx_ori[:, 2],
                  length=0.35, color="r", linewidth=0.5)
        
        ax.quiver(tx_pos[0], tx_pos[1], tx_pos[2],
                  tx_ori[0], tx_ori[1], tx_ori[2],
                  length=0.5, color="b")
        
        ax.view_init(elev=20, azim=135)
        ax.grid(False)

        plt.savefig(os.path.join(rir_dir, f"{file_base}.png"))
        plt.close('all')

        # =================== ADDITIONAL 4-VIEW PLOT ===================
        if original_positions is not None:
            create_4view_plot(rx_pos, tx_pos, position_rx, position_tx, 
                            original_positions, rir_dir, file_base)

def create_4view_plot(all_rx_pos, tx_pos, current_rx_pos, current_tx_pos, 
                     original_positions, save_dir, file_base):
    """Create 4-view plot with jittered positions highlighted"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Position Views - Current RX: [{current_rx_pos[0]:.2f}, {current_rx_pos[1]:.2f}, {current_rx_pos[2]:.2f}]", 
                 fontsize=14)
    
    # Determine which positions are original vs jittered
    original_indices = []
    jittered_indices = []
    
    for idx, pos in enumerate(all_rx_pos):
        # Check if this position is close to any original position (within small tolerance)
        distances = np.linalg.norm(original_positions - pos, axis=1)
        if np.min(distances) < 0.01:  # 1cm tolerance for "original"
            original_indices.append(idx)
        else:
            jittered_indices.append(idx)
    
    # Get room bounds for consistent axis limits
    room_bounds = {
        'x': [np.min(all_rx_pos[:, 0]) - 0.5, np.max(all_rx_pos[:, 0]) + 0.5],
        'y': [np.min(all_rx_pos[:, 1]) - 0.2, np.max(all_rx_pos[:, 1]) + 0.2],
        'z': [np.min(all_rx_pos[:, 2]) - 0.5, np.max(all_rx_pos[:, 2]) + 0.5]
    }
    
    # 1. Front View (looking from +X towards -X, Y-Z plane)
    ax1 = plt.subplot(221)
    ax1.set_title("Front View (+X → -X)")
    
    # Plot original positions
    if original_indices:
        ax1.scatter(all_rx_pos[original_indices, 1], all_rx_pos[original_indices, 2], 
                   c="blue", s=40, alpha=0.8, label=f"Original ({len(original_indices)})")
    
    # Plot jittered positions
    if jittered_indices:
        ax1.scatter(all_rx_pos[jittered_indices, 1], all_rx_pos[jittered_indices, 2], 
                   c="red", s=20, alpha=0.6, label=f"Jittered ({len(jittered_indices)})")
    
    # Highlight current RX
    ax1.scatter(current_rx_pos[1], current_rx_pos[2], c="orange", s=100, 
               marker="*", edgecolors="black", linewidth=2, label="Current RX")
    
    # TX position
    ax1.scatter(current_tx_pos[1], current_tx_pos[2], c="green", s=120, 
               marker="^", edgecolors="black", linewidth=2, label="TX")
    
    ax1.set_xlabel("Y (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_xlim(room_bounds['y'])
    ax1.set_ylim(room_bounds['z'])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    
    # 2. Right View (looking from +Y towards -Y, X-Z plane)
    ax2 = plt.subplot(222)
    ax2.set_title("Right View (+Y → -Y)")
    
    # Plot original positions
    if original_indices:
        ax2.scatter(all_rx_pos[original_indices, 0], all_rx_pos[original_indices, 2], 
                   c="blue", s=40, alpha=0.8, label=f"Original ({len(original_indices)})")
    
    # Plot jittered positions
    if jittered_indices:
        ax2.scatter(all_rx_pos[jittered_indices, 0], all_rx_pos[jittered_indices, 2], 
                   c="red", s=20, alpha=0.6, label=f"Jittered ({len(jittered_indices)})")
    
    # Highlight current RX
    ax2.scatter(current_rx_pos[0], current_rx_pos[2], c="orange", s=100, 
               marker="*", edgecolors="black", linewidth=2, label="Current RX")
    
    # TX position
    ax2.scatter(current_tx_pos[0], current_tx_pos[2], c="green", s=120, 
               marker="^", edgecolors="black", linewidth=2, label="TX")
    
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_xlim(room_bounds['x'])
    ax2.set_ylim(room_bounds['z'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_aspect('equal')
    
    # 3. Top View (looking from +Z towards -Z, X-Y plane)
    ax3 = plt.subplot(223)
    ax3.set_title("Top View (+Z → -Z)")
    
    # Plot original positions
    if original_indices:
        ax3.scatter(all_rx_pos[original_indices, 0], all_rx_pos[original_indices, 1], 
                   c="blue", s=40, alpha=0.8, label=f"Original ({len(original_indices)})")
    
    # Plot jittered positions
    if jittered_indices:
        ax3.scatter(all_rx_pos[jittered_indices, 0], all_rx_pos[jittered_indices, 1], 
                   c="red", s=20, alpha=0.6, label=f"Jittered ({len(jittered_indices)})")
    
    # Highlight current RX
    ax3.scatter(current_rx_pos[0], current_rx_pos[1], c="orange", s=100, 
               marker="*", edgecolors="black", linewidth=2, label="Current RX")
    
    # TX position
    ax3.scatter(current_tx_pos[0], current_tx_pos[1], c="green", s=120, 
               marker="^", edgecolors="black", linewidth=2, label="TX")
    
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_xlim(room_bounds['x'])
    ax3.set_ylim(room_bounds['y'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.set_aspect('equal')
    
    # 4. 3D Angled View (same as original but with jitter distinction)
    ax4 = plt.subplot(224, projection='3d')
    ax4.set_title("3D View (with jitter highlight)")
    
    # Plot original positions
    if original_indices:
        ax4.scatter(all_rx_pos[original_indices, 0], all_rx_pos[original_indices, 1], 
                   all_rx_pos[original_indices, 2], c="blue", s=40, alpha=0.8, 
                   label=f"Original ({len(original_indices)})")
    
    # Plot jittered positions
    if jittered_indices:
        ax4.scatter(all_rx_pos[jittered_indices, 0], all_rx_pos[jittered_indices, 1], 
                   all_rx_pos[jittered_indices, 2], c="red", s=20, alpha=0.6, 
                   label=f"Jittered ({len(jittered_indices)})")
    
    # Highlight current RX
    ax4.scatter(current_rx_pos[0], current_rx_pos[1], current_rx_pos[2], 
               c="orange", s=100, marker="*", edgecolors="black", linewidth=2)
    
    # TX position
    ax4.scatter(current_tx_pos[0], current_tx_pos[1], current_tx_pos[2], 
               c="green", s=120, marker="^", edgecolors="black", linewidth=2)
    
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")
    ax4.set_xlim(room_bounds['x'])
    ax4.set_ylim(room_bounds['y'])
    ax4.set_zlim(room_bounds['z'])
    ax4.view_init(elev=20, azim=135)
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{file_base}_4views.png"), dpi=150, bbox_inches='tight')
    plt.close()

def generate_rx_samples(n_random_samples, xyz_max, xyz_min):
    """ Randomly generate the rx samples with in a xyz boundary
    """
    x_flat = np.random.rand(n_random_samples) * (xyz_max[0] - xyz_min[0]) + xyz_min[0]
    y_flat = np.random.rand(n_random_samples) * (xyz_max[1] - xyz_min[1]) + xyz_min[1]
    z_flat = np.random.rand(n_random_samples) * (xyz_max[2] - xyz_min[2]) + xyz_min[2]

    rx_pos = np.stack([x_flat, y_flat, z_flat], axis=-1)
    rx_ori = (np.random.rand(*rx_pos.shape) - 0.5) 
    rx_ori[:,2] *= 0
    rx_ori = rx_ori / np.linalg.norm(rx_ori, axis=-1, keepdims=True)

    return rx_pos, rx_ori

def generate_grid(xyz_min, xyz_max, grid_n, margin):
    # Compute inner start/end for each axis
    mins = np.array(xyz_min)
    maxs = np.array(xyz_max)
    starts = mins + margin
    ends   = maxs - margin

    # Build each axis with grid_n evenly spaced points
    axes = [
        np.linspace(starts[i], ends[i], grid_n)
        for i in range(3)
    ]

    # Make the 3×3×3 mesh
    xs, ys, zs = np.meshgrid(*axes, indexing="ij")

    # Flatten to (N,3) and convert to Python lists if needed
    poses = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    return poses
