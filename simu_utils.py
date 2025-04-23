import numpy as np
import os
import tensorflow as tf
from pattern import angle_transformation
import matplotlib.pyplot as plt
from rapidfuzz import process
import yaml
import json

import soundfile as sf


# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RadioMaterial, LambertianPattern
from sionna import SPEED_OF_LIGHT, DIELECTRIC_PERMITTIVITY_VACUUM


from ir_utils import compute_metric
from pattern import Pattern



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


        energy, t60, C50 = compute_metric(sampled_ir, fs)
 
        plt.figure(figsize=(12,12))
        plt.suptitle(f"t60 = {t60}, C50 = {C50}")


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


        plt.axis("equal")
        plt.grid(True)
        plt.show()



def save_ir(ir_samples, rx_pos, rx_ori, tx_pos, tx_ori, save_path, prefix):
    """ Function to save the impulse response samples
    """
    for i in range(ir_samples.shape[0]):
        ir = ir_samples[i,:]
        position_rx = rx_pos[i,:]
        orientation_rx = rx_ori[i,:]
        position_tx = tx_pos
        orientation_tx = tx_ori


        np.savez(os.path.join(save_path, f'ir_{str(int(prefix+i)).zfill(6)}.npz'),
                ir=ir,
                position_rx=position_rx,
                position_tx=position_tx,
                orientation_rx=orientation_rx,
                orientation_tx=orientation_tx)

def save_ir_raw(ir_samples, rx_pos, rx_ori, tx_pos, tx_ori, save_path, prefix, sample_rate=16000):
    for i in range(ir_samples.shape[0]):
        ir = ir_samples[i,:]
        position_rx = rx_pos[i,:]
        orientation_rx = rx_ori[i,:]
        position_tx = tx_pos
        orientation_tx = tx_ori
        
        folder_name = f"{prefix+i:06d}"
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Save RIR as rir.wav
        rir_path = os.path.join(folder_path, 'rir.wav')
        sf.write(rir_path, ir, sample_rate)

        # Save rx position as rx_pos.txt
        rx_pos_path = os.path.join(folder_path, 'rx_pos.txt')
        with open(rx_pos_path, 'w') as f:
            f.write(f"{position_rx[0]},{position_rx[1]},{position_rx[2]}\n")

        # Save tx position and orientation as tx_pos.txt (x,y,z,ori_x,ori_y,ori_z)
        tx_pos_path = os.path.join(folder_path, 'tx_pos.txt')
        with open(tx_pos_path, 'w') as f:
            f.write(f"{position_tx[0]},{position_tx[1]},{position_tx[2]},{orientation_tx[0]},{orientation_tx[1]},{orientation_tx[2]}\n")

        




# This should be added to simu_utils.py
def generate_rx_samples(n_random_samples=10, boundaries=None, xyz_min=None, xyz_max=None):
    """
    Generate random receiver samples within specified boundaries.
   
    Args:
        n_random_samples: Number of samples to generate
        boundaries: List of boundary boxes, each with [min_point, max_point] where each point is [x,y,z]
        xyz_min, xyz_max: Legacy parameters for backward compatibility
       
    Returns:
        rx_pos: Generated positions
        rx_ori: Generated orientations
    """
    # Initialize arrays to store results
    rx_pos = np.zeros((n_random_samples, 3))
    rx_ori = np.zeros((n_random_samples, 3))
   
    # Use legacy parameters if boundaries not provided
    if boundaries is None and xyz_min is not None and xyz_max is not None:
        boundaries = [[xyz_min, xyz_max]]
   
    # Calculate volumes for weighted sampling
    volumes = []
    for box in boundaries:
        min_point, max_point = np.array(box[0]), np.array(box[1])
        volumes.append(np.prod(max_point - min_point))
    total_volume = sum(volumes)
   
    # Calculate samples per box proportional to volume
    samples_per_box = [int(n_random_samples * vol / total_volume) for vol in volumes]
    # Ensure we have exactly n_random_samples
    samples_per_box[-1] += n_random_samples - sum(samples_per_box)
   
    # Generate samples for each box
    index = 0
    for i, box in enumerate(boundaries):
        min_point, max_point = np.array(box[0]), np.array(box[1])
        samples = samples_per_box[i]
       
        # Generate random positions within this box
        for j in range(samples):
            if index < n_random_samples:
                # Random position within the box
                rx_pos[index] = min_point + np.random.random(3) * (max_point - min_point)
               
                # Random orientation (normalized vector)
                ori = np.random.randn(3)
                rx_ori[index] = ori / np.linalg.norm(ori)
               
                index += 1
   
    return rx_pos, rx_ori


def generate_rx_samples_path(path_points, n_samples):
    """
    Generate receiver samples along a specified path.


    The path is defined by a list/array of coordinates [x, y, z].
    The function interpolates along the line segments connecting each
    waypoint in order to generate n_samples that are uniformly spaced
    along the total path length. The orientation for each sample is computed
    as the normalized vector (with z set to 0) pointing from the current sample
    to the next sample. For the last sample, the orientation is set to be equal
    to that of the previous sample.


    Parameters:
        path_points: array-like of shape (M,3)
                     List of M waypoints that define the path.
        n_samples: int
                   Total number of samples to generate along the entire path.
                   
    Returns:
        rx_pos: numpy.ndarray of shape (n_samples, 3)
                The generated positions along the path.
        rx_ori: numpy.ndarray of shape (n_samples, 3)
                The generated orientations as horizontal unit vectors.
    """
    # Convert the path to a NumPy array.
    path_points = np.array(path_points)
    if path_points.shape[0] < 2:
        raise ValueError("At least two points are required to define a path.")
   
    # Compute differences and segment distances.
    seg_vecs = path_points[1:] - path_points[:-1]
    seg_dists = np.linalg.norm(seg_vecs, axis=1)
   
    # Compute cumulative distance along the path. The first point is at distance 0.
    cum_dist = np.concatenate(([0], np.cumsum(seg_dists)))
    total_length = cum_dist[-1]
   
    # Uniformly spaced distances along the entire path.
    target_dists = np.linspace(0, total_length, n_samples)
   
    rx_pos = np.zeros((n_samples, 3))
   
    # Interpolate positions along the path.
    for i, d in enumerate(target_dists):
        # Find which segment the distance d falls in.
        seg_idx = np.searchsorted(cum_dist, d, side='right') - 1
        # Clamp seg_idx to a valid segment index.
        if seg_idx >= len(seg_vecs):
            seg_idx = len(seg_vecs) - 1
       
        # Fractional distance along the segment.
        if seg_dists[seg_idx] > 0:
            t = (d - cum_dist[seg_idx]) / seg_dists[seg_idx]
        else:
            t = 0.0
        # Linear interpolation between the segment's endpoints.
        rx_pos[i] = path_points[seg_idx] + t * seg_vecs[seg_idx]
   
    # Compute orientations. For each sample (except the last),
    # the orientation points from the current position to the next.
    rx_ori = np.zeros((n_samples, 3))
    for i in range(n_samples - 1):
        diff = rx_pos[i+1] - rx_pos[i]
        # Use only the horizontal (xy) components.
        diff[2] = 0
        norm = np.linalg.norm(diff)
        if norm > 0:
            rx_ori[i] = diff / norm
        else:
            rx_ori[i] = np.array([1, 0, 0])
   
    # For the last sample, copy the orientation from the previous sample.
    if n_samples > 1:
        rx_ori[-1] = rx_ori[-2]
    else:
        rx_ori[-1] = np.array([1, 0, 0])
   
    return rx_pos, rx_ori