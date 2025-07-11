import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import json
import numpy as np
import argparse
from tqdm import tqdm
from shutil import copyfile

from simu_utils import ir_simulation, plot_figure, save_ir, load_cfg, generate_rx_samples

# GPU setup
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f"Using GPU: {gpus[0]}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1)

# L-Room specific configuration
L_ROOM_CONFIG = {
    'part1': {
        'x_range': (0, 4),
        'y_range': (0, 3),
        'z_range': (0, 2),
        'volume': 4 * 3 * 2  # 24
    },
    'part2': {
        'x_range': (2, 4),
        'y_range': (0, 3),
        'z_range': (-2, 0),
        'volume': 2 * 3 * 2  # 12
    }
}

def generate_orientations(num_orientations=1, mode='random'):
    """Generate multiple orientations for transmitter/receiver"""
    orientations = []
    
    if mode == 'random':
        for _ in range(num_orientations):
            # Random spherical coordinates
            theta = np.random.uniform(0, 2 * np.pi)  # azimuth
            phi = np.random.uniform(0, np.pi)        # elevation
            
            # Convert to cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            ori = np.array([x, y, z])
            orientations.append(ori / np.linalg.norm(ori))
    
    elif mode == 'uniform':
        # Generate uniform orientations around sphere
        for i in range(num_orientations):
            theta = (2 * np.pi * i) / num_orientations
            phi = np.pi / 3  # 60 degrees elevation
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            ori = np.array([x, y, z])
            orientations.append(ori / np.linalg.norm(ori))
    
    elif mode == 'cardinal':
        # Cardinal directions + up/down
        cardinal_dirs = [
            [1, 0, 0],   # +X
            [-1, 0, 0],  # -X
            [0, 1, 0],   # +Y
            [0, -1, 0],  # -Y
            [0, 0, 1],   # +Z
            [0, 0, -1]   # -Z
        ]
        for i in range(min(num_orientations, len(cardinal_dirs))):
            orientations.append(np.array(cardinal_dirs[i]))
    
    return np.array(orientations)

def is_point_in_l_room(point, margin=0.1):
    """Check if a point is inside the L-shaped room"""
    x, y, z = point
    
    # Add margin check
    part1 = (margin <= x <= 4-margin and 
             margin <= y <= 3-margin and 
             margin <= z <= 2-margin)
    
    part2 = (2+margin <= x <= 4-margin and 
             margin <= y <= 3-margin and 
             -2+margin <= z <= -margin)
    
    return part1 or part2

def generate_l_room_grid(grid_density=5, margin=0.1):
    """Generate uniform grid for L-shaped room"""
    positions = []
    
    # Part 1: X 0:4, Y 0:3, Z 0:2
    part1 = L_ROOM_CONFIG['part1']
    x_range = part1['x_range']
    y_range = part1['y_range'] 
    z_range = part1['z_range']
    
    # Calculate grid points for part 1
    nx1 = int((x_range[1] - x_range[0]) * grid_density / 2)  # Adjust density
    ny1 = int((y_range[1] - y_range[0]) * grid_density / 3)
    nz1 = int((z_range[1] - z_range[0]) * grid_density / 2)
    
    print(f"Part 1 grid: {nx1}x{ny1}x{nz1} = {nx1*ny1*nz1} points")
    
    for i in range(nx1):
        for j in range(ny1):
            for k in range(nz1):
                x = x_range[0] + margin + i * (x_range[1] - x_range[0] - 2*margin) / max(1, nx1-1)
                y = y_range[0] + margin + j * (y_range[1] - y_range[0] - 2*margin) / max(1, ny1-1)
                z = z_range[0] + margin + k * (z_range[1] - z_range[0] - 2*margin) / max(1, nz1-1)
                positions.append([x, y, z])
    
    # Part 2: X 2:4, Y 0:3, Z -2:0
    part2 = L_ROOM_CONFIG['part2']
    x_range = part2['x_range']
    y_range = part2['y_range']
    z_range = part2['z_range']
    
    # Calculate grid points for part 2
    nx2 = int((x_range[1] - x_range[0]) * grid_density / 2)
    ny2 = int((y_range[1] - y_range[0]) * grid_density / 3)
    nz2 = int((z_range[1] - z_range[0]) * grid_density / 2)
    
    print(f"Part 2 grid: {nx2}x{ny2}x{nz2} = {nx2*ny2*nz2} points")
    
    for i in range(nx2):
        for j in range(ny2):
            for k in range(nz2):
                x = x_range[0] + margin + i * (x_range[1] - x_range[0] - 2*margin) / max(1, nx2-1)
                y = y_range[0] + margin + j * (y_range[1] - y_range[0] - 2*margin) / max(1, ny2-1)
                z = z_range[0] + margin + k * (z_range[1] - z_range[0] - 2*margin) / max(1, nz2-1)
                
                # Avoid overlap with part 1
                if not (x >= 2+margin and z >= -margin):
                    positions.append([x, y, z])
    
    return np.array(positions)

def generate_l_room_positions(target_points=100, margin=0.1):
    """Generate specified number of positions in L-room"""
    
    # Calculate optimal grid density
    total_volume = L_ROOM_CONFIG['part1']['volume'] + L_ROOM_CONFIG['part2']['volume']
    density = (target_points / total_volume) ** (1/3)
    
    positions = generate_l_room_grid(grid_density=density*3, margin=margin)
    
    # If we have too many points, randomly sample
    if len(positions) > target_points:
        indices = np.random.choice(len(positions), target_points, replace=False)
        positions = positions[indices]
    
    print(f"Generated {len(positions)} positions for L-shaped room")
    return positions

def visualize_l_room_grid(positions):
    """Visualize L-room grid in 3D"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot grid points
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c='blue', s=30, alpha=0.6)
    
    # Draw room boundaries
    # Part 1 boundaries
    ax.plot([0, 4, 4, 0, 0], [0, 0, 3, 3, 0], [0, 0, 0, 0, 0], 'k-', linewidth=2)  # bottom
    ax.plot([0, 4, 4, 0, 0], [0, 0, 3, 3, 0], [2, 2, 2, 2, 2], 'k-', linewidth=2)  # top
    
    # Part 2 boundaries  
    ax.plot([2, 4, 4, 2, 2], [0, 0, 3, 3, 0], [-2, -2, -2, -2, -2], 'k-', linewidth=2)  # bottom
    ax.plot([2, 4, 4, 2, 2], [0, 0, 3, 3, 0], [0, 0, 0, 0, 0], 'k-', linewidth=2)  # top
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'L-Room Grid ({len(positions)} points)')
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="L-Room impulse response simulation")
    parser.add_argument('--save-data', action="store_true", help='save the simulated impulse response data')
    parser.add_argument('--visualize', action="store_true", help='visualize the grid layout')
    
    parser.add_argument('--rx-gen', default='grid', choices=['random', 'grid'], 
                       help='receiver generation mode')
    parser.add_argument('--tx-gen', default='grid', choices=['grid', 'center', 'random'], 
                       help='transmitter generation mode')
    
    parser.add_argument('--tx-orientations', type=int, default=1, 
                       help='number of orientations per transmitter position')
    parser.add_argument('--rx-orientations', type=int, default=1, 
                       help='number of orientations per receiver position')
    parser.add_argument('--tx-ori-mode', default='random', 
                       choices=['random', 'uniform', 'cardinal'], 
                       help='transmitter orientation generation mode')
    parser.add_argument('--rx-ori-mode', default='random', 
                       choices=['random', 'uniform', 'cardinal'], 
                       help='receiver orientation generation mode')
    
    parser.add_argument('--target-points', type=int, default=100, 
                       help='target number of grid points')
    parser.add_argument('--margin', type=float, default=0.2, 
                       help='margin from walls in meters')
    
    parser.add_argument('--config-file', default="./simu_config/basic_config.yml", 
                       help='config file for the impulse response rendering')
    parser.add_argument('--scene-file', default="./extract_scene/LRoom/LRoom.xml", 
                       help='path to the L-room XML file')
    parser.add_argument('--dataset-name', default="LRoom_dataset", 
                       help='dataset name')

    args = parser.parse_args()
    
    # Setup paths
    scene_path = args.scene_file
    dataset_name = args.dataset_name
    scene_folder = os.path.dirname(os.path.abspath(scene_path))
    
    output_path = os.path.join(scene_folder, 
                              f"{dataset_name}_tx{args.tx_orientations}_rx{args.rx_orientations}")

    # Setup output directory and backup config
    os.makedirs(output_path, exist_ok=True)
    copyfile(args.config_file, os.path.join(output_path, "config.yml"))

    # Load simulation config
    simu_config = load_cfg(config_file=args.config_file)

    # Generate grid positions
    if args.rx_gen == 'grid':
        grid_positions = generate_l_room_positions(target_points=args.target_points, 
                                                  margin=args.margin)
    else:
        # For random mode, generate random positions within L-room bounds
        grid_positions = []
        while len(grid_positions) < args.target_points:
            # Random point in bounding box
            x = np.random.uniform(0, 4)
            y = np.random.uniform(0, 3)
            z = np.random.uniform(-2, 2)
            
            if is_point_in_l_room([x, y, z], args.margin):
                grid_positions.append([x, y, z])
        
        grid_positions = np.array(grid_positions)

    print(f"Generated {len(grid_positions)} positions")

    # Visualize if requested
    if args.visualize:
        visualize_l_room_grid(grid_positions)

    # Generate orientations
    tx_orientations = generate_orientations(args.tx_orientations, args.tx_ori_mode)
    rx_orientations = generate_orientations(args.rx_orientations, args.rx_ori_mode)

    # Create TX/RX combinations
    if args.tx_gen == 'grid':
        tx_positions = grid_positions
    elif args.tx_gen == 'center':
        tx_positions = np.array([[2.0, 1.5, 0.0]])  # Center of L-room
    elif args.tx_gen == 'random':
        tx_positions = grid_positions[:1]  # Use first grid position

    # Create all combinations
    all_tx_poses = []
    all_tx_oris = []
    for tx_pos in tx_positions:
        for tx_ori in tx_orientations:
            all_tx_poses.append(tx_pos)
            all_tx_oris.append(tx_ori)

    all_rx_poses = []
    all_rx_oris = []
    for rx_pos in grid_positions:
        for rx_ori in rx_orientations:
            all_rx_poses.append(rx_pos)
            all_rx_oris.append(rx_ori)

    all_rx_poses = np.array(all_rx_poses)
    all_rx_oris = np.array(all_rx_oris)

    print(f"Total TX configurations: {len(all_tx_poses)}")
    print(f"Total RX configurations: {len(all_rx_poses)}")
    print(f"Total RIRs to generate: {len(all_tx_poses)} × {len(all_rx_poses)} = {len(all_tx_poses) * len(all_rx_poses)}")

    # Save configuration
    config_data = {
        "l_room_config": L_ROOM_CONFIG,
        "tx_positions": [list(pos) for pos in all_tx_poses],
        "tx_orientations": [list(ori) for ori in all_tx_oris],
        "rx_positions": [list(pos) for pos in all_rx_poses],
        "rx_orientations": [list(ori) for ori in all_rx_oris],
        "simulation_params": {
            "target_points": args.target_points,
            "margin": args.margin,
            "tx_gen": args.tx_gen,
            "rx_gen": args.rx_gen
        }
    }
    with open(os.path.join(output_path, 'simulation_config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)

    # Main simulation loop
    rir_count = 0
    for tx_ind, (tx_pos, tx_ori) in enumerate(tqdm(zip(all_tx_poses, all_tx_oris), 
                                                  desc="Processing TX positions", 
                                                  total=len(all_tx_poses))):
        tx_pos = np.array(tx_pos)
        tx_ori = np.array(tx_ori) / np.linalg.norm(tx_ori)
        
        # Calculate prefix for unique file naming
        prefix = tx_ind * len(all_rx_poses)
        
        # Run IR simulation from this TX to all RX positions
        ir_time_all, rx_pos_used, rx_ori_used = ir_simulation(
            scene_path=scene_path, 
            rx_pos=all_rx_poses, 
            tx_pos=tx_pos,
            rx_ori=all_rx_oris, 
            tx_ori=tx_ori, 
            simu_config=simu_config
        )

        rir_count += len(ir_time_all)
        
        # Output results
        if not args.save_data:
            plot_figure(ir_samples=ir_time_all, rx_pos=rx_pos_used, tx_pos=tx_pos,
                       rx_ori=rx_ori_used, tx_ori=tx_ori, fs=simu_config['fs'])
        else:
            save_ir(ir_samples=ir_time_all, rx_pos=rx_pos_used, rx_ori=rx_ori_used,
                   tx_pos=tx_pos, tx_ori=tx_ori, save_path=output_path,
                   prefix=prefix, fs=simu_config['fs'])

        expected_rirs = len(all_rx_poses)
        actual_rirs = len(ir_time_all)
        
        if actual_rirs != expected_rirs:
            print(f"WARNING: TX {tx_ind + 1}/{len(all_tx_poses)} at {tx_pos}: Expected {expected_rirs} RIRs, got {actual_rirs}")
        else:
            print(f"TX {tx_ind + 1}/{len(all_tx_poses)}: Generated {actual_rirs} RIRs from TX at {tx_pos}")

    print(f"\nFinal summary:")
    print(f"Total RIRs generated: {rir_count}")
    print(f"Expected total: {len(all_tx_poses) * len(all_rx_poses)}")