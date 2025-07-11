import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import json
import numpy as np
import argparse
from tqdm import tqdm
from shutil import copyfile

from simu_utils import ir_simulation, plot_figure, save_ir, load_cfg, generate_rx_samples, generate_grid
from config import *

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

def generate_orientations(num_orientations=4, mode='random'):
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

def apply_jitter(positions, jitter_std=0.05, num_jittered=3, xyz_min=None, xyz_max=None):
    """Apply random jitter to grid positions
    
    Args:
        positions: numpy array of original positions [N, 3]
        jitter_std: standard deviation of jitter in meters
        num_jittered: number of jittered positions per original position
        xyz_min, xyz_max: room bounds to ensure jittered positions stay in bounds
    
    Returns:
        jittered_positions: numpy array including original + jittered positions
    """
    if num_jittered == 0:
        return positions
    
    original_positions = positions.copy()
    jittered_positions = [original_positions]
    
    for jitter_idx in range(num_jittered):
        # Generate random jitter for all positions
        jitter = np.random.normal(0, jitter_std, positions.shape)
        new_positions = original_positions + jitter
        
        # Clamp to room bounds if provided
        if xyz_min is not None and xyz_max is not None:
            new_positions = np.clip(new_positions, xyz_min, xyz_max)
        
        jittered_positions.append(new_positions)
    
    # Combine all positions
    all_positions = np.concatenate(jittered_positions, axis=0)
    
    print(f"Applied jitter: {len(original_positions)} original -> {len(all_positions)} total positions")
    print(f"Jitter parameters: std={jitter_std:.3f}m, num_jittered={num_jittered}")
    
    return all_positions

def generate_grid_positions(xyz_min, xyz_max):
    """Generate receiver positions in a 3D grid based on config.py settings"""
    positions = []
    
    # Calculate actual usable space (accounting for margins)
    usable_min_x = xyz_min[0] + NODE_MARGIN
    usable_max_x = xyz_max[0] - NODE_MARGIN
    usable_min_y = xyz_min[1] + NODE_MARGIN
    usable_max_y = xyz_max[1] - NODE_MARGIN
    usable_min_z = xyz_min[2] + NODE_MARGIN
    usable_max_z = xyz_max[2] - NODE_MARGIN
    
    # Generate grid positions
    for i in range(GRID_N_X):
        for j in range(GRID_N_Y):
            for k in range(GRID_N_Z):
                if GRID_N_X == 1:
                    x = (usable_min_x + usable_max_x) / 2
                else:
                    x = usable_min_x + i * (usable_max_x - usable_min_x) / (GRID_N_X - 1)
                
                if GRID_N_Y == 1:
                    y = (usable_min_y + usable_max_y) / 2
                else:
                    y = usable_min_y + j * (usable_max_y - usable_min_y) / (GRID_N_Y - 1)
                
                if GRID_N_Z == 1:
                    z = (usable_min_z + usable_max_z) / 2
                else:
                    z = usable_min_z + k * (usable_max_z - usable_min_z) / (GRID_N_Z - 1)
                
                positions.append([x, y, z])
    
    return np.array(positions)

def get_scene_bounds(scene):
    """Get scene boundary coordinates"""
    bounds = {
        'Avonia': ([-3, -1, -6], [5, 3, 3]),
        'Montreal': ([-9, -1, -6], [2, 3, 1]),
        'EmptyRoom': ([-2.5, 0.25, -2.5], [2.5, 2.75, 2.5]),
        'SmallRoom': ([-2, -1, -2], [2, 2, 2]),
        'LRoom': ([-2, -1, -2], [4, 3, 2]),
    }
    return bounds.get(scene, ([-3, -1, -3], [3, 4, 3]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AcoustiX impulse response simulation")
    parser.add_argument('--save-data', action="store_true", help='save the simulated impulse response data')
    parser.add_argument('--scene', default='EmptyRoom', choices=['EmptyRoom', 'Avonia', 'Montreal', 'SmallRoom', 'LRoom'], 
                       help='scene to simulate')
    parser.add_argument('--rx-gen', default='random', choices=['random', 'grid'], 
                       help='receiver generation mode: random or grid')
    parser.add_argument('--tx-gen', default='Default', choices=['Default', 'Random', 'Custom'], 
                       help='transmitter generation mode')
    
    # Jitter arguments
    parser.add_argument('--jitter', action='store_true', help='enable position jittering')
    parser.add_argument('--jitter-std', type=float, default=0.05, help='standard deviation of jitter in meters')
    parser.add_argument('--num-jittered', type=int, default=9, help='number of jittered positions per original position')
    parser.add_argument('--tx-jitter', action='store_true', help='apply jitter to transmitter positions (default: only RX)')
    
    # Orientation arguments
    parser.add_argument('--tx-orientations', type=int, default=1, help='number of orientations per transmitter position')
    parser.add_argument('--rx-orientations', type=int, default=1, help='number of orientations per receiver position')
    parser.add_argument('--tx-ori-mode', default='random', choices=['random', 'uniform', 'cardinal'], 
                       help='transmitter orientation generation mode')
    parser.add_argument('--rx-ori-mode', default='random', choices=['random', 'uniform', 'cardinal'], 
                       help='receiver orientation generation mode')
    
    parser.add_argument('--config-file', default="./simu_config/basic_config.yml", 
                       help='config file for the impulse response rendering')
    parser.add_argument('--num-samples', type=int, default=50, help='num of simulated impulse response')
    parser.add_argument('--num-batch', type=int, default=27, help='num of IR per batch to be simulated')
    
    parser.add_argument('--scene-file', default=None, help='path to the scene XML file (auto-generated if not provided)')
    parser.add_argument('--dataset-name', default=None, help='dataset name (auto-generated if not provided)')

    args = parser.parse_args()
    
    # Setup paths
    scene_path = args.scene_file or f'./extract_scene/{args.scene}/{args.scene}.xml'
    dataset_name = args.dataset_name or f'../../RIR/{args.scene}'
    scene_folder = os.path.dirname(os.path.abspath(scene_path))
    
    # Modify output path based on rx_gen mode, orientations, and jitter
    if args.rx_gen == 'grid':
        base_name = f"{dataset_name}_grid_{GRID_N_X}x{GRID_N_Y}x{GRID_N_Z}_tx{args.tx_orientations}_rx{args.rx_orientations}"
        if args.jitter:
            base_name += f"_jitter{args.num_jittered}_{args.jitter_std:.3f}"
        output_path = os.path.join(scene_folder, base_name)
    else:
        base_name = f"{dataset_name}_tx{args.tx_orientations}_rx{args.rx_orientations}"
        if args.jitter:
            base_name += f"_jitter{args.num_jittered}_{args.jitter_std:.3f}"
        output_path = os.path.join(scene_folder, base_name)

    # Setup output directory and backup config
    os.makedirs(output_path, exist_ok=True)
    copyfile(args.config_file, os.path.join(output_path, "config.yml"))

    # Load simulation config
    simu_config = load_cfg(config_file=args.config_file)

    # Get scene boundaries
    xyz_min, xyz_max = get_scene_bounds(args.scene)

    if args.rx_gen == 'grid':
        # Generate the base grid positions
        base_grid_positions = generate_grid_positions(xyz_min, xyz_max)
        print(f"Generated {len(base_grid_positions)} base grid positions")
        print(f"Grid configuration: {GRID_N_X}x{GRID_N_Y}x{GRID_N_Z} = {TOTAL_NODES} positions")
        
        # Apply jitter to create more positions
        if args.jitter:
            # Add some margin to bounds for jitter clamping
            jitter_xyz_min = np.array(xyz_min) + NODE_MARGIN/2
            jitter_xyz_max = np.array(xyz_max) - NODE_MARGIN/2
            
            grid_positions = apply_jitter(
                base_grid_positions, 
                jitter_std=args.jitter_std, 
                num_jittered=args.num_jittered,
                xyz_min=jitter_xyz_min,
                xyz_max=jitter_xyz_max
            )
        else:
            grid_positions = base_grid_positions

        if args.rx_gen == 'grid' and args.jitter:
            # Store original positions for the 4-view plots
            original_grid_positions = base_grid_positions
        else:
            original_grid_positions = None
        
        # Generate orientations
        tx_orientations = generate_orientations(args.tx_orientations, args.tx_ori_mode)
        rx_orientations = generate_orientations(args.rx_orientations, args.rx_ori_mode)
        
        print(f"TX orientations: {len(tx_orientations)}")
        print(f"RX orientations: {len(rx_orientations)}")
        
        # Create TX positions (with optional jitter)
        if args.tx_jitter and args.jitter:
            tx_grid_positions = apply_jitter(
                base_grid_positions,
                jitter_std=args.jitter_std,
                num_jittered=args.num_jittered,
                xyz_min=jitter_xyz_min,
                xyz_max=jitter_xyz_max
            )
        else:
            tx_grid_positions = base_grid_positions
        
        # Create all TX position/orientation combinations
        all_tx_poses = []
        all_tx_oris = []
        for tx_pos in tx_grid_positions:
            for tx_ori in tx_orientations:
                all_tx_poses.append(tx_pos)
                all_tx_oris.append(tx_ori)
        
        # Create all RX position/orientation combinations  
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

        # Save configuration including jitter info
        config_data = {
            "simulation_config": {
                "scene": args.scene,
                "rx_gen": args.rx_gen,
                "tx_gen": args.tx_gen,
                "grid_config": {
                    "GRID_N_X": GRID_N_X,
                    "GRID_N_Y": GRID_N_Y,
                    "GRID_N_Z": GRID_N_Z,
                    "base_positions": len(base_grid_positions)
                },
                "jitter_config": {
                    "enabled": args.jitter,
                    "jitter_std": args.jitter_std,
                    "num_jittered": args.num_jittered,
                    "tx_jitter": args.tx_jitter,
                    "final_rx_positions": len(all_rx_poses),
                    "final_tx_positions": len(all_tx_poses)
                },
                "orientation_config": {
                    "tx_orientations": args.tx_orientations,
                    "rx_orientations": args.rx_orientations,
                    "tx_ori_mode": args.tx_ori_mode,
                    "rx_ori_mode": args.rx_ori_mode
                }
            },
            "speaker": {
                "positions": [list(pos) for pos in all_tx_poses],
                "orientations": [list(ori) for ori in all_tx_oris]
            }
        }
        with open(os.path.join(output_path, 'simulation_config.json'), 'w') as json_file:
            json.dump(config_data, json_file, indent=4)

        # Main simulation loop - for each TX, simulate to all RX positions
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
                       prefix=prefix, fs=simu_config['fs'], roomname=args.scene, original_positions=original_grid_positions)

            expected_rirs = len(all_rx_poses)
            actual_rirs = len(ir_time_all)
            
            if actual_rirs != expected_rirs:
                print(f"WARNING: TX {tx_ind + 1}/{len(all_tx_poses)} at {tx_pos}: Expected {expected_rirs} RIRs, got {actual_rirs}")
            else:
                print(f"TX {tx_ind + 1}/{len(all_tx_poses)}: Generated {actual_rirs} RIRs from TX at {tx_pos}")

        print(f"\nFinal summary:")
        print(f"Total RIRs generated: {rir_count}")
        print(f"Expected total: {len(all_tx_poses) * len(all_rx_poses)}")

    else:
        # Random mode - existing logic stays the same
        n_random_samples = args.num_batch
        num_sample = args.num_samples
        num_iter = int(num_sample / n_random_samples)
        
        # Generate single TX position for random mode
        if args.tx_gen == 'Default':
            tx_pos = [abs(xyz_max[0]) - abs(xyz_min[0]),
                     abs(xyz_max[1]) - abs(xyz_min[1]),
                     abs(xyz_max[2]) - abs(xyz_min[2])]
        elif args.tx_gen == 'Random':
            tx_pos = [np.random.uniform(xyz_min[i], xyz_max[i]) for i in range(3)]
        
        tx_ori = generate_orientations(1, args.tx_ori_mode)[0]
        
        pbar = tqdm(range(num_iter), desc=f"TX pos: {tx_pos}")
        for iter in pbar:
            prefix = iter * n_random_samples
            
            # Generate receiver positions and orientations
            rx_pos, rx_ori = generate_rx_samples(n_random_samples=n_random_samples, 
                                                xyz_max=xyz_max, xyz_min=xyz_min)

            # Apply jitter to random positions if enabled
            if args.jitter:
                rx_pos = apply_jitter(rx_pos, args.jitter_std, args.num_jittered, xyz_min, xyz_max)

            # Run IR simulation
            ir_time_all, rx_pos, rx_ori = ir_simulation(
                scene_path=scene_path, rx_pos=rx_pos, tx_pos=tx_pos,
                rx_ori=rx_ori, tx_ori=tx_ori, simu_config=simu_config
            )

            # Output results
            if not args.save_data:
                plot_figure(ir_samples=ir_time_all, rx_pos=rx_pos, tx_pos=tx_pos,
                           rx_ori=rx_ori, tx_ori=tx_ori, fs=simu_config['fs'])
            else:
                save_ir(ir_samples=ir_time_all, rx_pos=rx_pos, rx_ori=rx_ori,
                       tx_pos=tx_pos, tx_ori=tx_ori, save_path=output_path,
                       prefix=prefix, fs=simu_config['fs'], roomname=args.scene, original_positions=original_grid_positions)