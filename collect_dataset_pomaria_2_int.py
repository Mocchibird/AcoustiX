import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import tensorflow as tf
gpu_num = "0" # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1) # Set global random seed for reproducibility
    

import json
import numpy as np
from tqdm import tqdm
import yaml
from simu_utils import ir_simulation, plot_figure, save_ir, load_cfg, generate_rx_samples, generate_rx_samples_path, save_ir_raw
from shutil import copyfile
import argparse
import ast
import soundfile as sf


def parse_path(s):
    """
    Parse the --path argument.
   
    If the string is "random", return it unchanged. Otherwise,
    expect a string formatted like:
       [5,1,1],[1,2,3],[3,2,1]
    and convert that into a list of waypoints.
    """
    if s == "random":
        return s
    else:
        # We expect the user input to be a series of comma-separated lists.
        # Wrap the entire string in extra square brackets to make it a list of lists.
        if not s.strip().startswith("[["):
            s = "[" + s + "]"
        try:
            path_points = ast.literal_eval(s)
        except Exception as e:
            raise argparse.ArgumentTypeError("Invalid path format: " + str(e))
       
        if not isinstance(path_points, list):
            raise argparse.ArgumentTypeError("Path must be provided as a list of coordinates.")
       
        for pt in path_points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 3:
                raise argparse.ArgumentTypeError("Each coordinate must be a list or tuple of three numbers.")
        return path_points


def parse_boundaries(s):
    """
    Parse the --boundaries argument.
   
    Expects a string formatted like:
       [[-4,-4,-1],[3,5,4]],[[1,0,0],[5,2,3]]
    which represents multiple boundary boxes, each with min and max coordinates.
    """
    try:
        boundaries = ast.literal_eval(s)
       
        # Validate format: list of pairs, each pair has two lists of 3 values
        if not isinstance(boundaries, list):
            raise ValueError("Boundaries must be a list of min/max pairs")
       
        for box in boundaries:
            if not isinstance(box, list) or len(box) != 2:
                raise ValueError("Each boundary must be a pair of min/max points")
            for point in box:
                if not isinstance(point, list) or len(point) != 3:
                    raise ValueError("Each point must be a list of 3 coordinates")
                   
        return boundaries
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid boundaries format: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Help for simulation files.")
    parser.add_argument('--save-data', action="store_true", help='save the simulated impulse response data')


    parser.add_argument('--config-file', default="./simu_config/basic_config.yml", help='config file for the impulse response rendering')
    parser.add_argument('--num-samples', default=10000, help='num of simulated impulse response')
    parser.add_argument('--num-batch', default=20, help='num of IR per batch to be simulated')
   
    parser.add_argument('--scene-file', default='./extract_scene/Pomaria_2_int/output_scene.xml', help='path to the scene XML file')
    parser.add_argument('--dataset-name', default='test', help='dataset name')


    parser.add_argument('--path', type=parse_path, default='random',
                        help='Path as list of waypoints, e.g. --path [5,1,1],[1,2,3],[3,2,1]. "random" uses random sample generation.')
    parser.add_argument('--boundaries', type=parse_boundaries,
                        default='[[[-4,-4,-1],[3,5,4]]]',
                        help='List of boundary boxes as: [[min1,max1],[min2,max2],...] where each min/max is [x,y,z]')
   
    args = parser.parse_args()


    save_data = args.save_data
    show_image = not save_data
    config_file = args.config_file


    path = args.path


    # output path setup
    dataset_name = args.dataset_name
    scene_path = args.scene_file
    scene_folder = os.path.dirname(os.path.abspath(scene_path))
    output_path = os.path.join(scene_folder, dataset_name)


    # back up the simulation config
    os.makedirs(output_path, exist_ok=True)
    copyfile(config_file,  os.path.join(output_path, "config.yml"))


    ## simulation setup
    simu_config = load_cfg(config_file=config_file)


    # scene boundary setup
    xyz_min = [-4,-4,-1]
    xyz_max = [3,5,4]


    # Parse boundary boxes
    boundaries = args.boundaries
   
    # For backward compatibility, create a default boundary if not provided
    if not boundaries:
        boundaries = [[[4.2,2.4,3],[-1.1,0,-2.06]],[[2.3,2.4,-2.06],[-1.06,0,-5.71]]]


    # simulation number
    n_random_samples = args.num_batch # each simulation batch simulation these random samples
    num_sample = args.num_samples # total simulation samples
    num_iter = int(num_sample / n_random_samples)


    # this example shows the fixed tx positions
    tx_poses = [[1,1,-5]]
    tx_oris = [[-0.5, 0, -0.8660254]]


    data = {"speaker": {"positions": tx_poses, "orientations": tx_oris}}
    with open(os.path.join(output_path, 'speaker_data.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)


    for tx_ind, (tx_pos, tx_ori) in enumerate(zip(tx_poses, tx_oris)):
        tx_pos = np.array(tx_pos)
        tx_ori = np.array(tx_ori)
        tx_ori /= np.linalg.norm(tx_ori)
       
        pbar = tqdm(range(0, num_iter))
        for iter in pbar:
            prefix = (tx_ind+1) * n_random_samples * iter
            pbar.set_description(f"tx pos: {str(tx_pos)}")
           
            # randomly generate samples
            if path != "random":
                rx_pos, rx_ori = generate_rx_samples_path(path, n_random_samples)
            else:
                # Pass boundaries to the generate_rx_samples function
                rx_pos, rx_ori = generate_rx_samples(n_random_samples=n_random_samples, boundaries=boundaries)


            # ir simulations
            ir_time_all, rx_pos, rx_ori = ir_simulation(scene_path=scene_path, rx_pos=rx_pos, tx_pos=tx_pos, rx_ori=rx_ori, tx_ori=tx_ori, simu_config=simu_config)


            # save or show the ir samples
            if show_image == True:
                plot_figure(ir_samples=ir_time_all, rx_pos=rx_pos, tx_pos=tx_pos, rx_ori=rx_ori, tx_ori=tx_ori, fs=simu_config['fs'])


            if save_data == True:                
                save_ir_raw(ir_samples=ir_time_all, rx_pos=rx_pos, tx_pos=tx_pos, rx_ori=rx_ori, tx_ori=tx_ori, save_path=output_path, prefix=prefix, sample_rate=simu_config['fs'])

print("RIRs saved successfully!")