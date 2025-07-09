# config.py
import numpy as np

# Room Configuration
ROOMNAME = "LRoom"
ROOM_MIN = [-1, -1, -3]  # X, Y, Z minimum coordinates
ROOM_MAX = [5, 4, 1]  # X, Y, Z maximum coordinates

#ROOM_MIN = [-2.5, 0.25, -2.5]
#ROOM_MAX = [2.5, 2.75, 2.5]

# Improved Grid Configuration
GRID_N_X = 7  # X-axis divisions
GRID_N_Y = 3  # Y-axis divisions (height - fewer needed)
GRID_N_Z = 7  # Z-axis divisions
TOTAL_NODES = GRID_N_X * GRID_N_Y * GRID_N_Z  # 147 nodes

NODE_MARGIN = 0.35  # meters

# Calculate spacing
ROOM_SIZE_X = ROOM_MAX[0] - ROOM_MIN[0]  # 5.0m
ROOM_SIZE_Y = ROOM_MAX[1] - ROOM_MIN[1]  # 2.5m  
ROOM_SIZE_Z = ROOM_MAX[2] - ROOM_MIN[2]  # 5.0m

NODE_SPACING_X = ROOM_SIZE_X / (GRID_N_X - 1)  # ~0.83m
NODE_SPACING_Y = ROOM_SIZE_Y / (GRID_N_Y - 1)  # ~1.25m
NODE_SPACING_Z = ROOM_SIZE_Z / (GRID_N_Z - 1)  # ~0.83m

ROOM_SHAPES = {
    'rectangular': {
        'interior_bounds': [ROOM_MIN, ROOM_MAX],
        'description': 'Standard rectangular room'
    },
    'L-shaped': {
        'interior_bounds': {
            'part1': [[0, 0, 0], [4, 3, 2]],  # X 0:4, Z 0:2, Y 0:3
            'part2': [[2, 0, -2], [4, 3, 0]]  # X 2:4, Z -2:0, Y 0:3
        },
        'description': 'L-shaped room with 1m thick walls'
    }
}

# Audio Configuration  
FS = 48000
RIR_LEN = 4800

# Training Configuration (adjusted for larger grid)
HIDDEN_DIM = 128
NUM_MESSAGE_PASSING_ROUNDS = 3  # More rounds for larger grid
NUM_EPOCHS = 200  # Fewer epochs needed with better data
BATCH_SIZE = 49  # 7×7 subset batches
LEARNING_RATE = 5e-4  # Lower LR for stability

# Learning Rate Scheduler
WARMUP_EPOCHS = 10
MIN_LR_FACTOR = 0.05

# Training Phases
TIME_ONLY_EPOCHS = NUM_EPOCHS // 5
ADD_SPEC_EPOCHS = NUM_EPOCHS // 3

# RIR Peak Configuration
PEAK_SAMPLES = 960  # Peak region samples
DECAY_SAMPLES = 1920  # Decay region samples
LATE_SAMPLES = RIR_LEN  - DECAY_SAMPLES  # Late region samples

# Loss Weights
LAMBDA_SPEC = 1.0
LAMBDA_AMP = 0.5
LAMBDA_PHASE = 0.5
LAMBDA_TIME = 20

# Model Selection Options
AVAILABLE_MODELS = [
    "GCN_Simple",
    "GCN_DistanceAware",
    "GCN_SourceLearning", 
    "GCN_SimpleTransform",
    "GCN_SimpleFluid",
    "GCN_FluidPeakAware",
    "Custom_MessagePassing"
]

# Helper functions for dynamic paths
def get_data_dir(dataset_name):
    return f"./data/{dataset_name}/RIR"

def get_models_save_dir(dataset_name):
    return f"./models/{dataset_name}"

def get_inference_output_dir(dataset_name):
    return f"./inference/{dataset_name}"