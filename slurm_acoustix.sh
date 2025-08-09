#!/bin/bash         
#SBATCH --mem=64G                        
#SBATCH --mail-type=END                                                           
#SBATCH --time=24:00:00
#SBATCH --nodelist=tikgpu06                                
#SBATCH --output=/itet-stor/changh/net_scratch/master_thesis/AcoustiX/jobs/acoustix_%j.out # Output file
#SBATCH --error=/itet-stor/changh/net_scratch/master_thesis/AcoustiX/jobs/acoustix_%j.err  # Error file

ETH_USERNAME=changh
PROJECT_NAME=master_thesis
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
CONDA_ENVIRONMENT=acoustix

# Create jobs directory if it doesn't exist
mkdir -p ${DIRECTORY}/jobs

# Set SLURM configuration as required by ETH TIK cluster
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change to temporary directory
cd "${TMPDIR}" || exit 1


# Activate conda environment (adjust path to your conda installation)
[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"

# Force CPU-only execution for all frameworks
export CUDA_VISIBLE_DEVICES=""
export TF_FORCE_GPU_ALLOW_GROWTH=false
export TF_CPP_MIN_LOG_LEVEL=1

# Change to AcoustiX directory
cd ${DIRECTORY}/AcoustiX

# Verify environment
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "CPU cores available: 20"
echo "MI_VARIANT: $MI_VARIANT"
echo "DRJIT_VARIANT: $DRJIT_VARIANT"

# CPU-only TensorFlow test
echo "TensorFlow CPU test..."
python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('CPU devices:', len(tf.config.list_physical_devices('CPU')))
print('GPU devices (should be empty):', len(tf.config.list_physical_devices('GPU')))
print('TensorFlow CPU: Ready!')
"

# Set environment variables for better CPU performance
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "=== Starting AcoustiX Data Generation (CPU Mode) ==="

# Dataset Generation: L-Room with adaptive grid + jittering
echo "Starting L-Room dataset generation with adaptive grid + jittering..."
python collect_dataset_LRoom.py \
    --save-data \
    --rx-gen grid \
    --tx-gen grid \
    --target-points 50 \
    --jitter \
    --jitter-std 0.05 \
    --num-jittered 9 \
    --tx-orientations 1 \
    --rx-orientations 1 \
    --margin 0.2 \
    --config-file ./simu_config/basic_config.yml

echo "L-Room dataset generation completed."

# Optional: Generate visualization
echo "Generating L-Room grid visualization..."
python collect_dataset_LRoom.py \
    --visualize \
    --rx-gen grid \
    --target-points 50 \
    --margin 0.2

echo "=== Data Generation Summary ==="
echo "Generated datasets:"
echo "L-Room: 50 TX positions (grid only) × 500 RX positions (grid + jitter) = 25,000 RIRs"

# List generated datasets
echo ""
echo "Generated L-Room dataset directories:"
find ${DIRECTORY}/AcoustiX/extract_scene -type d -name "*LRoom*" -o -name "*jitter*" 2>/dev/null | head -10

# Show disk usage
echo ""
echo "L-Room dataset disk usage:"
du -sh ${DIRECTORY}/AcoustiX/extract_scene/*/LRoom_* 2>/dev/null | head -5

# Send completion info to output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
