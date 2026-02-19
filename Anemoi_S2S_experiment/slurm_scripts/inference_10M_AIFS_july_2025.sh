#!/bin/bash

# The job name
#SBATCH --job-name=inference_10M_AIFS_july_2025

# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=./slurm_scripts/output_slurm/inference_10M_AIFS_july_2025.out
#SBATCH --error=./slurm_scripts/output_slurm/inference_10M_AIFS_july_2025.out

#SBATCH --array=0-9
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

export ANEMOI_BASE_SEED=$SLURM_ARRAY_TASK_ID
export HYDRA_FULL_ERROR=1

source /ec/res4/hpcperm/nld4584/anemoi_core_2026_16_02/venv_anemoi_core_2026_16_02/bin/activate


OUTPUT_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/AIFS/aifs-subs-pretrain-july-2025"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
# Create output filename with member number
OUTPUT_FILE="${OUTPUT_DIR}-member-${SLURM_ARRAY_TASK_ID}.nc"
# Export environment variable for this run
export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}

anemoi-inference run "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/Inference/inference_aifs_subs_pretrain.yaml"