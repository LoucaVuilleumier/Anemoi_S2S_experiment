#!/bin/bash

# The job name
#SBATCH --job-name=run_12_inference_runs_10k_rollout

# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=./slurm_scripts/output_slurm/run_12_inference_runs_10k_rollout-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/run_12_inference_runs_10k_rollout-%J.out

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

export ANEMOI_BASE_SEED=42
export HYDRA_FULL_ERROR=1
source /ec/res4/hpcperm/nld4584/anemoi_core_2025_17_11/venv_anemoi_core_2025_17_11/bin/activate

# Full path to the YAML config file
CONFIG_FILE="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new/Inference/inference_8weeks_12runs_summer2022.yaml"

CHECKPOINT_PATH="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_training/checkpoint/c682ce55de1142e38269c45fc68c2511/inference-last.ckpt"
export CHECKPOINT_PATH=${CHECKPOINT_PATH}
# Base date for inference runs (2022-05-16)
BASE_DATE="2022-05-16"
OUTPUT_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}


# Run 12 inference runs with 1-week offset each
for i in {0..11}; do
    # Calculate the offset in days (i weeks * 7 days)
    offset_days=$((i * 7))
    
    # Calculate the new date (requires GNU date)
    INFERENCE_DATE=$(date -d "${BASE_DATE} + ${offset_days} days" +"%Y-%m-%dT%H:%M")
    
    # Create output filename with run number and date
    OUTPUT_FILE="${OUTPUT_DIR}/output_inference_10k_rollout_8weeks_run$(printf "%02d" $i)_${INFERENCE_DATE:0:10}.nc"
    
    # Export environment variables for this run
    export INFERENCE_DATE
    export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
    
    echo "========================================"
    echo "Starting inference run $((i+1))/12"
    echo "Date: ${INFERENCE_DATE}"
    echo "Output: ${OUTPUT_FILE}"
    echo "========================================"
    
    # Run the inference directly (no srun needed, already in slurm job)
    # Using single task to avoid multiple processes writing to same file
    anemoi-inference run ${CONFIG_FILE}
    
    # Check if the run was successful
    if [ $? -eq 0 ]; then
        echo "Run $((i+1))/12 completed successfully"
    else
        echo "ERROR: Run $((i+1))/12 failed!"
        exit 1
    fi
done

echo "========================================"
echo "All 12 inference runs completed!"
echo "========================================"