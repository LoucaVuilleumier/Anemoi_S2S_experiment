#!/bin/bash

# The job name
#SBATCH --job-name=rerun_failed_inference_all_members

# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=./slurm_scripts/output_slurm/rerun_failed_inference-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/rerun_failed_inference-%J.out

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

export HYDRA_FULL_ERROR=1

source /ec/res4/hpcperm/nld4584/anemoi_python_3_11_ag_2026_23_04/.venv/bin/activate

# Full path to the YAML config file
CONFIG_FILE="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/Inference/inference_aifs_subs_pretrain.yaml"

# Initialization date to rerun
DATE="2011-07-21"
BASE_DATE="${DATE}T12:00"
OUTPUT_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/AIFS/aifs-subs-pretrain-${DATE}"

echo "============================================"
echo "Rerunning inference for ${DATE}"
echo "Running all members (0-7)"
echo "============================================"
echo ""

# Remove all existing member files for this date
echo "Cleaning up existing files in ${OUTPUT_DIR}..."
rm -f ${OUTPUT_DIR}/member-*.nc
mkdir -p ${OUTPUT_DIR}
echo ""

# Run inference for all 8 members
for member in {0..7}; do
    export ANEMOI_BASE_SEED=${member}
    OUTPUT_FILE="${OUTPUT_DIR}/member-${member}.nc"
    export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
    export INFERENCE_DATE=${BASE_DATE}
    
    echo "============================================"
    echo "Running inference for ${BASE_DATE} (member ${member})"
    echo "Output: ${OUTPUT_FILE}"
    echo "============================================"
    
    anemoi-inference run "${CONFIG_FILE}"
    
    # Check if the file was created successfully
    if [ -f "${OUTPUT_FILE}" ] && [ -s "${OUTPUT_FILE}" ]; then
        echo "✓ Successfully created ${OUTPUT_FILE} (size: $(du -h ${OUTPUT_FILE} | cut -f1))"
    else
        echo "✗ ERROR: Failed to create ${OUTPUT_FILE} or file is empty"
    fi
    echo ""
done

echo "============================================"
echo "Rerun completed for all members!"
echo "============================================"
