#!/bin/bash

# The job name
#SBATCH --job-name=inference_8M_AIFS_2007-2008

# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=./slurm_scripts/output_slurm/inference_8M_AIFS_2007-2008.out
#SBATCH --error=./slurm_scripts/output_slurm/inference_8M_AIFS_2007-2008.out

#SBATCH --array=0-7
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

# Full path to the YAML config file
CONFIG_FILE="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/Inference/inference_aifs_subs_pretrain.yaml"

# Automatically generate all first the 15th for each month from 2007-2011
# This creates 120 forecast start dates (5 years × 12 months × 2 starts per month)

for year in {2007..2009}; do
    for month in 7; do
        # Find the first day of the month
        first_day=$(printf "%04d-%02d-01" $year $month)
        
        # Calculate fifteenth day (14 days after first day)
        fifteenth=$(date -d "$first_day + 14 days" +%Y-%m-%d)
        
        # Run inference for first day
        BASE_DATE="${first_day}T12:00"
        OUTPUT_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/AIFS/aifs-subs-pretrain-${first_day}"
        mkdir -p ${OUTPUT_DIR}
        OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}.nc"
        export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
        export INFERENCE_DATE=${BASE_DATE}
        echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
        anemoi-inference run "${CONFIG_FILE}"
        
        # Run inference for fifteenth day
        BASE_DATE="${fifteenth}T12:00"
        OUTPUT_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/AIFS/aifs-subs-pretrain-${fifteenth}"
        mkdir -p ${OUTPUT_DIR}
        OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}.nc"
        export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
        export INFERENCE_DATE=${BASE_DATE}
        echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
        anemoi-inference run "${CONFIG_FILE}"
    done
done