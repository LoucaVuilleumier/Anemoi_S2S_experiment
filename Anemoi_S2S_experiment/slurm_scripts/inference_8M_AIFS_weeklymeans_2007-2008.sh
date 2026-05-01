#!/bin/bash

# The job name
#SBATCH --job-name=inference_8M_AIFS_weeklymeans_2007-2008

# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=./slurm_scripts/output_slurm/inference_8M_AIFS_weeklymeans_72k-_lr_0.625e-7_2007-2008.out
#SBATCH --error=./slurm_scripts/output_slurm/inference_8M_AIFS_weeklymeans_72k-_lr_0.625e-7_2007-2008.out

#SBATCH --array=0-7
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00

export ANEMOI_BASE_SEED=$SLURM_ARRAY_TASK_ID
export HYDRA_FULL_ERROR=1

source /ec/res4/hpcperm/nld4584/anemoi_python3_11_ag_2026_16_03/.venv/bin/activate

# Full path to the YAML config file
CONFIG_FILE="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/Inference/inference_aifs_weekly_means.yaml"

# Automatically generate all first, 11 and 21 forMJJAS from 2007-2011
# This creates 75 forecast start dates (5 years × 5 months × 3 starts per month)

for year in {2007..2011}; do
    for month in {5..9}; do
        # Find the first day of the month
        first_day=$(printf "%04d-%02d-01" $year $month)
        
        # Calculate eleventh day (10 days after first day)
        eleventh=$(date -d "$first_day + 10 days" +%Y-%m-%d)
        
        # Calculate twenty-first day (20 days after first day)
        twenty_first=$(date -d "$first_day + 20 days" +%Y-%m-%d)
        
        # Run inference for first day
        BASE_DATE="${first_day}T12:00"
        OUTPUT_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/Weekly_Means_72k_lr_0.625e-7/aifs-subs-pretrain-weeklymeans-${first_day}"
        mkdir -p ${OUTPUT_DIR}
        OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}.nc"
        export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
        export INFERENCE_DATE=${BASE_DATE}
        echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
        anemoi-inference run "${CONFIG_FILE}"
        
        # Run inference for eleventh day
        BASE_DATE="${eleventh}T12:00"
        OUTPUT_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/Weekly_Means_72k_lr_0.625e-7/aifs-subs-pretrain-weeklymeans-${eleventh}"
        mkdir -p ${OUTPUT_DIR}
        OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}.nc"
        export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
        export INFERENCE_DATE=${BASE_DATE}
        echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
        anemoi-inference run "${CONFIG_FILE}"
        
        # Run inference for twenty-first day
        BASE_DATE="${twenty_first}T12:00"
        OUTPUT_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/Weekly_Means_72k_lr_0.625e-7/aifs-subs-pretrain-weeklymeans-${twenty_first}"
        mkdir -p ${OUTPUT_DIR}
        OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}.nc"
        export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
        export INFERENCE_DATE=${BASE_DATE}
        echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
        anemoi-inference run "${CONFIG_FILE}"
    done
done