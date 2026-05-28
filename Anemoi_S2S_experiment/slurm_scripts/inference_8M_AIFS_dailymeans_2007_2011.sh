#!/bin/bash

# The job name
#SBATCH --job-name=inference_8M_AIFS_dailymeans_2007-2011

# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=./slurm_scripts/output_slurm/inference_8M_AIFS_dailymeans-lr_0.625e-6_2007-2011.out
#SBATCH --error=./slurm_scripts/output_slurm/inference_8M_AIFS_dailymeans-lr_0.625e-6_2007-2011.out

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

source /ec/res4/hpcperm/nld4584/anemoi_python_3_11_ag_2026_23_04/.venv/bin/activate

# Full path to the YAML config file
CONFIG_FILE="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/Inference/inference_aifs_daily_means.yaml"

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
        OUTPUT_DIR="/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/Daily_Mean0.625e-6/aifs-subs-pretrain-dailymeans-${first_day}"
        mkdir -p ${OUTPUT_DIR}
        OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}.nc"
        export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
        export INFERENCE_DATE=${BASE_DATE}
        echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
        anemoi-inference run "${CONFIG_FILE}"
        
        # Run inference for eleventh day
        BASE_DATE="${eleventh}T12:00"
        OUTPUT_DIR="/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/Daily_Mean0.625e-6/aifs-subs-pretrain-dailymeans-${eleventh}"
        mkdir -p ${OUTPUT_DIR}
        OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}.nc"
        export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
        export INFERENCE_DATE=${BASE_DATE}
        echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
        anemoi-inference run "${CONFIG_FILE}"
        
        # Run inference for twenty-first day
        BASE_DATE="${twenty_first}T12:00"
        OUTPUT_DIR="/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/Daily_Mean0.625e-6/aifs-subs-pretrain-dailymeans-${twenty_first}"
        mkdir -p ${OUTPUT_DIR}
        OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}.nc"
        export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
        export INFERENCE_DATE=${BASE_DATE}
        echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
        anemoi-inference run "${CONFIG_FILE}"
    done
done