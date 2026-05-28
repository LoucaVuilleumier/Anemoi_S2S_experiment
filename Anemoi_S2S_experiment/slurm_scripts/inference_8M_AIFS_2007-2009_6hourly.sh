#!/bin/bash

# The job name
#SBATCH --job-name=inference_8M_AIFS_2007-2009_6hourly

# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=./slurm_scripts/output_slurm/inference_8M_AIFS_2007-2009_6hourly.out
#SBATCH --error=./slurm_scripts/output_slurm/inference_8M_AIFS_2007-2009_6hourly.out

#SBATCH --array=0-7
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

export ANEMOI_BASE_SEED=$SLURM_ARRAY_TASK_ID
export HYDRA_FULL_ERROR=1

source /ec/res4/hpcperm/nld4584/anemoi_python_3_11_ag_2026_23_04/.venv/bin/activate

# Full path to the YAML config file
CONFIG_FILE="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/Inference/inference_aifs_subs_pretrain.yaml"

# Generate inferences for 1st and 15th of July 2007-2009
# Run 4 inferences per base date (00:00, 06:00, 12:00, 18:00) for lagged ensemble
# Each inference produces 56 days of 6-hourly forecasts (sufficient for 8 weeks of data)

for year in {2007..2009}; do
    for month in 7; do
        # Find the first day of the month
        first_day=$(printf "%04d-%02d-01" $year $month)
        
        # Calculate fifteenth day (14 days after first day)
        fifteenth=$(date -d "$first_day + 14 days" +%Y-%m-%d)
        
        # Process both first and fifteenth
        for base_date in "$first_day" "$fifteenth"; do
            # Create output directory for this date
            OUTPUT_DIR="/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/AIFS-6hourly/aifs-subs-pretrain-${base_date}"
            mkdir -p ${OUTPUT_DIR}
            
            # Run inference at 4 initialization times: 00:00, 06:00, 12:00, 18:00
            for init_hour in "00:00" "06:00" "12:00" "18:00"; do
                # Convert time format for filename (00:00 -> 0000)
                hour_code=$(echo ${init_hour} | tr -d ':')
                
                BASE_DATE="${base_date}T${init_hour}"
                OUTPUT_FILE="${OUTPUT_DIR}/member-${SLURM_ARRAY_TASK_ID}-${hour_code}.nc"
                
                export INFERENCE_OUTPUT_PATH=${OUTPUT_FILE}
                export INFERENCE_DATE=${BASE_DATE}
                
                echo "Running inference for ${BASE_DATE} (member ${SLURM_ARRAY_TASK_ID})"
                anemoi-inference run "${CONFIG_FILE}"
            done
        done
    done
done

echo "Completed all 6-hourly inferences for member ${SLURM_ARRAY_TASK_ID}"
