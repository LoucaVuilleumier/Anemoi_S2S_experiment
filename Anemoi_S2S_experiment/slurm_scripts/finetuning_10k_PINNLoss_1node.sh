#!/bin/bash

# The job name
#SBATCH --job-name=finetuning_10k_PINNLoss

# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=./slurm_scripts/output_slurm/finetuning_10k_PINNLoss-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/finetuning_10k_PINNLoss-%J.out

#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=120G
#SBATCH --time=12:00:00



# export ANEMOI_BASE_SEED=16014075
export HYDRA_FULL_ERROR=1
source /ec/res4/hpcperm/nld4584/anemoi_core_2025_17_11/venv_anemoi_core_2025_17_11/bin/activate

export ANEMOI_CONFIG_PATH=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new/Training
srun anemoi-training train --config-name=finetuning_10k_PINNLoss.yaml