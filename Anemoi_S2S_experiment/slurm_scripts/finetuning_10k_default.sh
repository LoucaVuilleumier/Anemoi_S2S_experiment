#!/bin/bash
#SBATCH --job-name=finetuning_10k_default
#SBATCH --output=./slurm_scripts/output_slurm/finetuning_10k_default-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/finetuning_10k_default-%J.out
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment


#SBATCH --qos=ng
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=460G
#SBATCH --time=24:00:00

source /ec/res4/hpcperm/nld4584/anemoi_core_2025_17_11/venv_anemoi_core_2025_17_11/bin/activate
export ANEMOI_BASE_SEED=42

export ANEMOI_CONFIG_PATH=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new/Training
srun anemoi-training train --config-name=finetuning_10k_default.yaml