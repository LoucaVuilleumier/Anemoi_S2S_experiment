#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=./slurm_scripts/output_slurm/hello-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/hello-%J.out
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment
#SBATCH --qos=ng
#SBATCH --time=01:00:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source /ec/res4/hpcperm/nld4584/anemoi_core_2025_17_11/venv_anemoi_core_2025_17_11/bin/activate

export ANEMOI_CONFIG_PATH=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new/Training
anemoi-training train --config-name=test_training.yaml
