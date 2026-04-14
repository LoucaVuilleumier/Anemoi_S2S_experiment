#!/bin/bash

# The job name
#SBATCH --job-name=finetuning_weekly_means-100K_1node_lr-2.5e-4-from_ref



# Set the initial working directory
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

# Set the error and output files
#SBATCH --output=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/slurm_scripts/output_slurm/finetuning_weekly_means-100K_1node_lr-2.5e-4-from_ref%J.out
#SBATCH --error=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/slurm_scripts/output_slurm/finetuning_weekly_means-100K_1node_lr-2.5e-4-from_ref%J.out
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0 # Use all available memory on the node
#SBATCH --time=01-00:00:00




export ANEMOI_BASE_SEED=42
export HYDRA_FULL_ERROR=1



CONFIG_NAME=finetune_aifs_weekly_means.yaml
CONFIG_DIR=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/Training/

cp $CONFIG_DIR/$CONFIG_NAME "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/log_trainings/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_recipe.yaml"

# Load CUDA module for flash-attn
module load cuda/12.9

# Activate Python 3.11 environment with flash-attn
source /ec/res4/hpcperm/nld4584/anemoi_python3_11_ag_2026_16_03/.venv/bin/activate

# Add PyTorch library path for libc10_cuda.so
export LD_LIBRARY_PATH=/lus/h2resw01/hpcperm/nld4584/anemoi_python3_11_ag_2026_16_03/.venv/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH

export ANEMOI_CONFIG_PATH=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/Configs_new_2026_16_02/Training
srun anemoi-training train --config-name=finetune_aifs_weekly_means.yaml