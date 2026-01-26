#!/bin/bash
#SBATCH --job-name=compute_vars
#SBATCH --output=./slurm_scripts/output_slurm/run_bench_rollout-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/run_bench_rollout-%J.out
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment
#SBATCH --qos=np
#SBATCH --time=03:00:00
#SBATCH --mem=220G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


source /ec/res4/hpcperm/nld4584/anemoi_core_2025_17_11/venv_anemoi_core_2025_17_11/bin/activate

python3 /ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/bench_rollout_training.py

