#!/bin/bash
#SBATCH --job-name=metrics_weekly_means
#SBATCH --output=./slurm_scripts/output_slurm/run_metrics_weekly_means-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/run_metrics_weekly_means-%J.err
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment
#SBATCH --qos=np
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Activate the metrics virtual environment
source /ec/res4/hpcperm/nld4584/venv_metrics_2026_04_02/bin/activate

# Run the metrics computation script
python3 ./python_scripts/weekly_means/metrics_weekly_means.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
