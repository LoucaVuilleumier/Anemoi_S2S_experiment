#!/bin/bash
#SBATCH --job-name=preprocess_weekly_metrics
#SBATCH --output=./slurm_scripts/output_slurm/run_preprocessing_metrics_weekly_means-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/run_preprocessing_metrics_weekly_means-%J.out
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment
#SBATCH --qos=np
#SBATCH --time=06:00:00
#SBATCH --mem=0G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Memory-efficient preprocessing: processes one init date at a time
# for 6-hourly reference model to avoid memory overflow

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Configure threading to use all allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Threading configuration:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "  NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "=========================================="

# Activate the metrics virtual environment
source /ec/res4/hpcperm/nld4584/venv_metrics_2026_04_02/bin/activate

# Run the preprocessing script
python3 ./python_scripts/weekly_means/preprocessing_metrics_weekly_means.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
