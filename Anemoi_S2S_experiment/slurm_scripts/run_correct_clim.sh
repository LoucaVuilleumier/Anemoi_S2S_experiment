#!/bin/bash
#SBATCH --job-name=correct_clim_weekly_means-75-init
#SBATCH --output=./slurm_scripts/output_slurm/run_correct_clim_weekly_means-75-init-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/run_correct_clim_weekly_means-75-init-%J.out
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment
#SBATCH --qos=np
#SBATCH --time=24:00:00
#SBATCH --mem=0G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

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
python3 /ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/correct_climatology.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
