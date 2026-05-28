#!/bin/bash
#SBATCH --job-name=heatwave_analysis
#SBATCH --output=./slurm_scripts/output_slurm/heatwave_analysis-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/heatwave_analysis-%J.out
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment
#SBATCH --qos=np
#SBATCH --time=06:00:00
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

echo "============================================"
echo "Heat Wave Analysis with Comprehensive Physics"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 120GB"
echo "============================================"
echo ""

# Activate the virtual environment
source /ec/res4/hpcperm/nld4584/venv_metrics_2026_04_02/bin/activate

# Set Python to unbuffered mode so output appears immediately in logs
export PYTHONUNBUFFERED=1

# Set number of threads for numpy operations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the heat wave analysis script with unbuffered output
python3 -u ./python_scripts/weekly_means/heat_wave_analysis.py

echo ""
echo "============================================"
echo "Heat Wave Analysis completed!"
echo "End time: $(date)"
echo "============================================"
