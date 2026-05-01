#!/bin/bash
#SBATCH --job-name=preprocess_heatwave_physics
#SBATCH --output=./slurm_scripts/output_slurm/preprocess_heatwave_physics-%J.out
#SBATCH --error=./slurm_scripts/output_slurm/preprocess_heatwave_physics-%J.out
#SBATCH --chdir=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment
#SBATCH --qos=np
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

echo "============================================"
echo "Preprocessing Heatwave Physics Data"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 300GB"
echo "============================================"
echo ""

# Activate the virtual environment
source /ec/res4/hpcperm/nld4584/venv_metrics_2026_04_02/bin/activate

# Set Python to unbuffered mode so output appears immediately in logs
export PYTHONUNBUFFERED=1

# Set number of threads for numpy/dask operations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the preprocessing script with unbuffered output
python3 -u ./python_scripts/weekly_means/preprocess_heatwave_physics_data.py

echo ""
echo "============================================"
echo "Preprocessing completed!"
echo "End time: $(date)"
echo "============================================"
