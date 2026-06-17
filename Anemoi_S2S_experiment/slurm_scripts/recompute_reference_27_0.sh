#!/bin/bash
#SBATCH --job-name=recompute_ref_27_0
#SBATCH --output=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/slurm_scripts/output_slurm/recompute_ref_27_0_%j.out
#SBATCH --error=/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/slurm_scripts/output_slurm/recompute_ref_27_0_%j.out
#SBATCH --time=6:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

# Recompute only the NEW reference model (27-0) with corrected tp handling
# This is much faster than re-running the entire preprocessing pipeline

echo "Starting recomputation of reference model (27-0) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"

# Activate virtual environment
source /ec/res4/hpcperm/nld4584/anemoi_core_2025_17_11/venv_anemoi_core_2025_17_11/bin/activate

# Run the recomputation script
python /ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/recompute_reference_27_0.py

echo "Recomputation completed at $(date)"
