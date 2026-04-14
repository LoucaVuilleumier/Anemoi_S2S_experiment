#!/bin/bash
#SBATCH --job-name=test_zarr
#SBATCH --output=slurm_scripts/output_slurm/test_zarr-%j.out
#SBATCH --error=slurm_scripts/output_slurm/test_zarr-%j.err
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G # Use all available memory on the node
#SBATCH --time=01:00:00

# Load environment
source /ec/res4/hpcperm/nld4584/anemoi_python3_11_ag_2026_16_03/.venv/bin/activate

cd /ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment

echo "========================================="
echo "Starting Zarr corruption tests"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================="

# Run basic test
echo ""
echo "[1/3] Running basic corruption test..."
python test_bug_rolling_average.py

# Run grid shard test
echo ""
echo "[2/3] Running grid shard test..."
python test_zarr_grid_shards.py

# Run multiprocess test (most realistic)
echo ""
echo "[3/3] Running multiprocess test (simulates actual dataloader)..."
python test_zarr_multiprocess.py

echo ""
echo "========================================="
echo "Tests completed at: $(date)"
echo "Check the generated .log files for results:"
echo "  - zarr_corruption_test_*.log"
echo "  - zarr_grid_shard_test_*.log"
echo "  - zarr_multiproc_test_*.log"
echo "========================================="
