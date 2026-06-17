#!/bin/bash
#
# Run heat wave analysis with R integration for economic value computation
#

echo "=================================================="
echo "Heat Wave Analysis - Starting..."
echo "=================================================="

# Activate Python virtual environment
echo "→ Activating Python environment..."
source /ec/res4/hpcperm/nld4584/venv_metrics_2026_04_02/bin/activate

# Load R module
echo "→ Loading R module..."
module load R/4.5.3

# Run analysis
echo "→ Running heat wave analysis..."
cd /ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means
python heat_wave_analysis.py

echo ""
echo "=================================================="
echo "Analysis complete!"
echo "=================================================="
