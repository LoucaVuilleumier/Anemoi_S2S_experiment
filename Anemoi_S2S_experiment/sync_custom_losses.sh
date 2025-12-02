#!/bin/bash
# Sync custom losses between repo and anemoi installation
#
# Usage:
#   ./sync_custom_losses.sh to_anemoi    # Copy from repo to anemoi (deploy)
#   ./sync_custom_losses.sh from_anemoi  # Copy from anemoi to repo (backup changes)

REPO_DIR="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/custom_Lossses"
ANEMOI_DIR="/ec/res4/hpcperm/nld4584/anemoi_core_2025_17_11/anemoi-core/training/src/anemoi/training/losses"

LOSS_FILE="PINNmse.py"

if [ "$1" == "to_anemoi" ]; then
    echo "Copying $LOSS_FILE from repo to anemoi installation..."
    cp "$REPO_DIR/$LOSS_FILE" "$ANEMOI_DIR/$LOSS_FILE"
    echo "✓ Deployed to $ANEMOI_DIR/$LOSS_FILE"
    
elif [ "$1" == "from_anemoi" ]; then
    echo "Copying $LOSS_FILE from anemoi installation to repo..."
    cp "$ANEMOI_DIR/$LOSS_FILE" "$REPO_DIR/$LOSS_FILE"
    echo "✓ Backed up to $REPO_DIR/$LOSS_FILE"
    echo "Don't forget to git add and commit the changes!"
    
else
    echo "Usage: $0 {to_anemoi|from_anemoi}"
    echo ""
    echo "  to_anemoi    - Deploy custom loss from repo to anemoi installation"
    echo "  from_anemoi  - Backup changes from anemoi installation to repo"
    exit 1
fi
