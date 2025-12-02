# Custom Loss Functions

This directory contains custom loss functions for Anemoi training experiments.

## Available Losses

### PINNmse.py
Physics-Informed Neural Network (PINN) MSE loss that combines standard MSE with physics-based regularization using relative humidity (RH) and mixing ratio (r) diagnostics.

**Features:**
- Computes RH and mixing ratio from predicted 2t, sp, and 2d variables
- Normalizes physics residuals by typical physical scales (RH: 100%, r: 10 g/kg)
- Applies area-weighted averaging (when area scalers are configured)
- Blends data loss and physics loss via configurable `alpha` parameter

**Parameters:**
- `ignore_nans` (bool): Allow NaNs in loss computation (default: False)
- `physics_weight` (float): Per-point multiplier for physics residuals (default: 0.0)
- `alpha` (float): Blending factor between data and physics loss (default: 0.0)
  - Final loss = `(1 - alpha) * data_loss + alpha * physics_loss`

**Usage in config:**
```yaml
training:
  training_loss:
    _target_: anemoi.training.losses.PINNmse.PINNMSELoss
    physics_weight: 1.0
    alpha: 0.1  # 10% physics, 90% data loss
```

## Installation

Custom losses must be copied to your Anemoi installation to be used in training.

### Using the sync script 

From the repository root:

```bash
# Deploy custom losses to anemoi installation
./sync_custom_losses.sh to_anemoi

# Backup changes from anemoi to repo (if you edited files there)
./sync_custom_losses.sh from_anemoi
```

