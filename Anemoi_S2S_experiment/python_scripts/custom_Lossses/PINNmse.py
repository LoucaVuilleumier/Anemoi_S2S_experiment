# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""MSE loss with optional PINN-style physics regularization.

This module implements a loss that extends the existing FunctionalLoss
in the training package. It provides lightweight PyTorch versions of the
mixing-ratio and relative-humidity computations so the regularization is
fully differentiable and runs on the device (GPU) during training.

Usage:
  - Instantiate with `physics_weight > 0.0` to enable the PINN penalty.
  - Ensure the training code calls `set_data_indices(data_indices)` on the
    loss instance so variable name -> index mapping is available. The loss
    will look for common name variants (e.g. '2t' or 't2m'). If indices are
    not found the physics penalty is skipped.
"""

import logging

import torch

from anemoi.training.losses.base import FunctionalLoss


LOGGER = logging.getLogger(__name__)


class PINNMSELoss(FunctionalLoss):
    """MSE loss with optional PINN physics penalty.

    The physics penalty compares predicted and target-derived
    relative humidity and mixing ratio and adds their area-weighted, normalized squared residuals)
    """

    name: str = "pinn_mse"

    def __init__(self, ignore_nans: bool = False, *, physics_weight: float = 0.0, alpha: float = 0.0, dataset_path: str = None) -> None:
        super().__init__(ignore_nans=ignore_nans)
        self.physics_weight = float(physics_weight)
        # Blending factor between data loss and physics loss: final_loss =
        # (1 - alpha) * data_loss + alpha * physics_loss
        self.alpha = float(alpha)

        # Variable indices (set via set_data_indices)
        self._idx_2t = None
        self._idx_sp = None
        self._idx_2d = None
        # Will store denormalization parameters
        self._norm_mul = None
        self._norm_add = None
        self._dataset_path = dataset_path
        self._stats_loaded = False  # Flag to track if statistics have been loaded

    def set_data_indices(self, data_indices) -> None:
        """Set variable indices from IndexCollection and load statistics for denormalization.
        
        Looks for common name variants for 2m temperature, 2m dewpoint, and surface pressure.
        
        Called during get_loss_function() when loss is instantiated (I think).
        """
        # Access the name_to_index dictionary from model output
        name_to_index = data_indices.model.output.name_to_index
        
        # Try common variants for 2m temperature
        for name in ['2t', 't2m']:
            if name in name_to_index:
                self._idx_2t = name_to_index[name]
                LOGGER.info(f"PINNMSELoss: found 2m temperature at index {self._idx_2t} via '{name}'")
                break
        
        # Try common variants for 2m dewpoint temperature
        for name in ['2d', 'd2m']:
            if name in name_to_index:
                self._idx_2d = name_to_index[name]
                LOGGER.info(f"PINNMSELoss: found 2m dewpoint at index {self._idx_2d} via '{name}'")
                break
        
        # Try common variants for surface pressure
        for name in ['sp', 'msl']:
            if name in name_to_index:
                self._idx_sp = name_to_index[name]
                LOGGER.info(f"PINNMSELoss: found surface pressure at index {self._idx_sp} via '{name}'")
                break
        
        # Load statistics from dataset (only if not already loaded)
        if self._dataset_path is not None and not self._stats_loaded:
            try:
                import xarray as xr
                import numpy as np
                ds = xr.open_zarr(self._dataset_path, consolidated=False)
                
                if 'mean' in ds and 'stdev' in ds:
                    mean = ds['mean'].values
                    stdev = ds['stdev'].values
                    LOGGER.info("PINNMSELoss: found 'mean' and 'stdev' variables in dataset for statistics")
                else:
                    mean = None
                    stdev = None
                    LOGGER.warning("PINNMSELoss: 'mean' and 'stdev' variables not found in dataset")
                
                if mean is not None and stdev is not None:
                    self._norm_mul = torch.from_numpy(stdev).float()
                    self._norm_add = torch.from_numpy(mean).float()
                    self._stats_loaded = True
                    LOGGER.info("PINNMSELoss: loaded statistics from dataset")
                else:
                    LOGGER.error("PINNMSELoss: could not find statistics in dataset")
            except Exception as e:
                LOGGER.error(f"PINNMSELoss: failed to load statistics: {e}")
            
        
        
    def set_indices_manually(self, idx_2t: int, idx_2d: int, idx_sp: int) -> None:
        """Manually set variable indices for testing without IndexCollection.
        
        Parameters
        ----------
        idx_2t : int
            Index of 2m temperature in the last dimension of pred/target tensors
        idx_2d : int
            Index of 2m dewpoint temperature in the last dimension
        idx_sp : int
            Index of surface pressure in the last dimension
        """
        self._idx_2t = idx_2t
        self._idx_2d = idx_2d
        self._idx_sp = idx_sp
        LOGGER.info(
            f"PINNMSELoss: manually set indices - 2t={self._idx_2t}, 2d={self._idx_2d}, sp={self._idx_sp}"
        )
        

        
    @staticmethod
    def _compute_r_sur(t2m: torch.Tensor, dp2m: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        """Compute water-vapour mixing ratio at the surface (g/kg) with PyTorch.

        Inputs/outputs are elementwise and preserve tensor shape. Expected
        units: temperatures in degC, pressure in hPa.
        a, b, c from August–Roche–Magnus approximation.
        d derived from the ideal gas law for dry air and water vapor
        """
        a_pos, b_pos, c_pos = 17.368, 238.83, 6.107
        a_neg, b_neg, c_neg = 17.856, 245.52, 6.108
        d = 0.622  # 622 g/kg
        
        e = torch.where(
            t2m >= 0.0,
            c_pos * torch.exp((a_pos * dp2m) / (dp2m + b_pos)),
            c_neg * torch.exp((a_neg * dp2m) / (dp2m + b_neg)),
        )
        e = torch.clamp(e, min=0.0)
        sp_safe = torch.clamp(sp, min=1e-6)
        denom = sp_safe - e
        denom = torch.where(denom > 0.0, denom, torch.tensor(float('nan'), device=denom.device, dtype=denom.dtype))
        r_sur = d * (e / denom)
        return r_sur

    @staticmethod
    def _compute_rh_sur(t2m: torch.Tensor, dp2m: torch.Tensor, *, clip_for_plot: bool = False) -> torch.Tensor:
        """Compute relative humidity (%) at the surface with PyTorch.

        If `clip_for_plot` True, clip to [0,100] for visualization; for loss
        computation use `clip_for_plot=False` to preserve raw residuals.
        
        a, b, c from August–Roche–Magnus approximation.
        """
        a_pos, b_pos, c_pos = 17.368, 238.83, 6.107
        a_neg, b_neg, c_neg = 17.856, 245.52, 6.108

        
        e = torch.where(
            t2m >= 0.0,
            c_pos * torch.exp((a_pos * dp2m) / (dp2m + b_pos)),
            c_neg * torch.exp((a_neg * dp2m) / (dp2m + b_neg)),
        )
        e_sat = torch.where(
            t2m >= 0.0,
            c_pos * torch.exp((a_pos * t2m) / (t2m + b_pos)),
            c_neg * torch.exp((a_neg * t2m) / (t2m + b_neg)),
        )
        e = torch.clamp(e, min=0.0)
        e_sat = torch.clamp(e_sat, min=1e-12)
        rh = 100.0 * e / e_sat
        if clip_for_plot:
            rh = torch.clamp(rh, 0.0, 100.0)
        return rh

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return per-element squared difference only.

        Physics blending happens in forward(); this just computes element-wise MSE.
        """
        return torch.square(pred - target)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: object | None = None,
    ) -> torch.Tensor:
        """Compute combined loss: (1-alpha)*data_loss + alpha*physics_loss.

        The data_loss is computed by the superclass (area-weighted, scaled,
        and reduced). The physics_loss is computed from predicted and target
        diagnostics, normalized typicall values.
        """
        is_sharded = grid_shard_slice is not None

        # Data loss (reduced scalar)
        data_loss = super().forward(
            pred,
            target,
            squash=squash,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
            group=group if is_sharded else None,
        )

        # If blending not requested, or missing indices, return data_loss
        if not (self.alpha > 0.0 and self.physics_weight > 0.0):
            return data_loss

        if self._idx_2t is None or self._idx_sp is None or self._idx_2d is None:
            LOGGER.debug("PINNMSELoss: missing indices, skipping physics penalty")
            return data_loss
        
        if self._norm_mul is None or self._norm_add is None:
            LOGGER.warning("PINNMSELoss: statistics not loaded, cannot compute physics loss")
            return data_loss

        dtype = pred.dtype
        device = pred.device
        
        
        
        #Extract and Denormalize pred and target
        t2m_pred_K = pred[..., self._idx_2t] * self._norm_mul[self._idx_2t] + self._norm_add[self._idx_2t]
        dp2m_pred_K = pred[..., self._idx_2d] * self._norm_mul[self._idx_2d] + self._norm_add[self._idx_2d]
        sp_pred_Pa = pred[..., self._idx_sp] * self._norm_mul[self._idx_sp] + self._norm_add[self._idx_sp]
        
        t2m_tgt_K = target[..., self._idx_2t] * self._norm_mul[self._idx_2t] + self._norm_add[self._idx_2t]
        dp2m_tgt_K = target[..., self._idx_2d] * self._norm_mul[self._idx_2d] + self._norm_add[self._idx_2d]
        sp_tgt_Pa = target[..., self._idx_sp] * self._norm_mul[self._idx_sp] + self._norm_add[self._idx_sp]

        # Convert to units expected by physics functions (Celsius and hPa)
        t2m_pred = t2m_pred_K - 273.15
        dp2m_pred = dp2m_pred_K - 273.15
        sp_pred = sp_pred_Pa / 100.0

        t2m_tgt = t2m_tgt_K - 273.15
        dp2m_tgt = dp2m_tgt_K - 273.15
        sp_tgt = sp_tgt_Pa / 100.0

        # Compute diagnostics with converted units
        rh_pred = self._compute_rh_sur(t2m_pred, dp2m_pred, clip_for_plot=False)
        rh_tgt = self._compute_rh_sur(t2m_tgt, dp2m_tgt, clip_for_plot=False)

        r_pred = self._compute_r_sur(t2m_pred, dp2m_pred, sp_pred)
        r_tgt = self._compute_r_sur(t2m_tgt, dp2m_tgt, sp_tgt)

        # Normalize residuals by physical units (RH in %, r in g/kg)
        rh_variance = torch.var(rh_tgt)
        r_variance = torch.var(r_tgt)

        # Compute normalized squared errors per point
        rh_normed = torch.square(rh_pred - rh_tgt) / rh_variance
        r_normed = torch.square(r_pred - r_tgt) / r_variance
        
        # Stack as separate "variables" along last dimension: [..., 2]
        # This allows proper squashing (averaging) over the physics variables
        physics_per_point = torch.stack([rh_normed, r_normed], dim=-1)  # shape: [..., 2]

        # Apply physics_weight per-point
        physics_per_point = (self.physics_weight * physics_per_point).to(device=device, dtype=dtype)

        # Apply area weighting (grid dimension) and reduce
        physics_scaled = self.scale(physics_per_point, grid_shard_slice=grid_shard_slice,
                                    without_scalers=['pressure_level', 'general_variable', 'nan_mask_weights'])
        # Squash=True will average over the last dimension (the 2 physics variables)
        physics_loss = self.reduce(physics_scaled, squash=True, group=group if is_sharded else None)

        # Blend losses
        combined = (1.0 - self.alpha) * data_loss + self.alpha * physics_loss
        return combined


__all__ = ["PINNMSELoss"]


