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

    The physics penalty compares predicted and target-derived diagnostics
    (relative humidity and mixing ratio) and adds their squared residuals
    to selected output channels.
    """

    name: str = "pinn_mse"

    def __init__(self, ignore_nans: bool = False, *, physics_weight: float = 0.0, alpha: float = 0.0) -> None:
        super().__init__(ignore_nans=ignore_nans)
        self.physics_weight = float(physics_weight)
        # Blending factor between data loss and physics loss: final_loss =
        # (1 - alpha) * data_loss + alpha * physics_loss
        self.alpha = float(alpha)

        # Variable indices (set via set_data_indices)
        self._idx_2t = None
        self._idx_sp = None
        self._idx_2d = None

    @staticmethod
    def _compute_r_sur(t2m: torch.Tensor, dp2m: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        """Compute water-vapour mixing ratio at the surface (g/kg) with PyTorch.

        Inputs/outputs are elementwise and preserve tensor shape. Expected
        units: temperatures in degC, pressure in hPa.
        """
        e = torch.where(
            t2m >= 0.0,
            6.107 * torch.exp((17.368 * dp2m) / (dp2m + 238.83)),
            6.108 * torch.exp((17.856 * dp2m) / (dp2m + 245.52)),
        )
        e = torch.clamp(e, min=0.0)
        sp_safe = torch.clamp(sp, min=1e-6)
        denom = sp_safe - e
        denom = torch.where(denom > 0.0, denom, torch.tensor(float('nan'), device=denom.device, dtype=denom.dtype))
        r_sur = 622.0 * (e / denom)
        return r_sur

    @staticmethod
    def _compute_rh_sur(t2m: torch.Tensor, dp2m: torch.Tensor, *, clip_for_plot: bool = False) -> torch.Tensor:
        """Compute relative humidity (%) at the surface with PyTorch.

        If `clip_for_plot` True, clip to [0,100] for visualization; for loss
        computation use `clip_for_plot=False` to preserve raw residuals.
        """
        e = torch.where(
            t2m >= 0.0,
            6.107 * torch.exp((17.368 * dp2m) / (dp2m + 238.83)),
            6.108 * torch.exp((17.856 * dp2m) / (dp2m + 245.52)),
        )
        e_sat = torch.where(
            t2m >= 0.0,
            6.107 * torch.exp((17.368 * t2m) / (t2m + 238.83)),
            6.108 * torch.exp((17.856 * t2m) / (t2m + 245.52)),
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

        dtype = pred.dtype
        device = pred.device

        # Extract variables and compute diagnostics (on device)
        t2m_pred = pred[..., self._idx_2t].to(dtype=dtype)
        dp2m_pred = pred[..., self._idx_2d].to(dtype=dtype)
        sp_pred = pred[..., self._idx_sp].to(dtype=dtype)

        t2m_tgt = target[..., self._idx_2t].to(dtype=dtype)
        dp2m_tgt = target[..., self._idx_2d].to(dtype=dtype)
        sp_tgt = target[..., self._idx_sp].to(dtype=dtype)

        rh_pred = self._compute_rh_sur(t2m_pred, dp2m_pred, clip_for_plot=False)
        rh_tgt = self._compute_rh_sur(t2m_tgt, dp2m_tgt, clip_for_plot=False)

        r_pred = self._compute_r_sur(t2m_pred, dp2m_pred, sp_pred)
        r_tgt = self._compute_r_sur(t2m_tgt, dp2m_tgt, sp_tgt)

        # Normalize residuals by physical units (RH in %, r in g/kg)
        rh_scale = torch.tensor(100.0, device=device, dtype=dtype)
        r_scale = torch.tensor(10.0, device=device, dtype=dtype)

        rh_normed = (rh_pred - rh_tgt) / rh_scale
        r_normed = (r_pred - r_tgt) / r_scale

        physics_per_point = torch.square(rh_normed) + torch.square(r_normed)

        # Apply physics_weight per-point
        physics_per_point = (self.physics_weight * physics_per_point).to(device=device, dtype=dtype)

        # Add a singleton variable dim, apply area weighting, then reduce
        physics_per_point_expanded = physics_per_point[..., None]
        physics_scaled = self.scale(physics_per_point_expanded, grid_shard_slice=grid_shard_slice)
        physics_loss = self.reduce(physics_scaled, squash=True, group=group if is_sharded else None)

        # Blend losses
        combined = (1.0 - self.alpha) * data_loss + self.alpha * physics_loss
        return combined


__all__ = ["PINNMSELoss"]


