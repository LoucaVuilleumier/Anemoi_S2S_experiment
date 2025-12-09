import xarray as xr
import numpy as np

def compute_relative_mse(sur_field_dataset, sur_field_inference):
    """
    Compute time-resolved relative MSE:
        rMSE(t) = MSE(t) / Var_dataset(t)
    preserving the time dimension.
    """
    # Difference field
    diff = sur_field_dataset - sur_field_inference

    # MSE over the spatial dimension
    mse = (diff ** 2).mean(dim="values")

    # Variance of the truth over the same spatial dimension
    var = sur_field_dataset.var()

    # Return time series (xarray DataArray)
    return mse / var