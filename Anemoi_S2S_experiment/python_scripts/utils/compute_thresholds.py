import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LogNorm
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import plotting_functions as pf
from utils import metrics_function as mf
from collections import defaultdict
import importlib
importlib.reload(pf)
import glob
import os
import pandas as pd
import time

dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"

# Open with chunks to enable lazy loading
ds_dataset = xr.open_zarr(dataset_path)
times = ds_dataset.dates.values
var_names = ds_dataset.attrs["variables"]

print("Dataset opened...")

# Get only the indices of variables we need
var_of_interest = ["2t", "tp", "10u", "10v"]
var_indices = [var_names.index(var) for var in var_of_interest]

#Select dates for climatology
date_2019 = np.datetime64('2019-12-31T00:00')
idx_2019 = np.where(times == date_2019)[0][0]

# Process only the variables we need to save memory
ds_dataset_1979_2019 = ds_dataset.isel(time=slice(0, idx_2019 + 1), variable=var_indices)
# Set time as coordinate from dates for resampling
ds_dataset_1979_2019 = ds_dataset_1979_2019.assign_coords(time=ds_dataset_1979_2019.dates)


#resample to weekly means
ds_dataset_weekly = ds_dataset_1979_2019.resample(time='1W').mean()

#compute thresholds for SEDI
thresholds = ds_dataset_weekly.quantile(0.95, dim="time")

#convert to dataset for export
thresholds = thresholds.assign_coords(variable=var_of_interest)
thresholds_ds = thresholds.to_dataset(dim="variable")

#exports
# Export as NetCDF
output_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/thresholds_1979-2019.nc"
thresholds_ds.to_netcdf(output_path)
print(f"Thresholds saved to: {output_path}")
