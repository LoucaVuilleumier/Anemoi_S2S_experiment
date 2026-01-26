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


#get the variables of interest for the dataset
def get_var_dataset(dataset, variable):
    idx_variable = var_names.index(variable)
    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})

#select 1979-2019 for climatology - only load needed variables
date_2019 = np.datetime64('2019-12-31T00:00')
idx_2019 = np.where(times == date_2019)[0][0]

print("Computing climatologies for variables of interest...")

# Process only the variables we need to save memory
ds_dataset_1979_2019 = ds_dataset.isel(time=slice(0, idx_2019 + 1), variable=var_indices)
# Set time as coordinate from dates for resampling
ds_dataset_1979_2019 = ds_dataset_1979_2019.assign_coords(time=ds_dataset_1979_2019.dates)
# Resample to daily means
ds_dataset_daily = ds_dataset_1979_2019.resample(time='1D').mean()

#Compute daily climatology
ds_climatology_daily = ds_dataset_daily.groupby('time.dayofyear').mean('time')

#Compute weekly climatology by grouping days into weeks (7-day bins)
ds_climatology_weekly = ds_climatology_daily.groupby(ds_climatology_daily.dayofyear // 7).mean('dayofyear')
ds_climatology_weekly = ds_climatology_weekly.rename({'dayofyear': 'weekofyear'})

# Load climatology into memory now (small enough)
ds_climatology_weekly = ds_climatology_weekly.compute()

# Drop complex attributes that can't be serialized to NetCDF
attrs_to_keep = {'variables': var_names}
ds_climatology_weekly.attrs = attrs_to_keep

# Export as NetCDF
output_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/weekly_climatology_1979-2019.nc"
ds_climatology_weekly.to_netcdf(output_path)
print(f"Weekly climatology saved to: {output_path}")