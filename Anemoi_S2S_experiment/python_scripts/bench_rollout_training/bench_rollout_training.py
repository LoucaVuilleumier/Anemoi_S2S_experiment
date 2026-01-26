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

# Get only the indices of variables we need
var_of_interest = ["2t", "tp", "10u", "10v"]
var_indices = [var_names.index(var) for var in var_of_interest]


#get the variables of interest for the dataset
def get_var_dataset(dataset, variable, var_names):
    idx_variable = var_names.index(variable)
    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})

# Load weekly climatology computed previously
climatology_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/weekly_climatology_1979-2019.nc"
ds_climatology_weekly = xr.open_dataset(climatology_path)

var_names_clim = ds_climatology_weekly.attrs["variables"]


ACC = {var: [] for var in var_of_interest}

#loop over the 12 runs
for i in range(12):
    print(f"Processing run {i}...")
    #select inference dataset for each run
    inference_path = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_refmodel_8weeks_run{i:02d}*"
    ds_inference = xr.open_dataset(glob.glob(inference_path)[0], engine="netcdf4")
    
    #get init date and number of steps
    init_date = ds_inference.time.values[0].astype('datetime64[s]')  # Convert to match dataset precision
    n_steps = len(ds_inference.time)
    
    #slice era5 data to match inference period - only load needed variables
    times = ds_dataset.dates.values
    t0 = np.where(times == init_date)[0][0]
    ds_dataset_sliced = ds_dataset.isel(time = slice(t0, t0 + n_steps), variable=var_indices)
    # Set time as coordinate from dates for resampling
    ds_dataset_sliced = ds_dataset_sliced.assign_coords(time=ds_dataset_sliced.dates)
    # Load this slice into memory (small enough for one run)
    ds_dataset_sliced = ds_dataset_sliced.compute()
    
    #daily resample
    ds_dataset_sliced_daily = ds_dataset_sliced.resample(time='1D').mean()
    ds_inference_daily = ds_inference.resample(time='1D').mean()
    
    #compute weekly means
    ds_dataset_sliced_weekly = ds_dataset_sliced_daily.resample(time='7D').mean()
    ds_inference_weekly = ds_inference_daily.resample(time='7D').mean()
    
    #compute weekly anomalies - process one variable at a time to save memory
    for idx, var in enumerate(var_of_interest):
        # Get week numbers for alignment
        dataset_weeks = ds_dataset_sliced_weekly['time'].dt.isocalendar().week.values
        inference_weeks = ds_inference_weekly['time'].dt.isocalendar().week.values
        
        print(f"dataset weeks: {dataset_weeks}, should match inference weeks: {inference_weeks}")
        
        # Get data for this variable
        dataset_var = ds_dataset_sliced_weekly.isel(variable = idx)
        inference_var = ds_inference_weekly[var]
        
        #get climatology for this variable - extract values for the weeks we need
        climatology_var = get_var_dataset(ds_climatology_weekly, var, var_names_clim)
        clim_dataset = climatology_var.sel(weekofyear=dataset_weeks).values
        clim_inference = climatology_var.sel(weekofyear=inference_weeks).values
        
        # Subtract climatology - use .values to avoid coordinate broadcasting issues
        dataset_anom = dataset_var.copy(deep=False)
        dataset_anom["data"].values = dataset_var["data"].values - clim_dataset
        
        inference_anom = inference_var.copy(deep=False)
        inference_anom.values = inference_var.values - clim_inference
        
        # Compute ACC and immediately discard anomalies
        ACC[var].append(mf.compute_acc(inference_anom, dataset_anom))
        
        # Free memory
        del dataset_var, inference_var, dataset_anom, inference_anom, clim_dataset, clim_inference
    
    # Clean up after each run
    del ds_inference, ds_dataset_sliced, ds_dataset_sliced_daily, ds_inference_daily
    del ds_dataset_sliced_weekly, ds_inference_weekly

print("Computing complete. Exporting ACC results...")

# Convert ACC dictionary to pandas DataFrame for easy export
acc_df = pd.DataFrame(ACC)
acc_df.index.name = 'run'

# Export to CSV
output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "ACC_weekly_anomalies.csv")
acc_df.to_csv(csv_path)
print(f"ACC results saved to: {csv_path}")

# Also save as NetCDF for xarray compatibility
acc_ds = xr.Dataset({var: ('run', ACC[var]) for var in var_of_interest})
acc_ds.coords['run'] = np.arange(12)
nc_path = os.path.join(output_dir, "ACC_weekly_anomalies.nc")
acc_ds.to_netcdf(nc_path)
print(f"ACC results saved to: {nc_path}")

# Print summary statistics
print("\nACC Summary Statistics:")
print(acc_df.describe())