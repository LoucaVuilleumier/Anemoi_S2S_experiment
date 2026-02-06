import xarray as xr
import numpy as np
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import metrics_function as mf
import glob
import os


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


ACC_spatial = {var: [] for var in var_of_interest}

# For R_t computation: accumulate sums instead of storing all data
# This avoids memory issues
rt_sums = {var: None for var in var_of_interest}  # sum of (dataset * inference)
rt_sums_dataset_sq = {var: None for var in var_of_interest}  # sum of dataset^2
rt_sums_inference_sq = {var: None for var in var_of_interest}  # sum of inference^2
n_runs_processed = 0

# Store coordinates for later export
lat_lon_coords = None

#loop over the 12 runs
for run_idx in range(12):
    print(f"Processing run {run_idx}...")
    #select inference dataset for each run
    #inference_path = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_refmodel_8weeks_run{run_idx:02d}*"
    inference_path = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_10k_rollout_8weeks_run{run_idx:02d}*"

    ds_inference = xr.open_dataset(glob.glob(inference_path)[0], engine="netcdf4")
    
    #get init date and number of steps
    init_date = ds_inference.time.values[0].astype('datetime64[s]')  # Convert to match dataset precision
    n_steps = len(ds_inference.time)
    
    #slice era5 data to match inference period - only load needed variables
    times = ds_dataset.dates.values
    t0 = np.where(times == init_date)[0][0]
    ds_dataset_sliced = ds_dataset.isel(time = slice(t0, t0 + n_steps), variable=var_indices)
    # Set time as coordinate from dates for resampling
    ds_dataset_sliced = ds_dataset_sliced.assign_coords(time=ds_dataset_sliced.dates).squeeze(dim="ensemble").rename({"cell": "values"})
    # Load this slice into memory (small enough for one run)
    ds_dataset_sliced = ds_dataset_sliced.compute()
    
    #daily resample
    ds_dataset_sliced_daily = ds_dataset_sliced.resample(time='1D').mean()
    ds_inference_daily = ds_inference.resample(time='1D').mean()
    
    #compute weekly means
    ds_dataset_sliced_weekly = ds_dataset_sliced_daily.resample(time='7D').mean()
    ds_inference_weekly = ds_inference_daily.resample(time='7D').mean()
    
    # Keep only the first 8 weeks (in case a 9th partial week bin was created)
    ds_dataset_sliced_weekly = ds_dataset_sliced_weekly.isel(time=slice(0, 8))
    ds_inference_weekly = ds_inference_weekly.isel(time=slice(0, 8))
    
    # Compute latitude weights once (outside variable loop)
    lat_weights = np.cos(np.radians(ds_inference["latitude"].values))
    lat_weights = xr.DataArray(lat_weights, dims=["values"])
    
    # Save coordinates from first run for export
    if lat_lon_coords is None and 'latitude' in ds_inference_weekly:
        lat_lon_coords = {
            'latitude': ds_inference_weekly['latitude'],
            'longitude': ds_inference_weekly['longitude']
        }
    
    #compute weekly anomalies - process one variable at a time to save memory
    for idx, var in enumerate(var_of_interest):
        # Get week numbers for alignment
        dataset_weeks = ds_dataset_sliced_weekly['time'].dt.isocalendar().week.values
        inference_weeks = ds_inference_weekly['time'].dt.isocalendar().week.values
        
        print(f"dataset weeks: {dataset_weeks}, should match inference weeks: {inference_weeks}")
        
        # Get data for this variable
        dataset_var = ds_dataset_sliced_weekly.isel(variable=idx)["data"]
        print(f"dataset_var time shape: {dataset_var.time.shape}, values shape: {dataset_var.values.shape}")
        inference_var = ds_inference_weekly[var]
        print(f"inference_var time shape: {inference_var.time.shape}, values shape: {inference_var.values.shape}")
        
        #get climatology for this variable - extract values for the weeks we need
        climatology_var = ds_climatology_weekly.isel(variable=idx).squeeze(dim="ensemble").rename({"cell": "values"})["data"]
        print(f"climatology_var shape: {climatology_var.shape}")
        
        # Build climatology arrays matching the time dimension
        clim_dataset = np.zeros_like(dataset_var.values)
        clim_inference = np.zeros_like(inference_var.values)
        
        for i_time in range(len(dataset_weeks)):
            clim_dataset[i_time] = climatology_var.sel(weekofyear=dataset_weeks[i_time]).values
        
        for i_time in range(len(inference_weeks)):
            clim_inference[i_time] = climatology_var.sel(weekofyear=inference_weeks[i_time]).values
        
        # Subtract climatology
        dataset_anom = dataset_var.copy(deep=False)
        dataset_anom.values = dataset_var.values - clim_dataset
        
        inference_anom = inference_var.copy(deep=False)
        print(f"anomaly inference_var time shape: {inference_anom.time.shape}, values shape: {inference_anom.values.shape}")
        inference_anom.values = inference_var.values - clim_inference
        
        # Compute ACC with latitude weighting
        acc_spatial = mf.compute_acc(inference_anom, dataset_anom, weights=lat_weights)
        # Convert to numpy if it's an xarray object
        if hasattr(acc_spatial, 'values'):
            ACC_spatial[var].append(acc_spatial.values)
        else:
            ACC_spatial[var].append(acc_spatial)
        
        # Accumulate statistics for R_t computation (memory efficient)
        # Rename time dimension to leadtime for proper stacking
        dataset_anom_lt = dataset_anom.rename({'time': 'leadtime'})
        inference_anom_lt = inference_anom.rename({'time': 'leadtime'})
        
        if rt_sums[var] is None:
            # Initialize accumulators
            rt_sums[var] = (dataset_anom_lt * inference_anom_lt).values
            rt_sums_dataset_sq[var] = (dataset_anom_lt ** 2).values
            rt_sums_inference_sq[var] = (inference_anom_lt ** 2).values
        else:
            # Accumulate
            rt_sums[var] += (dataset_anom_lt * inference_anom_lt).values
            rt_sums_dataset_sq[var] += (dataset_anom_lt ** 2).values
            rt_sums_inference_sq[var] += (inference_anom_lt ** 2).values
        
        # Free memory
        del dataset_var, inference_var, dataset_anom, inference_anom, clim_dataset, clim_inference, acc_spatial
        del dataset_anom_lt, inference_anom_lt
    
    # Increment run counter
    n_runs_processed += 1
    
    # Clean up after each run
    del ds_inference, ds_dataset_sliced, ds_dataset_sliced_daily, ds_inference_daily
    del ds_dataset_sliced_weekly, ds_inference_weekly

print("Computing complete. Exporting ACC results...")

output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics"
os.makedirs(output_dir, exist_ok=True)

# Export spatially-averaged ACC
# Stack the list of arrays into 2D arrays (run, leadtime)
acc_spatial_arrays = {var: np.stack(ACC_spatial[var], axis=0) for var in var_of_interest}
n_runs = acc_spatial_arrays[var_of_interest[0]].shape[0]
n_leadtimes = acc_spatial_arrays[var_of_interest[0]].shape[1]

acc_spatial_ds = xr.Dataset(
    {var: (['run', 'leadtime'], acc_spatial_arrays[var]) for var in var_of_interest}
)
acc_spatial_ds.coords['run'] = np.arange(n_runs)
acc_spatial_ds.coords['leadtime'] = np.arange(n_leadtimes)

#nc_path_spatial = os.path.join(output_dir, "ACC_weekly_anomalies_spatial.nc")
nc_path_spatial = os.path.join(output_dir, "ACC_weekly_anomalies_10k_rollout.nc")
acc_spatial_ds.to_netcdf(nc_path_spatial)
print(f"Spatially-averaged ACC saved to: {nc_path_spatial}")

# Compute and export R_t (temporal correlation coefficient)
print("Computing R_t (temporal correlation across runs)...")
rt_dict = {}

for var in var_of_interest:
    # Compute R_t from accumulated statistics (memory efficient)
    # R_t = mean(xy) / sqrt(mean(x^2) * mean(y^2))
    numerator = rt_sums[var] / n_runs_processed
    denominator = np.sqrt((rt_sums_dataset_sq[var] / n_runs_processed) * (rt_sums_inference_sq[var] / n_runs_processed))
    rt_values = numerator / denominator
    
    # Create DataArray with proper dimensions
    rt_dict[var] = xr.DataArray(
        rt_values,
        dims=['leadtime', 'values'],
        name=var
    )

# Get dimensions
n_values = rt_dict[var_of_interest[0]].shape[1]

# Create dataset
rt_ds = xr.Dataset(rt_dict)
rt_ds.coords['leadtime'] = np.arange(n_leadtimes)
rt_ds.coords['values'] = np.arange(n_values)

# Copy latitude/longitude coordinates
if lat_lon_coords is not None:
    # Drop time coordinate and use only values dimension
    rt_ds.coords['latitude'] = lat_lon_coords['latitude'].isel(time=0).drop_vars('time')
    rt_ds.coords['longitude'] = lat_lon_coords['longitude'].isel(time=0).drop_vars('time')

#nc_path_rt = os.path.join(output_dir, "Rt_weekly_anomalies.nc")
nc_path_rt = os.path.join(output_dir, "Rt_weekly_anomalies_10k_rollout.nc")

rt_ds.to_netcdf(nc_path_rt)
print(f"R_t (temporal correlation) saved to: {nc_path_rt}")


