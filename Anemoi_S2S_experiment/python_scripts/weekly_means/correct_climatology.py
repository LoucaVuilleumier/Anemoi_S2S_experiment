import xarray as xr
import numpy as np
import pandas as pd
import glob
import gc
import importlib
import os
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')

preprocessed_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init"


print("Loading observations...", flush=True)
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-6h-v1-for-single-v2.zarr"
# Use consolidated=False to avoid the slow fallback warning
obs = xr.open_zarr(dataset_path, consolidated=False)
times = obs.dates.values
obs = obs.assign_coords(time=times).rename({"cell": "values"})

#select only a subset of variable
var_names = obs.attrs["variables"]
var_of_interest = ['2t', '2d', 'msl', 'sp', 'skt', 'tp', '10u', '10v','t_500', 't_700', 't_850', 't_925', 'q_500', 'q_700', 'q_850', 'q_925', "z_500", "z_700", "z_850", "z_925"]
var_indices = [var_names.index(var) for var in var_of_interest]
obs = obs.isel(variable=var_indices)

#create latitudinal weights for the metrics
lat_lon_coords = {
            'latitude': obs['latitudes'],
            'longitude': obs['longitudes']
        }
lat_weights = np.cos(np.radians(lat_lon_coords["latitude"].values))
lat_weights = xr.DataArray(lat_weights, dims=["values"])

print("Loading datasets for climatology correction...")
ds_inf_weekly_27_0 = xr.open_dataset(os.path.join(preprocessed_dir, "weekly_finetuned_model_27_0.nc"))
ds_inf_weekly_27_0["tp"].values = ds_inf_weekly_27_0["tp"].values * 7

weekly_init_dates = ds_inf_weekly_27_0.init_date.values

ds_clim_multiple_inf_weekly_27_0_init = []
print("Compute climatology...")
if weekly_init_dates is not None:
    for init_date in weekly_init_dates:
        year = pd.to_datetime(init_date).year
        month = pd.to_datetime(init_date).month
        day = pd.to_datetime(init_date).day
        # Climatology for weekly finetuned model (27-0)
        if ds_inf_weekly_27_0 is not None and init_date in ds_inf_weekly_27_0.init_date.values:
            mask_weekly_27_0 = (ds_inf_weekly_27_0.init_date.dt.month == month) & (ds_inf_weekly_27_0.init_date.dt.day == day) & (ds_inf_weekly_27_0.init_date.dt.year !=
       year)
            climatology_weekly_inf_27_0 = ds_inf_weekly_27_0.isel(init_date=mask_weekly_27_0).mean(dim="init_date")
            ds_clim_multiple_inf_weekly_27_0_init.append(climatology_weekly_inf_27_0)
    
    if ds_inf_weekly_27_0 is not None and ds_clim_multiple_inf_weekly_27_0_init:
        weekly_27_0_init_dates = [init_date for init_date in weekly_init_dates if init_date in ds_inf_weekly_27_0.init_date.values]
        ds_climatology_weekly_inf_27_0 = xr.concat(ds_clim_multiple_inf_weekly_27_0_init, dim=xr.DataArray(weekly_27_0_init_dates, dims='init_date', name='init_date'))
    else:
        ds_climatology_weekly_inf_27_0 = None
        
    print(f"  Weekly model (27-0): {len(ds_clim_multiple_inf_weekly_27_0_init)} climatologies")

print("Compute anomalies...")
predicted_anomalies_weekly_27_0 = ds_inf_weekly_27_0 - ds_climatology_weekly_inf_27_0

#Export
print("=" * 80)
print("EXPORTING PREPROCESSED DATA - 75 INIT DATES")
print("=" * 80)

output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init"
os.makedirs(output_dir, exist_ok=True)
print("Exporting climatology and anomalies...")

if predicted_anomalies_weekly_27_0 is not None:
    predicted_anomalies_weekly_27_0 = predicted_anomalies_weekly_27_0.assign_coords(
        latitude=("values", lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values))
    weekly_27_0_anom_path = os.path.join(output_dir, "weekly_finetuned_model_anomalies_27_0.nc")
    if os.path.exists(weekly_27_0_anom_path):
        os.remove(weekly_27_0_anom_path)
    print(f"Saving weekly finetuned model anomalies (27-0) to {weekly_27_0_anom_path}")
    # Use Zarr format for robust handling of large dask arrays
    zarr_path = weekly_27_0_anom_path.replace('.nc', '.zarr')
    if os.path.exists(zarr_path):
        import shutil
        shutil.rmtree(zarr_path)
    predicted_anomalies_weekly_27_0.to_zarr(zarr_path, mode='w', consolidated=True)
    print(f"  Saved to Zarr format: {zarr_path}")
    
if ds_climatology_weekly_inf_27_0 is not None:
    ds_climatology_weekly_inf_27_0 = ds_climatology_weekly_inf_27_0.assign_coords(
        latitude=("values", lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values))
    weekly_27_0_clim_path = os.path.join(output_dir, "weekly_finetuned_model_climatology_27_0.nc")
    if os.path.exists(weekly_27_0_clim_path):
        os.remove(weekly_27_0_clim_path)
    print(f"Saving weekly finetuned model climatology (27-0) to {weekly_27_0_clim_path}")
    ds_climatology_weekly_inf_27_0.to_netcdf(weekly_27_0_clim_path, mode='w')
