import xarray as xr
import numpy as np
import pandas as pd
from nwpeval import rmse, mae, acc, pod, fss
import glob
import os



#dataset paths
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
ds_dataset = xr.open_zarr(dataset_path)
times = ds_dataset.dates.values
var_names = ds_dataset.attrs["variables"]

# Get only the indices of variables we need
var_of_interest = ["2t", "tp", "10u", "10v"]
var_indices = [var_names.index(var) for var in var_of_interest]

climatology_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/weekly_climatology_1979-2019.nc"
ds_climatology_weekly = xr.open_dataset(climatology_path)

lat_lon_coords = {
            'latitude': ds_climatology_weekly ['latitudes'],
            'longitude': ds_climatology_weekly ['longitudes']
        }
lat_weights = np.cos(np.radians(lat_lon_coords["latitude"][1].values))
lat_weights = xr.DataArray(lat_weights, dims=["values"])


n_runs = 12
n_vars = len(var_of_interest)
n_leadtime = 8
n_values = 40320 

acc_results_all = xr.DataArray(
    data=np.full((n_runs, n_vars, n_leadtime), np.nan),
    dims=("run", "variable", "leadtime"),
    coords={
        "run": range(n_runs),
        "variable": var_of_interest,
        "leadtime": range(1, 9)
    }
)

#create array to store observed anomalies for all runs



observed_anomalies_runs = xr.DataArray(
    data=np.full((n_runs, n_vars, n_leadtime, n_values), np.nan),
    dims=("run", "variable", "leadtime", "values"),
    coords={
        "run": range(n_runs),
        "variable": var_of_interest,
        "leadtime": range(1, 9)
    }
)

modeled_anomalies_runs = xr.DataArray(
    data=np.full((n_runs, n_vars, n_leadtime, n_values), np.nan),
    dims=("run", "variable", "leadtime", "values"),
    coords={
        "run": range(n_runs),
        "variable": var_of_interest,
        "leadtime": range(1, 9)
    }
)
#loop over the runs
for run in range(n_runs):
    
    lead_time = range(1,9)
    print(f"Processing run {run}...")
    inference_path = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_refmodel_8weeks_run{run:02d}*"
    ds_inference = xr.open_dataset(glob.glob(inference_path)[0], engine="netcdf4")

    print("Slicing datasets...")
    #slice Era5 dataset
    init_date = ds_inference.time.values[0].astype('datetime64[s]')  # Convert to match dataset precision
    n_steps = len(ds_inference.time)

    #slice era5 data to match inference period - only load needed variables
    t0 = np.where(times == init_date)[0][0]
    ds_dataset_sliced = ds_dataset.isel(time = slice(t0, t0 + n_steps), variable=var_indices)
    ds_dataset_sliced = ds_dataset_sliced.assign_coords(time=ds_dataset_sliced.dates).squeeze(dim="ensemble").rename({"cell": "values"})

    print("Resampling to daily and weekly means...")
    #daily mean
    ds_dataset_sliced_daily = ds_dataset_sliced.resample(time='1D').mean()
    ds_inference_daily = ds_inference.resample(time='1D').mean()

    #weekly mean
    ds_dataset_sliced_weekly = ds_dataset_sliced_daily.resample(time='7D').mean()
    ds_inference_weekly = ds_inference_daily.resample(time='7D').mean()

    #remove the last incomplete week
    ds_dataset_sliced_weekly = ds_dataset_sliced_weekly.isel(time=slice(0, 8))
    ds_inference_weekly = ds_inference_weekly.isel(time=slice(0, 8))
        
    print("Computing anomalies for the weekly data...")
    # Get the corresponding climatology for the variable and week
    dataset_weeks = ds_dataset_sliced_weekly['time'].dt.isocalendar().week.values

    climatology_weekly = ds_climatology_weekly.sel(weekofyear=dataset_weeks).squeeze(dim="ensemble").rename({"cell": "values"})
    climatology_weekly = climatology_weekly.assign_coords(weekofyear=lead_time).rename({"weekofyear": "leadtime"}).transpose("variable","leadtime","values")
    
    #change dim names of inference and dataset to leadtime for easier comparison
    ds_dataset_sliced_weekly = ds_dataset_sliced_weekly.assign_coords(time=lead_time).rename({"time": "leadtime"})
    ds_inference_weekly = ds_inference_weekly.assign_coords(time=lead_time).rename({"time": "leadtime"})
    
    #observed anomalies
    observed_anomalies = climatology_weekly["data"] - ds_dataset_sliced_weekly["data"]
    
    #modeled anomalies
    modeled_anomalies = climatology_weekly["data"] - xr.concat(
        [ds_inference_weekly[var] for var in var_of_interest], 
        dim=xr.DataArray(var_of_interest, dims="variable", name="variable")
    )
    #add anomalies to the run arrays
    observed_anomalies_runs.loc[run, :, :, :] = observed_anomalies.values
    modeled_anomalies_runs.loc[run, :, :, :] = modeled_anomalies.values
    
    #ACC calculation
    print("Calculating ACC...")
    for var, i in zip(var_of_interest, range(len(var_of_interest))):
        acc_results_all.loc[run, var, :] = xr.corr(observed_anomalies.isel(variable=i), modeled_anomalies.isel(variable=i), dim="values")
        
#compute R_t
R_t = {var: [] for var in var_of_interest}
for var, i in zip(var_of_interest, range(len(var_of_interest))):
    R_t[var] = xr.corr(observed_anomalies_runs.isel(variable=i), modeled_anomalies_runs.isel(variable=i), dim="run", weights = lat_weights)

for var in var_of_interest:
    lat = lat_lon_coords["latitude"][1].values.ravel()
    lon = lat_lon_coords["longitude"][1].values.ravel()
    R_t[var] = R_t[var].assign_coords(
        latitude=("values", lat),
        longitude=("values", lon),
    )


#export
output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics"
os.makedirs(output_dir, exist_ok=True)

# Convert to Dataset for easier variable-based access
acc_results_ds = xr.Dataset({var: acc_results_all.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_acc_path = os.path.join(output_dir, "ACC_weekly_anomalies_refmodel_package.nc")
acc_results_ds.to_netcdf(nc_acc_path)

Rt_ds = xr.Dataset({var: R_t[var].drop_vars('variable', errors='ignore') for var in var_of_interest})
nc_Rt_path = os.path.join(output_dir, "Rt_weekly_anomalies_refmodel_package.nc")
Rt_ds.to_netcdf(nc_Rt_path)