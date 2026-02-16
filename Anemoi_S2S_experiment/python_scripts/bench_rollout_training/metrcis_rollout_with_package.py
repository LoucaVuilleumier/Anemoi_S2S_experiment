import xarray as xr
import numpy as np
import pandas as pd
from nwpeval import rmse, mae, acc, pod, fss, sedi
import glob
import os
from scores.categorical import probability_of_detection, probability_of_false_detection
from scores.plotdata import roc
from scipy.stats import beta, binom
from matplotlib import pyplot as plt



#dataset paths
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
ds_dataset = xr.open_zarr(dataset_path)
times = ds_dataset.dates.values
var_names = ds_dataset.attrs["variables"]

# Get only the indices of variables we need
var_of_interest = ["2t", "tp", "10u", "10v"]
var_indices = [var_names.index(var) for var in var_of_interest]

#load climatology
climatology_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/weekly_climatology_1979-2019.nc"
ds_climatology_weekly = xr.open_dataset(climatology_path)

#load thresholds
thresholds_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/thresholds_1979-2019.nc"
ds_thresholds = xr.open_dataset(thresholds_path)
ds_thresholds = ds_thresholds["data"].squeeze(dim="ensemble").rename({"cell": "values"}).assign_coords(variable=var_of_interest).transpose("variable","values")

lat_lon_coords = {
            'latitude': ds_climatology_weekly ['latitudes'],
            'longitude': ds_climatology_weekly ['longitudes']
        }
lat_weights = np.cos(np.radians(lat_lon_coords["latitude"][1].values))
lat_weights = xr.DataArray(lat_weights, dims=["values"])


n_runs = 1
n_vars = len(var_of_interest)
n_leadtime = 8
n_values = 40320 

#create array to store for all runs
#ACC
acc_results_all = xr.DataArray(
    data=np.full((n_runs, n_vars, n_leadtime), np.nan),
    dims=("run", "variable", "leadtime"),
    coords={
        "run": range(n_runs),
        "variable": var_of_interest,
        "leadtime": range(1, 9)
    }
)

#temporal RMSE
RMSE_temp = xr.DataArray(
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

#create array to store raw weekly values for all runs
observed_weekly_runs = xr.DataArray(
    data=np.full((n_runs, n_vars, n_leadtime, n_values), np.nan),
    dims=("run", "variable", "leadtime", "values"),
    coords={
        "run": range(n_runs),
        "variable": var_of_interest,
        "leadtime": range(1, 9)
    }
)

modeled_weekly_runs = xr.DataArray(
    data=np.full((n_runs, n_vars, n_leadtime, n_values), np.nan),
    dims=("run", "variable", "leadtime", "values"),
    coords={
        "run": range(n_runs),
        "variable": var_of_interest,
        "leadtime": range(1, 9)
    }
)

#SEDI
SEDI_temp = xr.DataArray(
    data=np.full((n_runs, n_vars, n_leadtime), np.nan),
    dims=("run", "variable", "leadtime"),
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
    ds_dataset_sliced_weekly = ds_dataset_sliced_weekly.assign_coords(time=lead_time, variable=var_of_interest).rename({"time": "leadtime"})
    ds_inference_weekly = ds_inference_weekly.assign_coords(time=lead_time, variable=var_of_interest).rename({"time": "leadtime"})
    
    #observed anomalies
    observed_anomalies = climatology_weekly["data"] - ds_dataset_sliced_weekly["data"]
    
    #modeled anomalies
    modeled_anomalies = climatology_weekly["data"] - xr.concat(
        [ds_inference_weekly[var] for var in var_of_interest], 
        dim=xr.DataArray(var_of_interest, dims="variable", name="variable")
    )
    modeled_weekly = xr.concat(
        [ds_inference_weekly[var] for var in var_of_interest],
        dim=xr.DataArray(var_of_interest, dims="variable", name="variable")
    )
    #add anomalies to the run arrays
    observed_anomalies_runs.loc[run, :, :, :] = observed_anomalies.values
    modeled_anomalies_runs.loc[run, :, :, :] = modeled_anomalies.values
    #add raw weekly values to the run arrays
    observed_weekly_runs.loc[run, :, :, :] = ds_dataset_sliced_weekly["data"].transpose("variable", "leadtime", "values").values
    modeled_weekly_runs.loc[run, :, :, :] = modeled_weekly.transpose("variable", "leadtime", "values").values
    
    #ACC calculation
    print("Calculating ACC...")
    for var, i in zip(var_of_interest, range(len(var_of_interest))):
        acc_results_all.loc[run, var, :] = xr.corr(observed_anomalies.isel(variable=i), modeled_anomalies.isel(variable=i), dim="values", weights = lat_weights)
        
    
    #RMSE calculation
    print("Calculating RMSE...")
    for var, i in zip(var_of_interest, range(len(var_of_interest))):
        RMSE_temp.loc[run, var, :] = rmse(ds_dataset_sliced_weekly["data"].isel(variable=i), ds_inference_weekly[var], dim="values")
        
    #SEDI calculation
    print("Calculating SEDI...")
    for var, i in zip(var_of_interest, range(len(var_of_interest))):
        threshold = ds_thresholds.sel(variable=var).values
        obs = ds_dataset_sliced_weekly["data"].isel(variable=i)
        model = ds_inference_weekly[var]
        SEDI_temp.loc[run, var, :] = sedi(obs, model, threshold=threshold, dim="values")
        
#compute R_t
R_t = {var: [] for var in var_of_interest}
for var, i in zip(var_of_interest, range(len(var_of_interest))):
    R_t[var] = xr.corr(observed_anomalies_runs.isel(variable=i), modeled_anomalies_runs.isel(variable=i), dim="run")

#Compute spatial RMSE for each variable and lead time
RMSE_spatial = {var: [] for var in var_of_interest}
for var, i in zip(var_of_interest, range(len(var_of_interest))):
    RMSE_spatial[var] = rmse(observed_anomalies_runs.isel(variable=i), modeled_anomalies_runs.isel(variable=i), dim=["run"])

#Compute spatial SEDI
SEDI_spatial = {var: [] for var in var_of_interest}
for var, i in zip(var_of_interest, range(len(var_of_interest))):
    threshold = ds_thresholds.sel(variable=var).values
    obs = observed_weekly_runs.isel(variable=i)
    model = modeled_weekly_runs.isel(variable=i)
    SEDI_spatial[var] = sedi(obs, model, threshold=threshold, dim=["run"])

#assign coordinate
for var in var_of_interest:
    lat = lat_lon_coords["latitude"][1].values.ravel()
    lon = lat_lon_coords["longitude"][1].values.ravel()
    R_t[var] = R_t[var].assign_coords(
        latitude=("values", lat),
        longitude=("values", lon),
    )
    
#Compute binary dataset for probability of detection and false detection
binary_obs = xr.where(observed_weekly_runs > ds_thresholds, 1, 0)
binary_model = xr.where(modeled_weekly_runs >  ds_thresholds, 1, 0)

roc_data = roc(binary_model.sel(variable="2t", leadtime = 2, run = 0), binary_obs.sel(variable="2t", leadtime = 2, run = 0), [0.00001, 0.1, .5, 0.9, .99999, 1])

#export
output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics"
os.makedirs(output_dir, exist_ok=True)

# Convert to Dataset for easier variable-based access and then export
#acc
acc_results_ds = xr.Dataset({var: acc_results_all.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_acc_path = os.path.join(output_dir, "ACC_weekly_anomalies_refmodel_package.nc")
acc_results_ds.to_netcdf(nc_acc_path)

#R_t
Rt_ds = xr.Dataset({var: R_t[var].drop_vars('variable', errors='ignore') for var in var_of_interest})
nc_Rt_path = os.path.join(output_dir, "Rt_weekly_anomalies_refmodel_package.nc")
Rt_ds.to_netcdf(nc_Rt_path)

#temporal RMSE 
RMSE_temp_ds = xr.Dataset({var: RMSE_temp.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_RMSE_temp_path = os.path.join(output_dir, "RMSE_weekly.nc")
RMSE_temp_ds.to_netcdf(nc_RMSE_temp_path)

#Spatial RMSE 
RMSE_spatial_ds = xr.Dataset({var: RMSE_spatial[var].drop_vars('variable', errors='ignore') for var in var_of_interest})
nc_RMSE_spatial_path = os.path.join(output_dir, "RMSE_spatial_weekly.nc")
RMSE_spatial_ds.to_netcdf(nc_RMSE_spatial_path)

#temporal SEDI
SEDI_temp_ds = xr.Dataset({var: SEDI_temp.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_SEDI_temp_path = os.path.join(output_dir, "SEDI_weekly.nc")
SEDI_temp_ds.to_netcdf(nc_SEDI_temp_path)

#Spatial SEDI
SEDI_spatial_ds = xr.Dataset({var: SEDI_spatial[var].drop_vars('variable', errors='ignore') for var in var_of_interest})
nc_SEDI_spatial_path = os.path.join(output_dir, "SEDI_spatial_weekly.nc")
SEDI_spatial_ds.to_netcdf(nc_SEDI_spatial_path)

fig = plt.figure(figsize=(10, 6))
plt.title("Receiver Operating Characteristic")
plt.plot(roc_data.POFD, roc_data.POD, "b", label="AUC = %0.2f" % roc_data.AUC.item())
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("POD")
plt.xlabel("POFD")
plt.savefig("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()