import xarray as xr
import numpy as np
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import metrics_function as mf
import glob
import os
from nwpeval import rmse, mae, acc, pod, fss, sedi
from scores.plotdata import roc
from scores.probability import crps_for_ensemble
import importlib
from utils import metrics_function as mf
importlib.reload(mf)
from matplotlib import pyplot as plt
############################################################################################################################################################
#loading and preprocessing the data

#load dataset with forecasts
paths = sorted(glob.glob(
    "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/AIFS/"
    "aifs-subs-pretrain-july-2025/aifs-subs-pretrain-july-2025-member-*.nc"
))

ds_inf_daily = xr.open_mfdataset(
    paths,
    combine="nested",
    concat_dim="member",
    parallel=False,  # set True if Dask is configured
    chunks={}        # optional: set chunking if you want
)


#load dataset with observations
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-6h-v1-for-single-v2.zarr"
ds_obs = xr.open_zarr(dataset_path)
times = ds_obs.dates.values

#load weekly climatology
climatology_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/weekly_climatology_1979-2019.nc"
ds_climatology_weekly = xr.open_dataset(climatology_path)


#create latitudinal weights for the metrics
lat_lon_coords = {
            'latitude': ds_obs ['latitudes'],
            'longitude': ds_obs ['longitudes']
        }
lat_weights = np.cos(np.radians(lat_lon_coords["latitude"].values))
lat_weights = xr.DataArray(lat_weights, dims=["values"])

#init date
init_date = ds_inf_daily.time.values[0].astype('datetime64[s]')
t0 = np.where(times == init_date)[0][0]

#last date
last_date = ds_inf_daily.time.values[-1].astype('datetime64[s]')
t1 = np.where(times == last_date)[0][0]

#slice observations to match the time range of the forecasts
ds_obs = ds_obs.isel(time=slice(t0, t1 + 1))
time_sliced = ds_obs.dates.values

ds_obs = ds_obs.assign_coords(time=time_sliced).rename({"cell": "values"})

#daily average of observations
ds_obs_daily = ds_obs.resample(time='1D').mean()


#select only a substet of variable
var_names = ds_obs.attrs["variables"]
var_of_interest = ["2t", "tp", "10u", "10v"]
var_indices = [var_names.index(var) for var in var_of_interest]

#for the forecast
ds_inf_daily = ds_inf_daily[var_of_interest]

#for the observations
ds_obs_daily = ds_obs_daily.isel(variable=var_indices)

#load weekly thresholds per grid cell
thresholds_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/thresholds_1979-2019.nc"
thresholds = xr.open_dataset(thresholds_path)
thresholds = thresholds["data"].squeeze(dim="ensemble").rename({"cell": "values"}).assign_coords(variable=var_of_interest).transpose("variable","values")
ds_thresholds = xr.Dataset({var: thresholds.sel(variable=var).drop_vars('variable') for var in var_of_interest})


#compute weekly means 
#for the forecasts
ds_inf_weekly = ds_inf_daily.resample(time='7D').mean()

#for the observations
ds_obs_weekly = ds_obs_daily.resample(time='7D').mean()

#remove the last incomplete week
ds_inf_weekly = ds_inf_weekly.isel(time=slice(0, 8))
ds_obs_weekly = ds_obs_weekly.isel(time=slice(0, 8))

# Reshape ds_obs_weekly to match ds_inf_weekly structure
# Extract the data variable and squeeze ensemble dimension
obs_data = ds_obs_weekly['data'].squeeze('ensemble')  # (time, variable, values)

# Split variable dimension into separate data variables
ds_obs_weekly = xr.Dataset({
    var_name: obs_data.isel(variable=i)
    for i, var_name in enumerate(var_of_interest)
})

time_weekly = ds_obs_weekly.time.values

#compute anomalies
weeks_of_interest = ds_obs_weekly['time'].dt.isocalendar().week.values

#slice the climatology to match the weeks of interest
ds_climatology_weekly = ds_climatology_weekly.isel(weekofyear=weeks_of_interest)
clim_data = ds_climatology_weekly['data'].squeeze('ensemble')  # (weekofyear, variable, values)
ds_climatology_weekly = xr.Dataset({
    var_name: clim_data.isel(variable=i)
    for i, var_name in enumerate(var_of_interest)
})

ds_climatology_weekly = ds_climatology_weekly.rename({"cell": "values"}).rename({"weekofyear": "time"}).assign_coords(time=time_weekly)

#observed anomalies
observed_anomalies = ds_obs_weekly - ds_climatology_weekly

#predicted anomalies
predicted_anomalies = ds_inf_weekly - ds_climatology_weekly

############################################################################################################################################################
#metrics
n_ensemble = ds_inf_weekly.member.size
n_leadtime = len(time_weekly)
n_vars = len(var_of_interest)
n_values = ds_obs_weekly[var_of_interest[0]].values.shape[-1]

#create empty array to store the results of the metrics
acc_results = xr.DataArray(
    data=np.full((n_vars, n_leadtime), np.nan),
    dims=("variable", "time"),
    coords={
        "variable": var_of_interest,
        "time": time_weekly
    }
)

rmse_results = xr.DataArray(
    data=np.full((n_vars, n_leadtime), np.nan),
    dims=("variable", "time"),
    coords={
        "variable": var_of_interest,
        "time": time_weekly
    }
)

spatial_rmse_results = xr.DataArray(
    data=np.full((n_vars, n_leadtime, n_values), np.nan),
    dims=("variable", "leadtime", "values"),
    coords={
        "variable": var_of_interest,
        "leadtime": time_weekly,
        "values": range(n_values)
    }
)

sedi_results = xr.DataArray(
    data=np.full(( n_vars, n_leadtime), np.nan),
    dims=("variable", "time"),
    coords={
        "variable": var_of_interest,
        "time": time_weekly
    }
)

# Compute CRPS for all weeks and variables
crps_results = xr.DataArray(
    data=np.full((n_vars, n_leadtime), np.nan),
    dims=("variable", "time"),
    coords={
        "variable": var_of_interest,
        "time": time_weekly
    }
)

#acc computation
for var in var_of_interest:
    acc_results.loc[var, :] = xr.corr(observed_anomalies[var], predicted_anomalies[var].mean(dim="member"), dim="values", weights = lat_weights).values


#rmse computation
for var in var_of_interest:
    rmse_results.loc[var, :] =rmse(ds_obs_weekly[var], ds_inf_weekly[var].mean(dim="member"), dim="values").values

for t_idx in range(n_leadtime):  
    for var in var_of_interest:    
        obs = ds_obs_weekly[var].isel(time=t_idx)       # (values,)
        pred = ds_inf_weekly[var].isel(time=t_idx)       # (member, values)
        pred_mean = pred.mean(dim="member")              # (values,)
        spatial_rmse_results.loc[var, time_weekly[t_idx], :] = np.abs(obs - pred_mean).values

#sedi computation
for var in var_of_interest:
    sedi_results.loc[var, :] = mf.sedi_ensemble(ds_obs_weekly[var], ds_inf_weekly[var], ds_thresholds[var], dim = "values").values
    

#Compute binary dataset for probability of detection and false detection
binary_obs = xr.where(ds_obs_weekly > ds_thresholds, 1, 0)
binary_model = xr.where(ds_inf_weekly >  ds_thresholds, 1, 0)
prob_model = binary_model.mean(dim="member")

# Compute ROC data for all weeks and variables using compact loops
p_threshold = np.linspace(0, 1, 50)
roc_data = {}

for t_idx in range(n_leadtime):
    week_key = f"week {t_idx}"
    roc_data[week_key] = {}
    for var in var_of_interest:
        roc_data[week_key][var] = roc(
            prob_model[var].isel(time=t_idx).compute(), 
            binary_obs[var].isel(time=t_idx).compute(), 
            p_threshold
        )


for t_idx in range(n_leadtime):
    for var in var_of_interest:
        crps_results.loc[var, time_weekly[t_idx]] = crps_for_ensemble(
            ds_inf_weekly[var].isel(time=t_idx).compute(), 
            ds_obs_weekly[var].isel(time=t_idx).compute(), 
            ensemble_member_dim="member",
            method="fair",
            weights=lat_weights
        ).values
############################################################################################################################################################
#export

output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS"
os.makedirs(output_dir, exist_ok=True)

# Convert to Dataset for easier variable-based access and then export
#acc
acc_results_ds = xr.Dataset({var: acc_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_acc_path = os.path.join(output_dir, "ACC_weekly_anomalies_AIFS.nc")
acc_results_ds.to_netcdf(nc_acc_path)

#rmse
rmse_results_ds = xr.Dataset({var: rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_rmse_path = os.path.join(output_dir, "RMSE_weekly_anomalies_AIFS.nc")
rmse_results_ds.to_netcdf(nc_rmse_path) 

spatial_rmse_results_ds = xr.Dataset({var: spatial_rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
spatial_rmse_results_ds = spatial_rmse_results_ds.assign_coords(
    latitude=("values", ds_obs.latitudes.values),
    longitude=("values", ds_obs.longitudes.values),)
nc_spatial_rmse_path = os.path.join(output_dir, "Spatial_RMSE_weekly_anomalies_AIFS.nc")
spatial_rmse_results_ds.to_netcdf(nc_spatial_rmse_path)

#sedi
sedi_results_ds = xr.Dataset({var: sedi_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_sedi_path = os.path.join(output_dir, "SEDI_weekly_anomalies_AIFS.nc")
sedi_results_ds.to_netcdf(nc_sedi_path)

#roc - save each week separately as they contain different variables and structures
for week_key, week_data in roc_data.items():
    # Merge all variables for this week into one dataset
    week_ds = xr.merge([week_data[var].rename({metric: f"{var}_{metric}" for metric in week_data[var].data_vars}) 
                        for var in var_of_interest])
    
    # Remove attributes that can't be serialized to NetCDF
    for var_name in week_ds.data_vars:
        week_ds[var_name].attrs = {k: v for k, v in week_ds[var_name].attrs.items() 
                                   if isinstance(v, (str, int, float, list, tuple, bytes, np.ndarray))}
    
    nc_roc_week_path = os.path.join(output_dir, f"ROC_{week_key.replace(' ', '_')}_AIFS.nc")
    week_ds.to_netcdf(nc_roc_week_path)
    
#crps
crps_results_ds = xr.Dataset({var: crps_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_crps_path = os.path.join(output_dir, "CRPS_weekly_AIFS.nc")
crps_results_ds.to_netcdf(nc_crps_path)