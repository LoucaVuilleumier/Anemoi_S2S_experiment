import xarray as xr
import numpy as np
import pandas as pd
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
#load dataset with observations
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-6h-v1-for-single-v2.zarr"
obs = xr.open_zarr(dataset_path)
times = obs.dates.values
obs = obs.assign_coords(time=times).rename({"cell": "values"})



#select only a substet of variable
var_names = obs.attrs["variables"]
var_of_interest = ["2t", "tp", "10u", "10v"]
var_indices = [var_names.index(var) for var in var_of_interest]
obs = obs.isel(variable=var_indices)

#load weekly climatology
climatology_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/weekly_climatology_1979-2019.nc"
ds_climatology_weekly = xr.open_dataset(climatology_path)

#load weekly thresholds per grid cell
thresholds_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/thresholds_1979-2019.nc"
thresholds = xr.open_dataset(thresholds_path)
thresholds = thresholds["data"].squeeze(dim="ensemble").rename({"cell": "values"}).assign_coords(variable=var_of_interest).transpose("variable","values")
ds_thresholds = xr.Dataset({var: thresholds.sel(variable=var).drop_vars('variable') for var in var_of_interest})


#create latitudinal weights for the metrics
lat_lon_coords = {
            'latitude': obs['latitudes'],
            'longitude': obs['longitudes']
        }
lat_weights = np.cos(np.radians(lat_lon_coords["latitude"].values))
lat_weights = xr.DataArray(lat_weights, dims=["values"])

#paths of the different initialization dates for the forecasts
paths_init = sorted(glob.glob(
    "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/AIFS/"
    "aifs-subs-pretrain-*"
))

# Load all reforecasts with an additional init_date dimension
reforecast_datasets = []
init_dates = []

for path_init in paths_init:
    # Get all member paths for this initialization date
    paths_member = sorted(glob.glob(f"{path_init}/*.nc"))
    
    # Load all members for this init date
    ds_init = xr.open_mfdataset(
        paths_member,
        combine="nested",
        concat_dim="member",
        parallel=False,  
        chunks={}        
    )
    
    # Extract initialization date from the dataset
    init_date = ds_init.time.values[0].astype('datetime64[ns]')
    init_dates.append(init_date)
    
    # Store the absolute time as a non-dimension coordinate
    ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
    
    # Replace time dimension with relative lead_time (in days from init)
    lead_times = (ds_init.time.values - init_date) / np.timedelta64(1, 'D')
    ds_init = ds_init.assign_coords(time=lead_times).rename({'time': 'lead_time'})
    
    reforecast_datasets.append(ds_init)

# Concatenate all reforecasts along a new 'init_date' dimension
ds_inf_daily = xr.concat(
    reforecast_datasets, 
    dim=xr.DataArray(init_dates, dims='init_date', name='init_date')
)

print(f"Loaded {len(init_dates)} reforecasts")
print(f"Dataset dimensions: {dict(ds_inf_daily.dims)}")

#select only variables of interest for the forecast
ds_inf_daily = ds_inf_daily[var_of_interest]

#Create observations dataset with the same structure as the forecasts for easier metrics computation
ds_obs_multiple_init = []

for init_date in init_dates:
    t0 = np.where(times == init_date)[0][0]
    t1 = np.where(times == ds_inf_daily.sel(init_date = init_date).forecast_time[-1].values)[0][0]
    ds_init = obs.isel(time=slice(t0, t1 + 1))
    
    # Store the absolute time as a non-dimension coordinate
    ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
    
    #select only forecast at 12:00
    ds_init_daily = ds_init.sel(time=ds_init.time.dt.hour == 12).squeeze("ensemble")
    ds_init_daily = ds_init_daily.assign_coords(variable=var_of_interest)
    
    # Replace time dimension with relative lead_time (in days from init)
    lead_times = (ds_init_daily.time.values - init_date) / np.timedelta64(1, 'D')
    ds_init_daily = ds_init_daily.assign_coords(time=lead_times).rename({'time': 'lead_time'})
    
    # Convert from 'data' variable with variable dimension to separate variables
    xr_ds_init = xr.Dataset({var: ds_init_daily['data'].sel(variable=var).drop_vars('variable') for var in var_of_interest})
    ds_obs_multiple_init.append(xr_ds_init)

ds_obs_daily = xr.concat(ds_obs_multiple_init, dim=xr.DataArray(init_dates, dims='init_date', name='init_date'))


#compute weekly means (7 days per week, 1-day timesteps = 7 timesteps per week)
#for the forecasts: (init_date, member, lead_time, values)
ds_inf_weekly = ds_inf_daily.coarsen(lead_time=7, boundary='trim').mean()
ds_inf_weekly = ds_inf_weekly.rename({"lead_time": "week_lead_time"}).assign_coords(week_lead_time=np.arange(1,9))

#for the observations: (init_date, lead_time, var, values)  
ds_obs_weekly = ds_obs_daily.coarsen(lead_time=7, boundary='trim').mean()
ds_obs_weekly = ds_obs_weekly.rename({"lead_time": "week_lead_time"}).assign_coords(week_lead_time=np.arange(1,9))

ds_clim_multiple__inf_init = []
ds_clim_multiple_obs_init = []
#compute climatology per member
for init_date in ds_inf_weekly.init_date.values:
    year = pd.to_datetime(init_date).year
    month = pd.to_datetime(init_date).month
    day = pd.to_datetime(init_date).day
    # Select the same calendar month and day across all years for the climatology
    mask = (ds_inf_weekly.init_date.dt.month == month) & (ds_inf_weekly.init_date.dt.day == day) & (ds_inf_weekly.init_date.dt.year != year)
    
    #build climatology per member for inference
    climatology_weekly_inf = ds_inf_weekly.isel(init_date=mask).mean(dim="init_date")
    ds_clim_multiple__inf_init.append(climatology_weekly_inf)
    
    #build climatology for observations
    climatology_weekly_obs = ds_obs_weekly.isel(init_date=mask).mean(dim="init_date")
    ds_clim_multiple_obs_init.append(climatology_weekly_obs)
    
ds_climatology_weekly_inf = xr.concat(ds_clim_multiple__inf_init, dim=xr.DataArray(init_dates, dims='init_date', name='init_date'))
ds_climatology_weekly_obs = xr.concat(ds_clim_multiple_obs_init, dim=xr.DataArray(init_dates, dims='init_date', name='init_date'))



#observed anomalies
observed_anomalies = ds_obs_weekly - ds_climatology_weekly_obs

#predicted anomalies
predicted_anomalies = ds_inf_weekly - ds_climatology_weekly_inf

############################################################################################################################################################
#metrics
n_ensemble = ds_inf_weekly.member.size
n_leadtime = 8
n_vars = len(var_of_interest)
n_values = ds_obs_weekly[var_of_interest[0]].values.shape[-1]
n_init_dates = ds_inf_weekly.init_date.size
weeks_lead_time = np.arange(1,9)

#create empty array to store the results of the metrics
acc_results = xr.DataArray(
    data=np.full((n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("init_date", "variable", "leadtime"),
    coords={
        "init_date": ds_inf_weekly.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

R_t_results =  xr.DataArray(
    data=np.full((n_vars, n_leadtime, n_values), np.nan),
    dims=("variable", "leadtime", "values"),
    coords={
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1),
        "values": range(n_values)
    }
)

rmse_results = xr.DataArray(
    data=np.full((n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("init_date", "variable", "leadtime"),
    coords={
        "init_date": ds_inf_weekly.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

spatial_rmse_results = xr.DataArray(
    data=np.full((n_vars, n_leadtime, n_values), np.nan),
    dims=("variable", "leadtime", "values"),
    coords={
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1),
        "values": range(n_values)
    }
)


# Compute CRPS for all init dates, weeks and variables
crps_results = xr.DataArray(
    data=np.full((n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("init_date", "variable", "leadtime"),
    coords={
        "init_date": ds_inf_weekly.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

#Spread/Skill for all init dates, variables and lead times
spead_skill_results = xr.DataArray(
    data=np.full((n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("init_date", "variable", "leadtime"),
    coords={
        "init_date": ds_inf_weekly.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

#acc computation
for init_date in init_dates:
    for var in var_of_interest:
        acc_results.loc[init_date, var, :] = xr.corr(observed_anomalies[var].sel(init_date=init_date), predicted_anomalies[var].sel(init_date=init_date).mean(dim="member"), dim="values", weights = lat_weights).values

#R_t computation
for var in var_of_interest:
    R_t_results.loc[var, :, :] = xr.corr(observed_anomalies[var], predicted_anomalies[var].mean(dim="member"), dim="init_date").values

#rmse computation with latitude weighting
for init_date in init_dates:
    for var in var_of_interest:
        obs = ds_obs_weekly[var].sel(init_date=init_date)
        pred = ds_inf_weekly[var].sel(init_date=init_date).mean(dim="member")
        squared_error = (obs - pred) ** 2
        weighted_mse = (squared_error * lat_weights).sum(dim="values") / lat_weights.sum()
        rmse_results.loc[init_date, var, :] = np.sqrt(weighted_mse).values

for t_idx in weeks_lead_time:  
    for var in var_of_interest:    
        obs = ds_obs_weekly[var].sel(week_lead_time=t_idx)       # (init_time, values,)
        pred = ds_inf_weekly[var].sel(week_lead_time=t_idx)         # (init_time, member, values)
        pred_mean = pred.mean(dim="member")              # (init_time, values)
        spatial_rmse_results.loc[var, t_idx, :] = np.sqrt(((obs - pred_mean) ** 2).mean(dim="init_date")).values


#Compute binary dataset for probability of detection and false detection
binary_obs = xr.where(ds_obs_weekly > ds_thresholds, 1, 0)
binary_model = xr.where(ds_inf_weekly >  ds_thresholds, 1, 0)
prob_model = binary_model.mean(dim="member")

# Compute ROC data for all weeks and variables using compact loops
p_threshold = np.linspace(0, 1, 8)
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

#compute crps for the different weeks and variables
for init_date in init_dates:
    for t_idx in weeks_lead_time:
        for var in var_of_interest:
            crps_results.loc[init_date, var, t_idx] = crps_for_ensemble(
                ds_inf_weekly[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                ds_obs_weekly[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                ensemble_member_dim="member",
                method="fair",
                weights=lat_weights
            ).values
            

#Computation of Spread/Skill ratio for the different weeks and variables

for var in var_of_interest:
    ensemble_spread = ds_inf_weekly[var].std(dim="member")
    # Spatially average the spread with latitude weighting
    weighted_spread = (ensemble_spread * lat_weights).sum(dim="values") / lat_weights.sum()
    weighted_spread = weighted_spread.rename({"week_lead_time": "leadtime"})
    ensemble_skill = rmse_results.loc[:, var, :]
    spead_skill_results.loc[:, var, :] = (weighted_spread / ensemble_skill).values
############################################################################################################################################################
#export

output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS"
os.makedirs(output_dir, exist_ok=True)

# Convert to Dataset for easier variable-based access and then export
#acc
acc_results_ds = xr.Dataset({var: acc_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_acc_path = os.path.join(output_dir, "ACC_weekly_anomalies_AIFS.nc")
acc_results_ds.to_netcdf(nc_acc_path)

#R_t
R_t_results_ds = xr.Dataset({var: R_t_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
R_t_results_ds = R_t_results_ds.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)
nc_R_t_path = os.path.join(output_dir, "R_t_weekly_anomalies_AIFS.nc")
R_t_results_ds.to_netcdf(nc_R_t_path)

#rmse
rmse_results_ds = xr.Dataset({var: rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_rmse_path = os.path.join(output_dir, "RMSE_weekly_AIFS.nc")
rmse_results_ds.to_netcdf(nc_rmse_path) 

spatial_rmse_results_ds = xr.Dataset({var: spatial_rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
spatial_rmse_results_ds = spatial_rmse_results_ds.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)
nc_spatial_rmse_path = os.path.join(output_dir, "Spatial_RMSE_weekly_AIFS.nc")
spatial_rmse_results_ds.to_netcdf(nc_spatial_rmse_path)


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

#Spread/Skill
spead_skill_results_ds = xr.Dataset({var: spead_skill_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_spead_skill_path = os.path.join(output_dir, "Spread_Skill_weekly_AIFS.nc")
spead_skill_results_ds.to_netcdf(nc_spead_skill_path)