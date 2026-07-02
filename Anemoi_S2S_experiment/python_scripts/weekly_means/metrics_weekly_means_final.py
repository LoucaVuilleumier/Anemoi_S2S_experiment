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
from xskillscore import roc, brier_score, reliability
############################################################################################################################################################

# Load preprocessed data
preprocessed_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init"

# Variable of interest
var_of_interest = ["2t", "tp", "10u", "10v"]

# Load latitude weights
print("Loading latitude weights...")
lat_weights_ds = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed/lat_weights.nc")
lat_weights = lat_weights_ds['lat_weights']

# Load observations
print("Loading observations...")
ds_obs = xr.open_dataset(os.path.join(preprocessed_dir, "observations_weekly_27_0.nc"))
ds_obs_anomalies = xr.open_zarr(os.path.join(preprocessed_dir, "observations_anomalies_weekly_27_0.zarr"))
ds_obs_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "observations_climatology_weekly_27_0.nc"))

ds_obs_anomalies_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/observations_anomalies_weekly_27_0_ROBUST.zarr")
ds_obs_climatology_ROBUST = xr.open_dataset("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/observations_climatology_weekly_27_0_ROBUST.nc")

# Get lat/lon coordinates from observations
lat_lon_coords = {
    'latitude': ds_obs['latitude'],
    'longitude': ds_obs['longitude']
}

weekly_init_dates = ds_obs.init_date.values

# Load reference model (if available)
print("Loading reference model...")
ref_path = os.path.join(preprocessed_dir, "reference_model_weekly_27_0.nc")
ds_ref = xr.open_dataset(ref_path)
ds_ref_anomalies = xr.open_zarr(os.path.join(preprocessed_dir, "reference_model_anomalies_weekly_27_0.zarr"))
ds_ref_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "reference_model_climatology_weekly_27_0.nc"))


ds_ref_anomalies_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/reference_model_anomalies_weekly_27_0_ROBUST.zarr")
ds_ref_climatology_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/reference_model_climatology_weekly_27_0_ROBUST.zarr")

#Load weekly finetuned model
print("Loading weekly finetuned model...")
ds_weekly_finetuned = xr.open_dataset(os.path.join(preprocessed_dir, "weekly_finetuned_model_27_0.nc"))
ds_weekly_finetuned_anomalies = xr.open_zarr(os.path.join(preprocessed_dir, "weekly_finetuned_model_anomalies_27_0.zarr"))
ds_weekly_finetuned_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "weekly_finetuned_model_climatology_27_0.nc"))

ds_weekly_finetuned_anomalies_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/weekly_finetuned_model_anomalies_27_0_ROBUST.zarr")
ds_weekly_finetuned_climatology_ROBUST = xr.open_dataset("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/weekly_finetuned_model_climatology_27_0_ROBUST.nc")
#ds_weekly_finetuned["tp"].values = ds_weekly_finetuned["tp"].values * 7

#Load daily finetuned model
print("Loading daily finetuned model...")
ds_daily_finetuned = xr.open_dataset(os.path.join(preprocessed_dir, "daily_finetuned_model_weekly_3_0.nc"))
ds_daily_finetuned_anomalies = xr.open_zarr(os.path.join(preprocessed_dir, "daily_finetuned_model_weekly_anomalies_3_0.zarr"))
ds_daily_finetuned_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "daily_finetuned_model_weekly_climatology_3_0.nc"))

ds_daily_finetuned_anomalies_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/daily_finetuned_model_weekly_anomalies_3_0_ROBUST.zarr")
ds_daily_finetuned_climatology_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/daily_finetuned_model_weekly_climatology_3_0_ROBUST.zarr")
############################################################################################################################################################
#create empty datasets to store metrics results
print("Creating empty datasets for metrics results...")
models = []
preds_list = []
anomalies_preds_list = []
anomalies_preds_list_ROBUST = []

if ds_ref is not None and ds_ref_anomalies is not None:
    models.append("reference")
    preds_list.append(ds_ref)
    anomalies_preds_list.append(ds_ref_anomalies)
    anomalies_preds_list_ROBUST.append(ds_ref_anomalies_ROBUST)
    
if ds_weekly_finetuned is not None and ds_weekly_finetuned_anomalies is not None:
    models.append("weekly_finetuned")
    preds_list.append(ds_weekly_finetuned)
    anomalies_preds_list.append(ds_weekly_finetuned_anomalies)
    anomalies_preds_list_ROBUST.append(ds_weekly_finetuned_anomalies_ROBUST)
    
if ds_daily_finetuned is not None and ds_daily_finetuned_anomalies is not None:
    models.append("daily_finetuned")
    preds_list.append(ds_daily_finetuned)
    anomalies_preds_list.append(ds_daily_finetuned_anomalies)
    anomalies_preds_list_ROBUST.append(ds_daily_finetuned_anomalies_ROBUST)
    
n_ensemble = anomalies_preds_list[0].member.size
n_leadtime = 8  # 8 weeks
n_vars = len(var_of_interest)
n_values = ds_obs[var_of_interest[0]].values.shape[-1]
n_init_dates = ds_obs.init_date.size
weeks_lead_time = np.arange(1, 9)  # Weeks 1-8

acc_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

acc_ROBUST_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

R_t_results =  xr.DataArray(
    data=np.full((len(models),n_vars, n_leadtime, n_values), np.nan),
    dims=("model","variable", "leadtime", "values"),
    coords={
        "model": models,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1),
        "values": range(n_values)
    }
)

R_t_ROBUST_results =  xr.DataArray(
    data=np.full((len(models),n_vars, n_leadtime, n_values), np.nan),
    dims=("model","variable", "leadtime", "values"),
    coords={
        "model": models,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1),
        "values": range(n_values)
    }
)

rmse_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

rmse_anomalies_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

rmse_anomalies_ROBUST_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

spatial_rmse_results = xr.DataArray(
    data=np.full((len(models), n_vars, n_leadtime, n_values), np.nan),
    dims=("model", "variable", "leadtime", "values"),
    coords={
        "model": models,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1),
        "values": range(n_values)
    }
)

spatial_rmse_anomalies_results = xr.DataArray(
    data=np.full((len(models), n_vars, n_leadtime, n_values), np.nan),
    dims=("model", "variable", "leadtime", "values"),
    coords={
        "model": models,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1),
        "values": range(n_values)
    }
)

crps_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)
anomalies_crps_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

anomalies_crps_ROBUST_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

RMS_spread_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model","init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

############################################################################################################################################################
# Compute metrics for each model, variable, and lead time
print("Computing metrics for each model, variable, and lead time...")

print("Computing ACC for NEW models...")
for model, pred_anomalies in zip(models, anomalies_preds_list):
    for init_date in weekly_init_dates:
        for var in var_of_interest:
            acc_results.loc[model, init_date, var, :] = xr.corr(
                ds_obs_anomalies[var].sel(init_date=init_date), 
                pred_anomalies[var].sel(init_date=init_date).mean(dim="member"), 
                dim="values", 
                weights=lat_weights
            ).values
            
for model, pred_anomalies_ROBUST in zip(models, anomalies_preds_list_ROBUST):
    for init_date in weekly_init_dates:
        for var in var_of_interest:
            acc_ROBUST_results.loc[model, init_date, var, :] = xr.corr(
                ds_obs_anomalies_ROBUST[var].sel(init_date=init_date), 
                pred_anomalies_ROBUST[var].sel(init_date=init_date).mean(dim="member"), 
                dim="values", 
                weights=lat_weights
            ).values
            
# R_t computation for NEW models
print("Computing R_t for NEW models...")
for model, pred_anomalies in zip(models, anomalies_preds_list):
    for var in var_of_interest:
        R_t_results.loc[model, var, :, :] = xr.corr(
            ds_obs_anomalies[var], 
            pred_anomalies[var].mean(dim="member"), 
            dim="init_date"
        ).values
        
for model, pred_anomalies_ROBUST in zip(models, anomalies_preds_list_ROBUST):
    for var in var_of_interest:
        R_t_ROBUST_results.loc[model, var, :, :] = xr.corr(
            ds_obs_anomalies_ROBUST[var], 
            pred_anomalies_ROBUST[var].mean(dim="member"), 
            dim="init_date"
        ).values
        
# RMSE computation for NEW models
print("Computing RMSE for NEW models...")
for model, preds in zip(models, preds_list):
    for init_date in weekly_init_dates:
        for var in var_of_interest:
            obs = ds_obs[var].sel(init_date=init_date)
            pred = preds[var].sel(init_date=init_date).mean(dim="member")
            squared_error = (obs - pred) ** 2
            weighted_mse = (squared_error * lat_weights).sum(dim="values") / lat_weights.sum()
            rmse_results.loc[model, init_date, var, :] = np.sqrt(weighted_mse).values

for model, anomalies in zip(models, anomalies_preds_list):
    for init_date in weekly_init_dates:
        for var in var_of_interest:
            obs_anom = ds_obs_anomalies[var].sel(init_date=init_date)
            pred_anom = anomalies[var].sel(init_date=init_date).mean(dim="member")
            squared_error_anom = (obs_anom - pred_anom) ** 2
            weighted_mse_anom = (squared_error_anom * lat_weights).sum(dim="values") / lat_weights.sum()
            rmse_anomalies_results.loc[model, init_date, var, :] = np.sqrt(weighted_mse_anom).values

for model, anomalies_ROBUST in zip(models, anomalies_preds_list_ROBUST):
    for init_date in weekly_init_dates:
        for var in var_of_interest:
            obs_anom_ROBUST = ds_obs_anomalies_ROBUST[var].sel(init_date=init_date)
            pred_anom_ROBUST = anomalies_ROBUST[var].sel(init_date=init_date).mean(dim="member")
            squared_error_anom_ROBUST = (obs_anom_ROBUST - pred_anom_ROBUST) ** 2
            weighted_mse_anom_ROBUST = (squared_error_anom_ROBUST * lat_weights).sum(dim="values") / lat_weights.sum()
            rmse_anomalies_ROBUST_results.loc[model, init_date, var, :] = np.sqrt(weighted_mse_anom_ROBUST).values

# Spatial RMSE computation for NEW models
print("Computing Spatial RMSE for NEW models...")
for model, preds in zip(models, preds_list):
    for t_idx in weeks_lead_time:
        for var in var_of_interest:
            obs = ds_obs[var].sel(week_lead_time=t_idx)
            pred = preds[var].sel(week_lead_time=t_idx)
            pred_mean = pred.mean(dim="member")
            spatial_rmse_results.loc[model, var, t_idx, :] = np.sqrt(((obs - pred_mean) ** 2).mean(dim="init_date")).values
            
for model, anomalies in zip(models, anomalies_preds_list_ROBUST):
    for t_idx in weeks_lead_time:
        for var in var_of_interest:
            obs_anom_ROBUST = ds_obs_anomalies_ROBUST[var].sel(week_lead_time=t_idx)
            pred_anom_ROBUST = anomalies[var].sel(week_lead_time=t_idx)
            pred_anom_ROBUST_mean = pred_anom_ROBUST.mean(dim="member")
            spatial_rmse_anomalies_results.loc[model, var, t_idx, :] = np.sqrt(((obs_anom_ROBUST - pred_anom_ROBUST_mean) ** 2).mean(dim="init_date")).values
    
# CRPS computation for NEW models
print("Computing CRPS for NEW models...")
for model, preds, anomalies in zip(models, preds_list, anomalies_preds_list):
    for init_date in weekly_init_dates:
        for t_idx in weeks_lead_time:
            for var in var_of_interest:
                crps_results.loc[model, init_date, var, t_idx] = crps_for_ensemble(
                    preds[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                    ds_obs[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                    ensemble_member_dim="member",
                    method="fair",
                    weights=lat_weights
                ).values
                
                anomalies_crps_results.loc[model, init_date, var, t_idx] = crps_for_ensemble(
                    anomalies[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                    ds_obs_anomalies[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                    ensemble_member_dim="member",
                    method="fair",
                    weights=lat_weights
                ).values
                
for model, anomalies_robust in zip(models, anomalies_preds_list_ROBUST):
    for init_date in weekly_init_dates:
        for t_idx in weeks_lead_time:
            for var in var_of_interest:
                anomalies_crps_ROBUST_results.loc[model, init_date, var, t_idx] = crps_for_ensemble(
                    anomalies_robust[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                    ds_obs_anomalies_ROBUST[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                    ensemble_member_dim="member",
                    method="fair",
                    weights=lat_weights
                ).values
                
# Spread computation for NEW models
print("Computing Spread for NEW models...")
for model, preds in zip(models, preds_list):
    for var in var_of_interest:
        ensemble_spread = preds[var].std(dim="member")
        spread_squared = ensemble_spread ** 2
        weighted_var = (spread_squared * lat_weights).sum(dim="values") / lat_weights.sum()
        weighted_spread = np.sqrt(weighted_var)
        weighted_spread = weighted_spread.rename({"week_lead_time": "leadtime"})
        weighted_spread = weighted_spread.transpose("init_date", "leadtime")
        RMS_spread_results.loc[model, :, var, :] = weighted_spread.values

print("NEW model metrics computation complete!")

############################################################################################################################################################
#export metrics results

output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("EXPORTING METRICS RESULTS")
print("=" * 80)

# Convert to Dataset for easier variable-based access and then export
#acc
acc_results_ds = xr.Dataset({var: acc_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_acc_path = os.path.join(output_dir, "ACC_weekly_anomalies_AIFS.nc")
# Remove existing file to ensure clean overwrite
if os.path.exists(nc_acc_path):
    os.remove(nc_acc_path)
# Compute values to ensure they're not lazy
acc_results_ds = acc_results_ds.compute()
# Write with explicit format and engine
acc_results_ds.to_netcdf(nc_acc_path, mode='w', format='NETCDF4', engine='netcdf4')
# Explicitly close to flush buffers
acc_results_ds.close()
del acc_results_ds

acc_ROBUST_results_ds = xr.Dataset({var: acc_ROBUST_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_acc_ROBUST_path = os.path.join(output_dir, "ACC_ROBUST_weekly_anomalies_AIFS.nc")
if os.path.exists(nc_acc_ROBUST_path):
    os.remove(nc_acc_ROBUST_path)
acc_ROBUST_results_ds = acc_ROBUST_results_ds.compute()
acc_ROBUST_results_ds.to_netcdf(nc_acc_ROBUST_path, mode='w', format='NETCDF4', engine='netcdf4')
acc_ROBUST_results_ds.close()
del acc_ROBUST_results_ds

#R_t
R_t_results_ds = xr.Dataset({var: R_t_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
R_t_results_ds = R_t_results_ds.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)
nc_R_t_path = os.path.join(output_dir, "R_t_weekly_anomalies_AIFS.nc")
if os.path.exists(nc_R_t_path):
    os.remove(nc_R_t_path)
R_t_results_ds = R_t_results_ds.compute()
R_t_results_ds.to_netcdf(nc_R_t_path, mode='w', format='NETCDF4', engine='netcdf4')
R_t_results_ds.close()
del R_t_results_ds

R_t_ROBUST_results_ds = xr.Dataset({var: R_t_ROBUST_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
R_t_ROBUST_results_ds = R_t_ROBUST_results_ds.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)
nc_R_t_ROBUST_path = os.path.join(output_dir, "R_t_ROBUST_weekly_anomalies_AIFS.nc")
if os.path.exists(nc_R_t_ROBUST_path):
    os.remove(nc_R_t_ROBUST_path)
R_t_ROBUST_results_ds = R_t_ROBUST_results_ds.compute()
R_t_ROBUST_results_ds.to_netcdf(nc_R_t_ROBUST_path, mode='w', format='NETCDF4', engine='netcdf4')
R_t_ROBUST_results_ds.close()
del R_t_ROBUST_results_ds

#rmse
rmse_results_ds = xr.Dataset({var: rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_rmse_path = os.path.join(output_dir, "RMSE_weekly_AIFS.nc")
if os.path.exists(nc_rmse_path):
    os.remove(nc_rmse_path)
rmse_results_ds.to_netcdf(nc_rmse_path, mode='w')


#rmse anomalies
rmse_anomalies_results_ds = xr.Dataset({var: rmse_anomalies_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_rmse_anomalies_path = os.path.join(output_dir, "RMSE_Anomalies_weekly_AIFS.nc")
if os.path.exists(nc_rmse_anomalies_path):
    os.remove(nc_rmse_anomalies_path)
rmse_anomalies_results_ds.to_netcdf(nc_rmse_anomalies_path, mode='w')

rmse_anomalies_ROBUST_results_ds = xr.Dataset({var: rmse_anomalies_ROBUST_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_rmse_anomalies_ROBUST_path = os.path.join(output_dir, "RMSE_Anomalies_ROBUST_weekly_AIFS.nc")
if os.path.exists(nc_rmse_anomalies_ROBUST_path):
    os.remove(nc_rmse_anomalies_ROBUST_path)
rmse_anomalies_ROBUST_results_ds.to_netcdf(nc_rmse_anomalies_ROBUST_path, mode='w')

spatial_rmse_results_ds = xr.Dataset({var: spatial_rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
spatial_rmse_results_ds = spatial_rmse_results_ds.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)
nc_spatial_rmse_path = os.path.join(output_dir, "Spatial_RMSE_weekly_AIFS.nc")
if os.path.exists(nc_spatial_rmse_path):
    os.remove(nc_spatial_rmse_path)
spatial_rmse_results_ds.to_netcdf(nc_spatial_rmse_path, mode='w')

spatial_anomalies_rmse_results_ds = xr.Dataset({var: spatial_rmse_anomalies_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
spatial_anomalies_rmse_results_ds = spatial_anomalies_rmse_results_ds.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)
nc_spatial_anomalies_rmse_path = os.path.join(output_dir, "Spatial_RMSE_Anomalies_weekly_AIFS.nc")
if os.path.exists(nc_spatial_anomalies_rmse_path):
    os.remove(nc_spatial_anomalies_rmse_path)
spatial_anomalies_rmse_results_ds.to_netcdf(nc_spatial_anomalies_rmse_path, mode='w')

#crps
crps_results_ds = xr.Dataset({var: crps_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_crps_path = os.path.join(output_dir, "CRPS_weekly_AIFS.nc")
if os.path.exists(nc_crps_path):
    os.remove(nc_crps_path)
crps_results_ds.to_netcdf(nc_crps_path, mode='w')

anomalies_crps_results_ds = xr.Dataset({var: anomalies_crps_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_anomalies_crps_path = os.path.join(output_dir, "CRPS_Anomalies_weekly_AIFS.nc")
if os.path.exists(nc_anomalies_crps_path):
    os.remove(nc_anomalies_crps_path)
anomalies_crps_results_ds.to_netcdf(nc_anomalies_crps_path, mode='w')

anomalies_crps_ROBUST_results_ds = xr.Dataset({var: anomalies_crps_ROBUST_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_anomalies_crps_ROBUST_path = os.path.join(output_dir, "CRPS_Anomalies_ROBUST_weekly_AIFS.nc")
if os.path.exists(nc_anomalies_crps_ROBUST_path):
    os.remove(nc_anomalies_crps_ROBUST_path)
anomalies_crps_ROBUST_results_ds.to_netcdf(nc_anomalies_crps_ROBUST_path, mode='w')

RMS_spread_results_ds = xr.Dataset({var: RMS_spread_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_RMS_spread_path = os.path.join(output_dir, "RMS_Spread_weekly_AIFS.nc")
if os.path.exists(nc_RMS_spread_path):
    os.remove(nc_RMS_spread_path)
RMS_spread_results_ds.to_netcdf(nc_RMS_spread_path, mode='w')

print("=" * 80)
print("METRICS COMPUTATION COMPLETE")
print(f"All metrics results saved to: {output_dir}")
print("=" * 80)


