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

print("=" * 80)
print("LOADING PREPROCESSED DATA FOR METRICS COMPUTATION")
print("=" * 80)

# Load preprocessed data
preprocessed_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed"

# Variable of interest
var_of_interest = ["2t", "tp", "10u", "10v"]

# Load latitude weights
print("Loading latitude weights...")
lat_weights_ds = xr.open_dataset(os.path.join(preprocessed_dir, "lat_weights.nc"))
lat_weights = lat_weights_ds['lat_weights']

# Load observations
print("Loading observations...")
ds_obs_weekly = xr.open_dataset(os.path.join(preprocessed_dir, "observations_weekly.nc"))
observed_anomalies = xr.open_dataset(os.path.join(preprocessed_dir, "observations_anomalies_weekly.nc"))
ds_climatology_weekly_obs = xr.open_dataset(os.path.join(preprocessed_dir, "observations_climatology_weekly.nc"))

# Load observations from daily (for comparison with daily finetuned model)
print("Loading observations from daily (for daily finetuned model)...")
ds_obs_weekly_from_daily = xr.open_dataset(os.path.join(preprocessed_dir, "observations_weekly_from_daily.nc"))
observed_anomalies_from_daily = xr.open_dataset(os.path.join(preprocessed_dir, "observations_anomalies_weekly_from_daily.nc"))
ds_climatology_weekly_obs_from_daily = xr.open_dataset(os.path.join(preprocessed_dir, "observations_climatology_weekly_from_daily.nc"))

# Get lat/lon coordinates from observations
lat_lon_coords = {
    'latitude': ds_obs_weekly['latitude'],
    'longitude': ds_obs_weekly['longitude']
}

weekly_init_dates = ds_obs_weekly.init_date.values

# Load reference model (if available)
ref_path = os.path.join(preprocessed_dir, "reference_model_weekly.nc")
if os.path.exists(ref_path):
    print(f"Loading reference model from {ref_path}")
    ds_inf_weekly = xr.open_dataset(ref_path)
    
    # Try to load anomalies and climatology (may not exist if insufficient init dates)
    ref_anom_path = os.path.join(preprocessed_dir, "reference_model_anomalies_weekly.nc")
    ref_clim_path = os.path.join(preprocessed_dir, "reference_model_climatology_weekly.nc")
    
    if os.path.exists(ref_anom_path):
        predicted_anomalies = xr.open_dataset(ref_anom_path)
    else:
        print("  Warning: Reference model anomalies not found (likely insufficient init dates for climatology)")
        predicted_anomalies = None
    
    if os.path.exists(ref_clim_path):
        ds_climatology_weekly_inf = xr.open_dataset(ref_clim_path)
    else:
        print("  Warning: Reference model climatology not found (likely insufficient init dates)")
        ds_climatology_weekly_inf = None
else:
    print("Reference model not found in preprocessed data")
    ds_inf_weekly = None
    predicted_anomalies = None
    ds_climatology_weekly_inf = None

# Load daily finetuned model (if available)
daily_path = os.path.join(preprocessed_dir, "daily_finetuned_model_weekly.nc")
if os.path.exists(daily_path):
    print(f"Loading daily finetuned model from {daily_path}")
    ds_inf_daily_weekly = xr.open_dataset(daily_path)
    predicted_anomalies_daily = xr.open_dataset(os.path.join(preprocessed_dir, "daily_finetuned_model_anomalies_weekly.nc"))
    ds_climatology_weekly_inf_daily = xr.open_dataset(os.path.join(preprocessed_dir, "daily_finetuned_model_climatology_weekly.nc"))
else:
    print("Daily finetuned model not found in preprocessed data")
    ds_inf_daily_weekly = None
    predicted_anomalies_daily = None
    ds_climatology_weekly_inf_daily = None

# Load weekly finetuned models
finetune_models = {}
finetune_models_list = ["Weekly_Means_14k_lr_0.625e-5", "Weekly_Means_14k_lr_0.625e-7", "Weekly_Means_18k_lr_0.625e-6"]

predicted_anomalies_finetuning_14k_e5 = None
predicted_anomalies_finetuning_14k_e7 = None
predicted_anomalies_finetuning_18k_e6 = None
ds_climatology_weekly_inf_finetuning_14k_e5 = None
ds_climatology_weekly_inf_finetuning_14k_e7 = None
ds_climatology_weekly_inf_finetuning_18k_e6 = None

for model_name in finetune_models_list:
    model_path = os.path.join(preprocessed_dir, f"{model_name}_weekly.nc")
    if os.path.exists(model_path):
        print(f"Loading {model_name} from {model_path}")
        finetune_models[model_name] = xr.open_dataset(model_path)
        
        # Load anomalies
        anom_path = os.path.join(preprocessed_dir, f"{model_name}_anomalies_weekly.nc")
        if os.path.exists(anom_path):
            if model_name == "Weekly_Means_14k_lr_0.625e-5":
                predicted_anomalies_finetuning_14k_e5 = xr.open_dataset(anom_path)
            elif model_name == "Weekly_Means_14k_lr_0.625e-7":
                predicted_anomalies_finetuning_14k_e7 = xr.open_dataset(anom_path)
            elif model_name == "Weekly_Means_18k_lr_0.625e-6":
                predicted_anomalies_finetuning_18k_e6 = xr.open_dataset(anom_path)
        
        # Load climatology
        clim_path = os.path.join(preprocessed_dir, f"{model_name}_climatology_weekly.nc")
        if os.path.exists(clim_path):
            if model_name == "Weekly_Means_14k_lr_0.625e-5":
                ds_climatology_weekly_inf_finetuning_14k_e5 = xr.open_dataset(clim_path)
            elif model_name == "Weekly_Means_14k_lr_0.625e-7":
                ds_climatology_weekly_inf_finetuning_14k_e7 = xr.open_dataset(clim_path)
            elif model_name == "Weekly_Means_18k_lr_0.625e-6":
                ds_climatology_weekly_inf_finetuning_18k_e6 = xr.open_dataset(clim_path)
    else:
        print(f"Warning: {model_name} not found in preprocessed data")

# Load thresholds
thresh_path = os.path.join(preprocessed_dir, "thresholds_95.nc")
if os.path.exists(thresh_path):
    print(f"Loading thresholds from {thresh_path}")
    ds_thresholds = xr.open_dataset(thresh_path)
else:
    print("Warning: Thresholds not found in preprocessed data")
    ds_thresholds = None

print("=" * 80)
print("PREPROCESSED DATA LOADED - READY FOR METRICS COMPUTATION")
print("=" * 80)
############################################################################################################################################################
#metrics
# Build dynamic model lists based on what was actually loaded
# Now includes reference model with 6-hourly data
models = []
preds_list = []
anomalies_preds_list = []

# Add reference model if available
if ds_inf_weekly is not None and predicted_anomalies is not None:
    models.append("reference")
    preds_list.append(ds_inf_weekly)
    anomalies_preds_list.append(predicted_anomalies)

# Add daily finetuned model if available
if ds_inf_daily_weekly is not None and predicted_anomalies_daily is not None:
    models.append("daily_finetuned")
    preds_list.append(ds_inf_daily_weekly)
    anomalies_preds_list.append(predicted_anomalies_daily)

if "Weekly_Means_14k_lr_0.625e-5" in finetune_models:
    models.append("Weekly_Means_14k_lr_0.625e-5")
    preds_list.append(finetune_models["Weekly_Means_14k_lr_0.625e-5"])
    anomalies_preds_list.append(predicted_anomalies_finetuning_14k_e5)

if "Weekly_Means_14k_lr_0.625e-7" in finetune_models:
    models.append("Weekly_Means_14k_lr_0.625e-7")
    preds_list.append(finetune_models["Weekly_Means_14k_lr_0.625e-7"])
    anomalies_preds_list.append(predicted_anomalies_finetuning_14k_e7)

if "Weekly_Means_18k_lr_0.625e-6" in finetune_models:
    models.append("Weekly_Means_18k_lr_0.625e-6")
    preds_list.append(finetune_models["Weekly_Means_18k_lr_0.625e-6"])
    anomalies_preds_list.append(predicted_anomalies_finetuning_18k_e6)

if "Weekly_Means_72k_lr_0.625e-7" in finetune_models:
    models.append("Weekly_Means_72k_lr_0.625e-7")
    preds_list.append(finetune_models["Weekly_Means_72k_lr_0.625e-7"])
    anomalies_preds_list.append(predicted_anomalies_finetuning_72k_e7)

print(f"Computing metrics for models: {models}")

# Get dimensions from observations and sample finetuned model
if preds_list:
    n_ensemble = preds_list[0].member.size
else:
    n_ensemble = 0
n_leadtime = 8  # 8 weeks with overlapping windows
n_vars = len(var_of_interest)
n_values = ds_obs_weekly[var_of_interest[0]].values.shape[-1]
n_init_dates = ds_obs_weekly.init_date.size
weeks_lead_time = np.arange(1, 9)  # Weeks 1-8

#create empty array to store the results of the metrics
acc_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs_weekly.init_date,
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

rmse_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs_weekly.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

unbiased_rmse_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs_weekly.init_date,
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

# Compute CRPS for all init dates, weeks and variables
crps_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs_weekly.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)
anomalies_crps_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model", "init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs_weekly.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

#Spread for all init dates, variables and lead times
RMS_spread_results = xr.DataArray(
    data=np.full((len(models), n_init_dates, n_vars, n_leadtime), np.nan),
    dims=("model","init_date", "variable", "leadtime"),
    coords={
        "model": models,
        "init_date": ds_obs_weekly.init_date,
        "variable": var_of_interest,
        "leadtime": np.arange(1, n_leadtime + 1)
    }
)

#acc computation
for model, pred_anomalies in zip(models, anomalies_preds_list):
    for init_date in weekly_init_dates:
        for var in var_of_interest:
            # For tp: always use original observations (tp not affected by rolling average)
            # For other variables: use from_daily for daily_finetuned model
            if var == "tp":
                obs_anom = observed_anomalies
            else:
                obs_anom = observed_anomalies_from_daily if model == "daily_finetuned" else observed_anomalies
            
            acc_results.loc[model, init_date, var, :] = xr.corr(obs_anom[var].sel(init_date=init_date), pred_anomalies[var].sel(init_date=init_date).mean(dim="member"), dim="values", weights = lat_weights).values

#R_t computation
for model, pred_anomalies in zip(models, anomalies_preds_list):
    for var in var_of_interest:
        # For tp: always use original observations (tp not affected by rolling average)
        # For other variables: use from_daily for daily_finetuned model
        if var == "tp":
            obs_anom = observed_anomalies
        else:
            obs_anom = observed_anomalies_from_daily if model == "daily_finetuned" else observed_anomalies
        
        R_t_results.loc[model, var, :, :] = xr.corr(obs_anom[var], pred_anomalies[var].mean(dim="member"), dim="init_date").values

#rmse computation with latitude weighting

for model, preds in zip(models, preds_list):
    for init_date in weekly_init_dates:
        for var in var_of_interest:
            # For tp: always use original observations (tp not affected by rolling average)
            # For other variables: use from_daily for daily_finetuned model
            if var == "tp":
                obs_data = ds_obs_weekly
            else:
                obs_data = ds_obs_weekly_from_daily if model == "daily_finetuned" else ds_obs_weekly
            
            obs = obs_data[var].sel(init_date=init_date)
            pred = preds[var].sel(init_date=init_date).mean(dim="member")
            squared_error = (obs - pred) ** 2
            weighted_mse = (squared_error * lat_weights).sum(dim="values") / lat_weights.sum()
            rmse_results.loc[model, init_date, var, :] = np.sqrt(weighted_mse).values
            #unbiased rmse
            bias = (obs - pred).mean(dim="values")
            unbiased_squared_error = ((obs - pred) - bias) ** 2
            weighted_unbiased_mse = (unbiased_squared_error * lat_weights).sum(dim="values") / lat_weights.sum()
            unbiased_rmse_results.loc[model, init_date, var, :] = np.sqrt(weighted_unbiased_mse).values

for model, preds in zip(models, preds_list):
    for t_idx in weeks_lead_time:  
        for var in var_of_interest:
            # For tp: always use original observations (tp not affected by rolling average)
            # For other variables: use from_daily for daily_finetuned model
            if var == "tp":
                obs_data = ds_obs_weekly
            else:
                obs_data = ds_obs_weekly_from_daily if model == "daily_finetuned" else ds_obs_weekly
            
            obs = obs_data[var].sel(week_lead_time=t_idx)       # (init_time, values,)
            pred = preds[var].sel(week_lead_time=t_idx)         # (init_time, member, values)
            pred_mean = pred.mean(dim="member")              # (init_time, values)
            spatial_rmse_results.loc[model, var, t_idx, :] = np.sqrt(((obs - pred_mean) ** 2).mean(dim="init_date")).values


#Compute binary dataset for probability of detection and false detection
# Note: Use standard observations for thresholds (computed from weekly finetuned models)
binary_obs = xr.where(ds_obs_weekly > ds_thresholds, 1, 0)
# For from_daily: use original ds_obs_weekly for tp (not affected by rolling average), from_daily for others
binary_obs_from_daily = {}
for var in var_of_interest:
    if var == "tp":
        # tp: use original observations (not affected by rolling average)
        binary_obs_from_daily[var] = xr.where(ds_obs_weekly[var] > ds_thresholds[var], 1, 0)
    else:
        # Other variables: use from_daily observations
        binary_obs_from_daily[var] = xr.where(ds_obs_weekly_from_daily[var] > ds_thresholds[var], 1, 0)
# Convert dict to Dataset
binary_obs_from_daily = xr.Dataset(binary_obs_from_daily)

# Build lists dynamically for models that were loaded
binary_model_list = []
prob_model_list = []

# Add reference model if available
if ds_inf_weekly is not None:
    binary_model_reference = xr.where(ds_inf_weekly > ds_thresholds, 1, 0)
    prob_model_reference = binary_model_reference.mean(dim="member")
    binary_model_list.append(binary_model_reference)
    prob_model_list.append(prob_model_reference)

# Add daily finetuned model if available
if ds_inf_daily_weekly is not None:
    binary_model_daily = xr.where(ds_inf_daily_weekly > ds_thresholds, 1, 0)
    prob_model_daily = binary_model_daily.mean(dim="member")
    binary_model_list.append(binary_model_daily)
    prob_model_list.append(prob_model_daily)

if "Weekly_Means_14k_lr_0.625e-5" in finetune_models:
    binary_model_finetuning_14k_e5 = xr.where(finetune_models["Weekly_Means_14k_lr_0.625e-5"] > ds_thresholds, 1, 0)
    prob_model_finetuning_14k_e5 = binary_model_finetuning_14k_e5.mean(dim="member")
    binary_model_list.append(binary_model_finetuning_14k_e5)
    prob_model_list.append(prob_model_finetuning_14k_e5)

if "Weekly_Means_14k_lr_0.625e-7" in finetune_models:
    binary_model_finetuning_14k_e7 = xr.where(finetune_models["Weekly_Means_14k_lr_0.625e-7"] > ds_thresholds, 1, 0)
    prob_model_finetuning_14k_e7 = binary_model_finetuning_14k_e7.mean(dim="member")
    binary_model_list.append(binary_model_finetuning_14k_e7)
    prob_model_list.append(prob_model_finetuning_14k_e7)

if "Weekly_Means_18k_lr_0.625e-6" in finetune_models:
    binary_model_finetuning_18k_e6 = xr.where(finetune_models["Weekly_Means_18k_lr_0.625e-6"] > ds_thresholds, 1, 0)
    prob_model_finetuning_18k_e6 = binary_model_finetuning_18k_e6.mean(dim="member")
    binary_model_list.append(binary_model_finetuning_18k_e6)
    prob_model_list.append(prob_model_finetuning_18k_e6)

if "Weekly_Means_72k_lr_0.625e-7" in finetune_models:
    binary_model_finetuning_72k_e7 = xr.where(finetune_models["Weekly_Means_72k_lr_0.625e-7"] > ds_thresholds, 1, 0)
    prob_model_finetuning_72k_e7 = binary_model_finetuning_72k_e7.mean(dim="member")
    binary_model_list.append(binary_model_finetuning_72k_e7)
    prob_model_list.append(prob_model_finetuning_72k_e7)

#roc computation (ROC AUC)
roc_results = {}
for prob_model_iter, model in zip(prob_model_list, models):
    # Use appropriate binary observations for each model
    binary_obs_data = binary_obs_from_daily if model == "daily_finetuned" else binary_obs
    
    roc_results[model] = {}
    for var in var_of_interest:
        roc_results[model][var] = roc(binary_obs_data[var], prob_model_iter[var], dim = "values", bin_edges='continuous', return_results = "all_as_metric_dim")

#Brier Score computation - compute dynamically for loaded models
brier_score_models = {}

# Track index for binary_model_list (models are added in order: reference, daily, then weekly finetuned)
binary_model_idx = 0

if ds_inf_weekly is not None:
    brier_score_reference_results = {}
    for var in var_of_interest:
        brier_score_reference_results[var] = brier_score(binary_obs[var], binary_model_list[binary_model_idx][var], member_dim = "member", fair=True, dim = "values", weights=lat_weights)
    brier_score_models["reference"] = brier_score_reference_results
    binary_model_idx += 1

if ds_inf_daily_weekly is not None:
    brier_score_daily_results = {}
    for var in var_of_interest:
        # Use observations from daily for daily finetuned model
        brier_score_daily_results[var] = brier_score(binary_obs_from_daily[var], binary_model_list[binary_model_idx][var], member_dim = "member", fair=True, dim = "values", weights=lat_weights)
    brier_score_models["daily_finetuned"] = brier_score_daily_results
    binary_model_idx += 1

if "Weekly_Means_14k_lr_0.625e-5" in finetune_models:
    brier_score_finetuning_14k_e5_results = {}
    for var in var_of_interest:
        brier_score_finetuning_14k_e5_results[var] = brier_score(binary_obs[var], binary_model_list[binary_model_idx][var], member_dim = "member", fair=True, dim = "values", weights=lat_weights)
    brier_score_models["Weekly_Means_14k_lr_0.625e-5"] = brier_score_finetuning_14k_e5_results
    binary_model_idx += 1

if "Weekly_Means_14k_lr_0.625e-7" in finetune_models:
    brier_score_finetuning_14k_e7_results = {}
    for var in var_of_interest:
        brier_score_finetuning_14k_e7_results[var] = brier_score(binary_obs[var], binary_model_list[binary_model_idx][var], member_dim = "member", fair=True, dim = "values", weights=lat_weights)
    brier_score_models["Weekly_Means_14k_lr_0.625e-7"] = brier_score_finetuning_14k_e7_results
    binary_model_idx += 1

if "Weekly_Means_18k_lr_0.625e-6" in finetune_models:
    brier_score_finetuning_18k_e6_results = {}
    for var in var_of_interest:
        brier_score_finetuning_18k_e6_results[var] = brier_score(binary_obs[var], binary_model_list[binary_model_idx][var], member_dim = "member", fair=True, dim = "values", weights=lat_weights)
    brier_score_models["Weekly_Means_18k_lr_0.625e-6"] = brier_score_finetuning_18k_e6_results
    binary_model_idx += 1

if "Weekly_Means_72k_lr_0.625e-7" in finetune_models:
    brier_score_finetuning_72k_e7_results = {}
    for var in var_of_interest:
        brier_score_finetuning_72k_e7_results[var] = brier_score(binary_obs[var], binary_model_list[binary_model_idx][var], member_dim = "member", fair=True, dim = "values", weights=lat_weights)
    brier_score_models["Weekly_Means_72k_lr_0.625e-7"] = brier_score_finetuning_72k_e7_results
    binary_model_idx += 1

#compute crps for the different weeks and variables
for model, preds, anomalies in zip(models, preds_list, anomalies_preds_list):
    for init_date in weekly_init_dates:
        for t_idx in weeks_lead_time:
            for var in var_of_interest:
                # For tp: always use original observations (tp not affected by rolling average)
                # For other variables: use from_daily for daily_finetuned model
                if var == "tp":
                    obs_data = ds_obs_weekly
                    obs_anom = observed_anomalies
                else:
                    obs_data = ds_obs_weekly_from_daily if model == "daily_finetuned" else ds_obs_weekly
                    obs_anom = observed_anomalies_from_daily if model == "daily_finetuned" else observed_anomalies
                
                crps_results.loc[model, init_date, var, t_idx] = crps_for_ensemble(
                    preds[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                obs_data[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                ensemble_member_dim="member",
                method="fair",
                weights=lat_weights
            ).values
                anomalies_crps_results.loc[model, init_date, var, t_idx] = crps_for_ensemble(
                    anomalies[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                obs_anom[var].sel(init_date=init_date, week_lead_time=t_idx).compute(), 
                ensemble_member_dim="member",
                method="fair",
                weights=lat_weights
            ).values

#Computation of Spread for the different weeks and variables
for model, preds in zip(models, preds_list):
    for var in var_of_interest:
        ensemble_spread = preds[var].std(dim="member")
        spread_squared = ensemble_spread ** 2
        # Spatially average the spread with latitude weighting
        weighted_var = (spread_squared * lat_weights).sum(dim="values") / lat_weights.sum()
        weighted_spread = np.sqrt(weighted_var)
        weighted_spread = weighted_spread.rename({"week_lead_time": "leadtime"})
        weighted_spread = weighted_spread.transpose("init_date", "leadtime")  # (leadtime, init_date)
        RMS_spread_results.loc[model, :, var, :] = weighted_spread.values
    
#reliability computation - compute dynamically for loaded models (finetuned only)
# Note: Reference model excluded - requires 6-hourly data
reliability_models = {}

if "Weekly_Means_14k_lr_0.625e-5" in finetune_models:
    reliablity_finetuning_14k_e5_results = {}
    for var in var_of_interest:
        reliablity_finetuning_14k_e5_results[var] = reliability(binary_obs[var], prob_model_finetuning_14k_e5[var], dim = "values", probability_bin_edges = np.linspace(0, 1, 9))
    reliability_models["Weekly_Means_14k_lr_0.625e-5"] = reliablity_finetuning_14k_e5_results

if "Weekly_Means_14k_lr_0.625e-7" in finetune_models:
    reliablity_finetuning_14k_e7_results = {}
    for var in var_of_interest:
        reliablity_finetuning_14k_e7_results[var] = reliability(binary_obs[var], prob_model_finetuning_14k_e7[var], dim = "values", probability_bin_edges = np.linspace(0, 1, 9))
    reliability_models["Weekly_Means_14k_lr_0.625e-7"] = reliablity_finetuning_14k_e7_results

if "Weekly_Means_18k_lr_0.625e-6" in finetune_models:
    reliablity_finetuning_18k_e6_results = {}
    for var in var_of_interest:
        reliablity_finetuning_18k_e6_results[var] = reliability(binary_obs[var], prob_model_finetuning_18k_e6[var], dim = "values", probability_bin_edges = np.linspace(0, 1, 9))
    reliability_models["Weekly_Means_18k_lr_0.625e-6"] = reliablity_finetuning_18k_e6_results

if "Weekly_Means_72k_lr_0.625e-7" in finetune_models:
    reliablity_finetuning_72k_e7_results = {}
    for var in var_of_interest:
        reliablity_finetuning_72k_e7_results[var] = reliability(binary_obs[var], prob_model_finetuning_72k_e7[var], dim = "values", probability_bin_edges = np.linspace(0, 1, 9))
    reliability_models["Weekly_Means_72k_lr_0.625e-7"] = reliablity_finetuning_72k_e7_results
############################################################################################################################################################
#export metrics results

output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS"
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

#R_t
R_t_results_ds = xr.Dataset({var: R_t_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
R_t_results_ds = R_t_results_ds.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)
nc_R_t_path = os.path.join(output_dir, "R_t_weekly_anomalies_AIFS.nc")
if os.path.exists(nc_R_t_path):
    os.remove(nc_R_t_path)
R_t_results_ds.to_netcdf(nc_R_t_path, mode='w')

#rmse
rmse_results_ds = xr.Dataset({var: rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_rmse_path = os.path.join(output_dir, "RMSE_weekly_AIFS.nc")
if os.path.exists(nc_rmse_path):
    os.remove(nc_rmse_path)
rmse_results_ds.to_netcdf(nc_rmse_path, mode='w')

unbiased_rmse_results_ds = xr.Dataset({var: unbiased_rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_unbiased_rmse_path = os.path.join(output_dir, "Unbiased_RMSE_weekly_AIFS.nc")
if os.path.exists(nc_unbiased_rmse_path):
    os.remove(nc_unbiased_rmse_path)
unbiased_rmse_results_ds.to_netcdf(nc_unbiased_rmse_path, mode='w')

spatial_rmse_results_ds = xr.Dataset({var: spatial_rmse_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
spatial_rmse_results_ds = spatial_rmse_results_ds.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)
nc_spatial_rmse_path = os.path.join(output_dir, "Spatial_RMSE_weekly_AIFS.nc")
if os.path.exists(nc_spatial_rmse_path):
    os.remove(nc_spatial_rmse_path)
spatial_rmse_results_ds.to_netcdf(nc_spatial_rmse_path, mode='w')

#roc - save all metrics (FPR, TPR, AUC) with probability bins (finetuned models only)
# Note: Reference model excluded - requires 6-hourly data

# Export ROC results for each loaded finetuned model
for model_name in ["Weekly_Means_14k_lr_0.625e-5", "Weekly_Means_14k_lr_0.625e-7", "Weekly_Means_18k_lr_0.625e-6", "Weekly_Means_72k_lr_0.625e-7"]:
    if model_name in roc_results:
        roc_results_model_ds = xr.Dataset(roc_results[model_name])
        nc_roc_model_path = os.path.join(output_dir, f"ROC_weekly_{model_name}_AIFS.nc")
        if os.path.exists(nc_roc_model_path):
            os.remove(nc_roc_model_path)
        roc_results_model_ds.to_netcdf(nc_roc_model_path, mode='w')

#brier score - dynamically export based on loaded models

# Build list of datasets for concatenation
brier_datasets_list = []
brier_model_names = []

for model_name in ["reference", "daily_finetuned", "Weekly_Means_14k_lr_0.625e-5", "Weekly_Means_14k_lr_0.625e-7", "Weekly_Means_18k_lr_0.625e-6", "Weekly_Means_72k_lr_0.625e-7"]:
    if model_name in brier_score_models:
        model_ds = xr.Dataset({var: brier_score_models[model_name][var] for var in var_of_interest})
        # Drop forecast_time coordinate if it exists to avoid concatenation issues
        if 'forecast_time' in model_ds.coords:
            model_ds = model_ds.drop_vars('forecast_time')
        brier_datasets_list.append(model_ds.expand_dims(model=[model_name]))
        brier_model_names.append(model_name)

# Concatenate all loaded models
if brier_datasets_list:
    brier_scores_combined_ds = xr.Dataset({
        var: xr.concat([ds[var] for ds in brier_datasets_list], dim="model")
        for var in var_of_interest
    })
    nc_brier_score_path = os.path.join(output_dir, "Brier_Score_weekly_AIFS.nc")
    if os.path.exists(nc_brier_score_path):
        os.remove(nc_brier_score_path)
    brier_scores_combined_ds.to_netcdf(nc_brier_score_path, mode='w')

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

#Spread/Skill
RMS_spread_results_ds = xr.Dataset({var: RMS_spread_results.sel(variable=var).drop_vars('variable') for var in var_of_interest})
nc_RMS_spread_path = os.path.join(output_dir, "RMS_Spread_weekly_AIFS.nc")
if os.path.exists(nc_RMS_spread_path):
    os.remove(nc_RMS_spread_path)
RMS_spread_results_ds.to_netcdf(nc_RMS_spread_path, mode='w')

#Reliability - save one file per variable to avoid conflicts (finetuned models only)
# Note: Reference model excluded - requires 6-hourly data
for var in var_of_interest:
    # Export for each loaded model
    for model_name in ["Weekly_Means_14k_lr_0.625e-5", "Weekly_Means_14k_lr_0.625e-7", "Weekly_Means_18k_lr_0.625e-6", "Weekly_Means_72k_lr_0.625e-7"]:
        if model_name in reliability_models:
            nc_reliability_path = os.path.join(output_dir, f"Reliability_{var}_weekly_{model_name}_AIFS.nc")
            # Remove existing file if it exists to avoid permission errors
            if os.path.exists(nc_reliability_path):
                os.remove(nc_reliability_path)
            reliability_models[model_name][var].to_netcdf(nc_reliability_path, mode='w')

print("=" * 80)
print("METRICS COMPUTATION COMPLETE")
print(f"All metrics results saved to: {output_dir}")
print("=" * 80)