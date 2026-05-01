import xarray as xr
import numpy as np
import pandas as pd
import glob

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import plotting_functions as pf
from collections import defaultdict
import importlib
importlib.reload(pf)

from matplotlib.colors import TwoSlopeNorm, Normalize





# Colors for variables
color_vars = {
    "2m Temperature": "#d62728",  # red
    "Total Precipitation": "#1f77b4",  # blue
    "10m U Wind": "#2ca02c",  # green
    "10m V Wind": "#ff7f0e",  # orange
}

# Colors for models
model_colors = {
    'reference': '#1f77b4',  # blue
    'Weekly_Means_14k_lr_0.625e-5': '#ff7f0e',  # orange
    'Weekly_Means_14k_lr_0.625e-7': '#2ca02c',  # green
    'Weekly_Means_18k_lr_0.625e-6': '#d62728'  # red
    Weekly_Means_64
}

#Load weekly mean datasets for forecasts and observations
weekly_forecasts_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Forecasts_weekly_AIFS.nc"
weekly_observations_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Observations_weekly_AIFS.nc" 
ds_inf_weekly = xr.open_dataset(weekly_forecasts_path)
ds_obs_weekly = xr.open_dataset(weekly_observations_path)

#ACC path and loading
acc_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/ACC_weekly_anomalies_AIFS.nc"
ACC_ds = xr.open_dataset(acc_path)

R_t_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/R_t_weekly_anomalies_AIFS.nc"
R_t_ds = xr.open_dataset(R_t_path)

#RMSE path and loading
rmse_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/RMSE_weekly_AIFS.nc"
RMSE_ds = xr.open_dataset(rmse_path)

unbiased_rmse_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Unbiased_RMSE_weekly_AIFS.nc"
Unbiased_RMSE_ds = xr.open_dataset(unbiased_rmse_path)

spatial_rmse_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Spatial_RMSE_weekly_AIFS.nc"
SPATIAL_RMSE_ds = xr.open_dataset(spatial_rmse_path)

#Roc path and loading
roc_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/ROC_weekly_AIFS.nc"
ROC_ds = xr.open_dataset(roc_path)

roc_path_14k_e5 = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/ROC_weekly_Weekly_Means_14k_lr_0.625e-5_AIFS.nc"
ROC_14k_e5_ds = xr.open_dataset(roc_path_14k_e5)

roc_path_14k_e7 = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/ROC_weekly_Weekly_Means_14k_lr_0.625e-7_AIFS.nc"
ROC_14k_e7_ds = xr.open_dataset(roc_path_14k_e7)

roc_path_18k_e6 = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/ROC_weekly_Weekly_Means_18k_lr_0.625e-6_AIFS.nc"
ROC_18k_e6_ds = xr.open_dataset(roc_path_18k_e6)

#Brier Score path and loading
brier_score_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Brier_Score_weekly_AIFS.nc"
brier_score_ds = xr.open_dataset(brier_score_path)
    
#CRPS path and loading
crps_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/CRPS_weekly_AIFS.nc"
CRPS_ds = xr.open_dataset(crps_path)

anomalies_crps_path = "//ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/CRPS_Anomalies_weekly_AIFS.nc"
CRPS_anomalies_ds = xr.open_dataset(anomalies_crps_path)

#SKill/Spread path and loading
spread_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/RMS_Spread_weekly_AIFS.nc"
spread_ds = xr.open_dataset(spread_path)

#Reliability  path and loading
reliability_ds_dict = {}
reliability_14k_e5_ds_dict = {}
reliability_14k_e7_ds_dict = {}
reliability_18k_e6_ds_dict = {}

for var in ["2t", "tp", "10u", "10v"]:
    reliability_path = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Reliability_{var}_weekly_AIFS.nc"
    reliability_ds = xr.open_dataset(reliability_path)
    reliability_ds_dict[var] = reliability_ds
    
    reliability_path_14k_e5 = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Reliability_{var}_weekly_Weekly_Means_14k_lr_0.625e-5_AIFS.nc"
    reliability_14k_e5_ds = xr.open_dataset(reliability_path_14k_e5)
    reliability_14k_e5_ds_dict[var] = reliability_14k_e5_ds
    
    reliability_path_14k_e7 = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Reliability_{var}_weekly_Weekly_Means_14k_lr_0.625e-7_AIFS.nc"
    reliability_14k_e7_ds = xr.open_dataset(reliability_path_14k_e7)
    reliability_14k_e7_ds_dict[var] = reliability_14k_e7_ds
    
    reliability_path_18k_e6 = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Reliability_{var}_weekly_Weekly_Means_18k_lr_0.625e-6_AIFS.nc"
    reliability_18k_e6_ds = xr.open_dataset(reliability_path_18k_e6)
    reliability_18k_e6_ds_dict[var] = reliability_18k_e6_ds



# Line plot for ACC with model comparison
variables = [
    ("2t", "2m Temperature"),
    ("tp", "Total Precipitation"),
    ("10u", "10m U Wind"),
    ("10v", "10m V Wind")
]

pf.plot_model_comparison_subplots(
    dataset=ACC_ds,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='Anomaly Correlation Coefficient',
    title='ACC for Weekly Forecasts: Model Comparison (Ensemble Mean)',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/acc_AIFS.png',
    ylim=(-.2, 1),
    add_zero_line=True,
    model_colors=model_colors
)

# Line plot for RMSE with model comparison
pf.plot_model_comparison_subplots(
    dataset=RMSE_ds,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='Root Mean Square Error',
    title='RMSE for Weekly Forecasts: Model Comparison (Ensemble Mean)',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/rmse_AIFS.png',
    ylim=None,
    model_colors=model_colors
)
#unbiased RMSE plot with model comparison
pf.plot_model_comparison_subplots(
    dataset=Unbiased_RMSE_ds,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='Unbiased Root Mean Square Error',
    title='Unbiased RMSE for Weekly Forecasts: Model Comparison (Ensemble Mean)',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/unbiased_rmse_AIFS.png',
    ylim=None,
    model_colors=model_colors
)

# Line plot for RMSE (only t2m) with model comparison
pf.plot_model_comparison_subplots(
    dataset=RMSE_ds,
    variables=[("2t", "2m Temperature")],
    var_labels={"2t": "2m Temperature"},
    ylabel='Root Mean Square Error [K]',
    title='RMSE of t2m for Weekly Forecasts: Model Comparison (Ensemble Mean)',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/rmse_2tm_AIFS.png',
    ylim=None,
    figsize=(7, 5),
    model_colors=model_colors
)


# Line plot for CRPS with model comparison
pf.plot_model_comparison_subplots(
    dataset=CRPS_ds,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='fCRPS',
    title='fCRPS for Weekly Forecasts: Model Comparison',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/crps_AIFS.png',
    ylim=None,
    model_colors=model_colors
)

pf.plot_model_comparison_subplots(
    dataset=CRPS_anomalies_ds,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='fCRPS of Anomalies',
    title='fCRPS of Anomalies for Weekly Forecasts: Model Comparison',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/crps_anomalies_AIFS.png',
    ylim=None,
    model_colors=model_colors
)


# Line plot for Spread/Skill with model comparison (RMSE overlay on same axis for ratio comparison)
var_units_dict = {"2t": "K", "tp": "m", "10u": "m/s", "10v": "m/s"}

pf.plot_model_comparison_subplots(
    dataset=spread_ds,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='RMS Spread',
    title='Spread & Skill for Weekly Forecasts: Model Comparison',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/spread_skill_AIFS.png',
    ylim=None,
    secondary_dataset=RMSE_ds,
    secondary_ylabel='RMSE',
    secondary_linestyle='--',
    model_colors=model_colors,
    var_units=var_units_dict,
    use_dual_axes=False  # Same scale to visually compare spread-skill ratio
)


# Line plot for Brier Score with model comparison
pf.plot_model_comparison_subplots(
    dataset=brier_score_ds,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='Brier Score',
    title='Brier Score for Weekly Forecasts: Model Comparison',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/brier_score_AIFS.png',
    ylim=None,
    model_colors=model_colors
)


#spatial maps of RMSE — one figure per variable, 4 weeks each, all models
spatial_vars = ['2t', 'tp', '10u', '10v']
spatial_var_labels = {"2t": "2m Temperature", "tp": "Total Precipitation",
                      "10u": "10m U Wind", "10v": "10m V Wind"}
spatial_var_units = {"2t": "K", "tp": "m", "10u": "m/s", "10v": "m/s"}
selected_weeks = [0, 2, 4, 6]

for var in spatial_vars:
    pf.plot_single_var_spatial_rmse(
        dataset=SPATIAL_RMSE_ds,
        var=var,
        weeks=selected_weeks,
        var_label=spatial_var_labels[var],
        unit=spatial_var_units[var],
        suptitle=f'Ensemble Mean RMSE — {spatial_var_labels[var]} — All Models',
        savename=f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/spatial_rmse_{var}_AIFS.png',
        models=['reference', 'Weekly_Means_14k_lr_0.625e-5', 'Weekly_Means_14k_lr_0.625e-7', 'Weekly_Means_18k_lr_0.625e-6']
    )
    
    

#Spatial map of R_t for each model
for model in ['reference', 'Weekly_Means_14k_lr_0.625e-5', 'Weekly_Means_14k_lr_0.625e-7', 'Weekly_Means_18k_lr_0.625e-6']:
    model_display = model.replace('_', ' ').replace('lr ', 'lr=')
    pf.plot_weekly_spatial_maps(
        R_t_ds.sel(model=model), 
        ['2t', 'tp', '10u', '10v'], 
        [0, 2, 4, 6], 
        "Temporal CC",
        f'Temporal Correlation Coefficient of {model_display}',
        f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/R_t_{model}_AIFS.png'
    )

#Forecasting
ds_inf_weekly = ds_inf_weekly.rename({"week_lead_time": "leadtime"})
ds_obs_weekly = ds_obs_weekly.rename({"week_lead_time": "leadtime"})


# Per-variable colormaps for better visualization
forecast_cmaps = {
    '2t': 'RdYlBu_r',  # Red-Yellow-Blue reversed for temperature
    'tp': 'YlGnBu',    # Yellow-Green-Blue for precipitation
    '10u': 'RdBu_r',   # Red-Blue diverging for wind
    '10v': 'RdBu_r'    # Red-Blue diverging for wind
}

forecast_norms = {
    '2t': Normalize(vmin=250, vmax=310),  # Temperature in Kelvin
    'tp': Normalize(vmin=0, vmax=0.02),    # Total precipitation in m
    '10u': Normalize(vmin=-15, vmax=15),   # U wind in m/s
    '10v': Normalize(vmin=-15, vmax=15)    # V wind in m/s
}

# Per-variable labels with units
forecast_labels = {
    '2t': '2m Temperature [K]',
    'tp': 'Total Precipitation [m]',
    '10u': '10m U Wind [m/s]',
    '10v': '10m V Wind [m/s]'
}

pf.plot_weekly_spatial_maps(
    ds_inf_weekly.isel(init_date=0).mean(dim="member"), 
    ['2t', 'tp', '10u', '10v'], 
    [1, 3, 5, 7], 
    label=forecast_labels,
    subtitle="Ensemble Mean Forecast of Reference Model",
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/forecast_weekly_AIFS',
    norm=forecast_norms,
    cmap=forecast_cmaps
)

#Era-5
pf.plot_weekly_spatial_maps(
    ds_obs_weekly.isel(init_date = 0),
    ['2t', 'tp', '10u', '10v'],
    [1, 3, 5, 7],
    label=forecast_labels,
    subtitle="ERA5 Ground Truth",
    savename = '/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/era5_weekly_AIFS',
    norm = forecast_norms,
    cmap = forecast_cmaps
)


# ROC curves — one variable per row, weeks across columns, models overlaid
var_short = ["2t", "tp", "10u", "10v"]
var_labels = {"2t": "2m Temperature", "tp": "Total Precipitation",
              "10u": "10m U Wind", "10v": "10m V Wind"}

# Model linestyles for differentiation
model_linestyles = {
    'reference': '-',
    'Weekly_Means_14k_lr_0.625e-5': '--',
    'Weekly_Means_14k_lr_0.625e-7': ':',
    'Weekly_Means_18k_lr_0.625e-6': '-.'
}

model_roc_datasets = {
    'reference': ROC_ds,
    'Weekly_Means_14k_lr_0.625e-5': ROC_14k_e5_ds,
    'Weekly_Means_14k_lr_0.625e-7': ROC_14k_e7_ds,
    'Weekly_Means_18k_lr_0.625e-6': ROC_18k_e6_ds
}

# Create one figure per variable with 8 weeks as subplots
for var in var_short:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for week_idx in range(8):
        ax = axes[week_idx]
        
        # Plot each model
        for model_name, roc_ds in model_roc_datasets.items():
            # Average over init_date and select the week
            roc_week_var = roc_ds[var].sel(week_lead_time=week_idx+1).mean(dim='init_date')
            
            # Extract FPR, TPR, and AUC from the metric dimension
            fpr = roc_week_var.sel(metric='false positive rate')
            tpr = roc_week_var.sel(metric='true positive rate')
            auc_val = roc_week_var.sel(metric='area under curve').mean().item()
            
            model_display = model_name.replace('_', ' ').replace('lr ', 'lr=')
            ax.plot(fpr, tpr,
                    color=model_colors[model_name],
                    linestyle=model_linestyles[model_name],
                    linewidth=2,
                    label=f"{model_display} (AUC={auc_val:.2f})")
        
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(f"Week {week_idx+1}", fontsize=10)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"ROC Curves: {var_labels[var]} — Model Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/roc_{var}_AIFS.png', dpi=150)
    plt.close()


# Reliability diagrams — one variable per row, weeks across columns, models overlaid
model_reliability_datasets = {
    'reference': reliability_ds_dict,
    'Weekly_Means_14k_lr_0.625e-5': reliability_14k_e5_ds_dict,
    'Weekly_Means_14k_lr_0.625e-7': reliability_14k_e7_ds_dict,
    'Weekly_Means_18k_lr_0.625e-6': reliability_18k_e6_ds_dict
}

model_markers = {
    'reference': 'o',
    'Weekly_Means_14k_lr_0.625e-5': 's',
    'Weekly_Means_14k_lr_0.625e-7': '^',
    'Weekly_Means_18k_lr_0.625e-6': 'D'
}

# Create one figure per variable with 8 weeks as subplots
for var in var_short:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for week_idx in range(8):
        ax = axes[week_idx]
        
        # Plot each model
        for model_name, rel_ds_dict in model_reliability_datasets.items():
            # Average over init_date and select the week
            rel_week_var = rel_ds_dict[var][var].sel(week_lead_time=week_idx+1).mean(dim='init_date')
            
            # Get forecast probability (x-axis) and observed frequency (y-axis)
            forecast_prob = rel_week_var['forecast_probability'].values
            observed_freq = rel_week_var.values
            
            model_display = model_name.replace('_', ' ').replace('lr ', 'lr=')
            ax.plot(forecast_prob, observed_freq,
                    marker=model_markers[model_name],
                    color=model_colors[model_name],
                    linewidth=2,
                    markersize=6,
                    label=model_display)
        
        # Add perfect calibration line (diagonal)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect calibration")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Forecast Probability", fontsize=9)
        ax.set_ylabel("Observed Frequency", fontsize=9)
        ax.set_title(f"Week {week_idx+1}", fontsize=10)
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Reliability Diagrams: {var_labels[var]} — Model Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/reliability_{var}_AIFS.png', dpi=150)
    plt.close()


# Ensemble plume plots for a specific location and initialization date
# De Bilt, Netherlands: approximately 52.1°N, 5.18°E
target_lat = 52.1
target_lon = 5.18

# Find closest grid point to De Bilt
lat_diff = np.abs(ds_inf_weekly.latitude - target_lat)
lon_diff = np.abs(ds_inf_weekly.longitude - target_lon)
distance = np.sqrt(lat_diff**2 + lon_diff**2)
closest_idx = distance.argmin().values.item()

print(f"Target location: {target_lat}°N, {target_lon}°E (De Bilt, NL)")
print(f"Closest grid point: {ds_inf_weekly.latitude.values[closest_idx]:.2f}°N, {ds_inf_weekly.longitude.values[closest_idx]:.2f}°E")

# Load finetuning model weekly forecasts
# Load finetuning model data
finetune_models_list = ["Weekly_Means_14k_lr_0.625e-5", "Weekly_Means_14k_lr_0.625e-7", "Weekly_Means_18k_lr_0.625e-6"]
finetune_weekly_data = {}

for finetune_model in finetune_models_list:
    paths_init_finetune = sorted(glob.glob(
        f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/{finetune_model}/"
        "aifs-subs-pretrain-weeklymeans-*"
    ))

    reforecast_datasets_finetuning = []
    init_dates_finetuning = []

    for path_init in paths_init_finetune:
        paths_member = sorted(glob.glob(f"{path_init}/*.nc"))
        
        ds_init = xr.open_mfdataset(
            paths_member,
            combine="nested",
            concat_dim="member",
            parallel=False,
            chunks={}
        )
        
        init_date = ds_init.time.values[0].astype('datetime64[ns]')
        init_dates_finetuning.append(init_date)
        
        ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
        lead_times = (ds_init.time.values - init_date) / np.timedelta64(7, 'D')
        ds_init = ds_init.assign_coords(time=lead_times).rename({'time': 'week_lead_time'})
        ds_init = ds_init.isel(week_lead_time=slice(1, None))
        
        reforecast_datasets_finetuning.append(ds_init)

    ds_inf_weekly_finetuning = xr.concat(
        reforecast_datasets_finetuning,
        dim=xr.DataArray(init_dates_finetuning, dims='init_date', name='init_date')
    )
    
    ds_inf_weekly_finetuning = ds_inf_weekly_finetuning[var_short]
    finetune_weekly_data[finetune_model] = ds_inf_weekly_finetuning

# Select first initialization date (July 1st)
first_init_date = ds_inf_weekly.init_date.values[0]
print(f"Selected initialization date: {first_init_date}")

# Variable units for y-axis labels
var_units = {"2t": "K", "tp": "m", "10u": "m/s", "10v": "m/s"}

# Create plume plots for each variable
for var in var_short:
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # Get data for all models to calculate common y-axis limits
    ref_data = ds_inf_weekly[var].sel(init_date=first_init_date).isel(values=closest_idx)
    weeks = ref_data.leadtime.values
    obs_data = ds_obs_weekly[var].sel(init_date=first_init_date).isel(values=closest_idx)
    ft_data_1 = finetune_weekly_data["Weekly_Means_14k_lr_0.625e-5"][var].sel(init_date=first_init_date).isel(values=closest_idx)
    ft_data_2 = finetune_weekly_data["Weekly_Means_14k_lr_0.625e-7"][var].sel(init_date=first_init_date).isel(values=closest_idx)
    ft_data_3 = finetune_weekly_data["Weekly_Means_18k_lr_0.625e-6"][var].sel(init_date=first_init_date).isel(values=closest_idx)
    
    # Calculate global min and max across all models and observations
    all_values = np.concatenate([
        ref_data.values.flatten(),
        obs_data.values.flatten(),
        ft_data_1.values.flatten(),
        ft_data_2.values.flatten(),
        ft_data_3.values.flatten()
    ])
    y_min = np.nanmin(all_values)
    y_max = np.nanmax(all_values)
    y_margin = (y_max - y_min) * 0.05  # 5% margin
    y_limits = [y_min - y_margin, y_max + y_margin]
    
    # Reference model
    ax = axes[0]
    
    # Plot each ensemble member
    for member in range(ref_data.member.size):
        ax.plot(weeks, ref_data.isel(member=member).values, 
                color=model_colors['reference'], alpha=0.3, linewidth=1)
    
    # Plot ensemble mean
    ax.plot(weeks, ref_data.mean(dim='member').values, 
            color=model_colors['reference'], linewidth=3, label='Ensemble Mean')
    
    # Plot observation
    ax.plot(weeks, obs_data.values, 'k-', linewidth=2, marker='o', label='Observation')
    
    ax.set_xlabel('Week Lead Time', fontsize=11)
    ax.set_ylabel(f'{var_labels[var]} [{var_units[var]}]', fontsize=11)
    ax.set_title('Reference Model', fontsize=12)
    ax.set_ylim(y_limits)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Finetuning model 14k lr=0.625e-5
    ax = axes[1]
    
    for member in range(ft_data_1.member.size):
        ax.plot(weeks, ft_data_1.isel(member=member).values, 
                color=model_colors['Weekly_Means_14k_lr_0.625e-5'], alpha=0.3, linewidth=1)
    
    ax.plot(weeks, ft_data_1.mean(dim='member').values, 
            color=model_colors['Weekly_Means_14k_lr_0.625e-5'], linewidth=3, label='Ensemble Mean')
    
    ax.plot(weeks, obs_data.values, 'k-', linewidth=2, marker='o', label='Observation')
    
    ax.set_xlabel('Week Lead Time', fontsize=11)
    ax.set_ylabel(f'{var_labels[var]} [{var_units[var]}]', fontsize=11)
    ax.set_title('14k steps, lr=0.625e-5', fontsize=12)
    ax.set_ylim(y_limits)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Finetuning model 14k lr=0.625e-7
    ax = axes[2]
    
    for member in range(ft_data_2.member.size):
        ax.plot(weeks, ft_data_2.isel(member=member).values, 
                color=model_colors['Weekly_Means_14k_lr_0.625e-7'], alpha=0.3, linewidth=1)
    
    ax.plot(weeks, ft_data_2.mean(dim='member').values, 
            color=model_colors['Weekly_Means_14k_lr_0.625e-7'], linewidth=3, label='Ensemble Mean')
    
    ax.plot(weeks, obs_data.values, 'k-', linewidth=2, marker='o', label='Observation')
    
    ax.set_xlabel('Week Lead Time', fontsize=11)
    ax.set_ylabel(f'{var_labels[var]} [{var_units[var]}]', fontsize=11)
    ax.set_title('14k steps, lr=0.625e-7', fontsize=12)
    ax.set_ylim(y_limits)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Finetuning model 18k lr=0.625e-6
    ax = axes[3]
    
    for member in range(ft_data_3.member.size):
        ax.plot(weeks, ft_data_3.isel(member=member).values, 
                color=model_colors['Weekly_Means_18k_lr_0.625e-6'], alpha=0.3, linewidth=1)
    
    ax.plot(weeks, ft_data_3.mean(dim='member').values, 
            color=model_colors['Weekly_Means_18k_lr_0.625e-6'], linewidth=3, label='Ensemble Mean')
    
    ax.plot(weeks, obs_data.values, 'k-', linewidth=2, marker='o', label='Observation')
    
    ax.set_xlabel('Week Lead Time', fontsize=11)
    ax.set_ylabel(f'{var_labels[var]} [{var_units[var]}]', fontsize=11)
    ax.set_title('18k steps, lr=0.625e-6', fontsize=12)
    ax.set_ylim(y_limits)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Ensemble Plumes: {var_labels[var]} at De Bilt, NL\n'
                 f'Init: {pd.to_datetime(first_init_date).strftime("%Y-%m-%d")}', 
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/plume_{var}_AIFS.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plume plot saved for {var}")


