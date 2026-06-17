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

legend_labels = {
    'reference': 'Reference',
    'weekly_finetuned': 'Weekly FT',
    'daily_finetuned': 'Daily FT'
}

# Colors for NEW models (27-0 structure)
model_colors = {
    'reference': '#ff7f0e',  # orange
    'weekly_finetuned': '#2ca02c',  # orange
    'daily_finetuned': '#d62728',  # green
}


#################################################################################################################################################################
# Load metrics data
preprocessed_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init"
ACC = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/ACC_weekly_anomalies_AIFS.nc")
CRPS = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/CRPS_weekly_AIFS.nc")
CRPS_anomalies = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/CRPS_Anomalies_weekly_AIFS.nc")
R_t = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/R_t_weekly_anomalies_AIFS.nc")
RMS_Spread = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/RMS_Spread_weekly_AIFS.nc")
RMSE = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/RMSE_weekly_AIFS.nc")
Spatial_RMSE = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/Spatial_RMSE_weekly_AIFS.nc") 
RMSE_anomalies = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/RMSE_Anomalies_weekly_AIFS.nc")

obs = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/observations_weekly_27_0.nc")
weekly_finetuned = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/weekly_finetuned_model_27_0.nc")

#switch from m to mm for precipitation

CRPS["tp"] = CRPS["tp"] * 1000
CRPS_anomalies["tp"] = CRPS_anomalies["tp"] * 1000
RMSE["tp"] = RMSE["tp"] * 1000
RMSE_anomalies["tp"] = RMSE_anomalies["tp"] * 1000
Spatial_RMSE["tp"] = Spatial_RMSE["tp"] * 1000
RMSE_anomalies["tp"] = RMSE_anomalies["tp"] * 1000
#######################################################################################################################################################################
variables = [
    ("2t", "2m Temperature"),
    ("tp", "Total Precipitation"),
    ("10u", "10m U Wind"),
    ("10v", "10m V Wind")
]

var_units = {
    "2t": "K",
    "tp": "mm",
    "10u": r"m s$^{-1}$",
    "10v": r"m s$^{-1}$"
}


pf.plot_model_comparison_subplots(
    dataset=ACC,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='Anomaly Correlation Coefficient',
    title='ACC for Weekly Forecasts: Model Comparison (Ensemble Mean)',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/acc_AIFS.png',
    ylim=(-.2, 1),
    add_zero_line=True,
    model_colors=model_colors,
    legend_labels=legend_labels,
    shared_legend=True
)

# Line plot for RMSE with model comparison
pf.plot_model_comparison_subplots(
    dataset=RMSE,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='Root Mean Square Error',
    title='RMSE for Weekly Forecasts: Model Comparison (Ensemble Mean)',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/rmse_AIFS.png',
    ylim=None,
    model_colors=model_colors,
    var_units=var_units,
    legend_labels=legend_labels,
    shared_legend=True
)


#Line plot for anomalies RMSE with model comparison
pf.plot_model_comparison_subplots(
    dataset=RMSE_anomalies,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='Root Mean Square Error of Anomalies',
    title='RMSE of Anomalies for Weekly Forecasts: Model Comparison (Ensemble Mean)',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/rmse_anomalies_AIFS.png',
    ylim=None,
    model_colors=model_colors,
    legend_labels=legend_labels,
    var_units=var_units,
    shared_legend=True
)

# Line plot for fCRPS with model comparison
pf.plot_model_comparison_subplots(
    dataset=CRPS,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='fair Continuous Ranked Probability Score',
    title='fCRPS for Weekly Forecasts: Model Comparison',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/fcrps_AIFS.png',
    ylim=None,
    model_colors=model_colors,
    legend_labels=legend_labels,
    var_units=var_units,
    shared_legend=True
)
# Line plot for Anomaly CRPS with model comparison
pf.plot_model_comparison_subplots(
    dataset=CRPS_anomalies,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='fair Continuous Ranked Probability Score',
    title='Anomaly fCRPS for Weekly Forecasts: Model Comparison',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/fcrps_anomalies_AIFS.png',
    ylim=None,
    model_colors=model_colors,
    legend_labels=legend_labels,
    var_units=var_units,
    shared_legend=True
)



pf.plot_model_comparison_subplots(
    dataset=RMS_Spread,
    variables=variables,
    var_labels={v[0]: v[1] for v in variables},
    ylabel='RMS Spread',
    title='Spread & Skill for Weekly Forecasts: Model Comparison',
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/spread_skill_AIFS.png',
    ylim=None,
    secondary_dataset=RMSE,
    secondary_ylabel='RMSE',
    secondary_linestyle='dashdot',
    model_colors=model_colors,
    var_units=var_units,
    use_dual_axes=False,  # Same scale to visually compare spread-skill ratio
    legend_labels=legend_labels,
    shared_legend=True
)

################################################################
#Spatial maps
#spatial maps of RMSE — one figure per variable, 4 weeks each, all models
spatial_vars = ['2t', 'tp', '10u', '10v']
spatial_var_labels = {"2t": "2m Temperature", "tp": "Total Precipitation",
                      "10u": "10m U Wind", "10v": "10m V Wind"}
spatial_var_units = {"2t": "K", "tp": "mm", "10u": "m/s", "10v": "m/s"}
selected_weeks = [0, 1, 2, 3]

for var in spatial_vars:
    pf.plot_single_var_spatial_rmse(
        dataset=Spatial_RMSE,
        var=var,
        weeks=selected_weeks,
        var_label=spatial_var_labels[var],
        unit=spatial_var_units[var],
        suptitle=f'Ensemble Mean RMSE — {spatial_var_labels[var]} — Models Comparison',
        savename=f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/spatial_rmse_{var}_AIFS.png',
        models=['reference', "weekly_finetuned", 'daily_finetuned']
    )

for var in spatial_vars:
    pf.plot_single_var_spatial_rmse(
        dataset=R_t,
        var=var,
        weeks=selected_weeks,
        var_label=spatial_var_labels[var],
        unit=None,
        suptitle=f'Temporal Correlation Coefficient — {spatial_var_labels[var]} — Models Comparison',
        savename=f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/R_t_{var}_AIFS.png',
        cmap = "RdBu",
        models=['reference', "weekly_finetuned", 'daily_finetuned'],
        norm= [-1,1],
        metric_cbar = r'$R_t$'
    )   

#Forecasting
ds_inf_weekly = weekly_finetuned.rename({"week_lead_time": "leadtime"})
ds_obs_weekly = obs.rename({"week_lead_time": "leadtime"})

ds_inf_weekly["tp"].values = ds_inf_weekly["tp"].values * 7


# Per-variable colormaps for better visualization
forecast_cmaps = {
    '2t': 'RdYlBu_r',  # Red-Yellow-Blue reversed for temperature
    'tp': 'YlGnBu',    # Yellow-Green-Blue for precipitation
    '10u': 'RdBu_r',   # Red-Blue diverging for wind
    '10v': 'RdBu_r'    # Red-Blue diverging for wind
}

forecast_norms = {
    '2t': Normalize(vmin=250, vmax=310),  # Temperature in Kelvin
    'tp': Normalize(vmin=0, vmax=0.1),    # Total precipitation in m
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
    subtitle="Ensemble Mean Forecast of Weekly FT Model",
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/forecast_finetuned_weekly',
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
    savename = '/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/images/era5_weekly',
    norm = forecast_norms,
    cmap = forecast_cmaps
)