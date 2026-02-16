import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import plotting_functions as pf
from utils.physics_functions import compute_r_sur, compute_rh_sur
from utils import metrics_function as mf
from collections import defaultdict
import importlib
importlib.reload(pf)
importlib.reload(mf)
from matplotlib.colors import TwoSlopeNorm, Normalize


#dataset paths
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
ds_dataset = xr.open_zarr(dataset_path)


#ACC path an loading
ACC_spatial_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/ACC_weekly_anomalies_spatial.nc"
ACC_rollout_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/ACC_weekly_anomalies_10k_rollout.nc"
ACC_ds = xr.open_dataset(ACC_spatial_path)
ACC_ds_rollout = xr.open_dataset(ACC_rollout_path)


boxplot_acc_2t = {
    f"week {i+1}": ACC_ds["2t"].isel(leadtime=i).values
    for i in range(len(ACC_ds["2t"].leadtime)-2)
}

boxplot_acc_tp = {
    f"week {i+1}": ACC_ds["tp"].isel(leadtime=i).values
    for i in range(len(ACC_ds["tp"].leadtime)-2)
}

boxplot_acc_10u = {
    f"week {i+1}": ACC_ds["10u"].isel(leadtime=i).values
    for i in range(len(ACC_ds["10u"].leadtime)-2)
}

boxplot_acc_10v = {
    f"week {i+1}": ACC_ds["10v"].isel(leadtime=i).values
    for i in range(len(ACC_ds["10v"].leadtime)-2)
}

color_weeks = {
    "week 1": "#1f77b4",  # blue
    "week 2": "#ff7f0e",  # orange
    "week 3": "#2ca02c",  # green
    "week 4": "#d62728",  # red
    "week 5": "#9467bd",  # purple
    "week 6": "#8c564b",  # brown
    "week 7": "#e377c2",  # pink
    "week 8": "#7f7f7f",  # gray
    "week 9": "#bcbd22",  # yellow-green
} 

# Restructure data for plot_boxplots: {subplot_name: {box_label: array}}
boxplot_data = {"2m Temperature": boxplot_acc_2t,
                "Total Precipitation": boxplot_acc_tp,
                "10m U Wind": boxplot_acc_10u,
                "10m V Wind": boxplot_acc_10v
                }

pf.plot_boxplots(
    data_dict=boxplot_data,
    title='ACC for Weekly Forecasts of reference Model over 12 Runs',
    colors=color_weeks,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/acc_regional_boxplot.png',
    ylabel='Anomaly Correlation Coefficient',
    sharey = True
)

#Same for rollout model
boxplot_acc_2t_rollout = {
    f"week {i+1}": ACC_ds_rollout["2t"].isel(leadtime=i).values
    for i in range(len(ACC_ds_rollout["2t"].leadtime)-2)
}
boxplot_acc_tp_rollout = {
    f"week {i+1}": ACC_ds_rollout["tp"].isel(leadtime=i).values
    for i in range(len(ACC_ds_rollout["tp"].leadtime)-2)
}
boxplot_acc_10u_rollout = {
    f"week {i+1}": ACC_ds_rollout["10u"].isel(leadtime=i).values
    for i in range(len(ACC_ds_rollout["10u"].leadtime)-2)
}
boxplot_acc_10v_rollout = {
    f"week {i+1}": ACC_ds_rollout["10v"].isel(leadtime=i).values
    for i in range(len(ACC_ds_rollout["10v"].leadtime)-2)
}

# Restructure data for plot_boxplots: {subplot_name: {box_label: array}}
boxplot_data_rollout = {"2m Temperature": boxplot_acc_2t_rollout,
                "Total Precipitation": boxplot_acc_tp_rollout,
                "10m U Wind": boxplot_acc_10u_rollout,
                "10m V Wind": boxplot_acc_10v_rollout
                }


pf.plot_boxplots(
    data_dict=boxplot_data_rollout,
    title='ACC for Weekly Forecasts of Rollout Model over 12 Runs',
    colors=color_weeks,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/acc_rollout_boxplot.png',
    ylabel='Anomaly Correlation Coefficient',
    sharey = True
)

#Comparison between the two models
# Structure: {week: {model: array}} - one subplot per week with two boxes per subplot
var_list = ['2t', 'tp', '10u', '10v']

# Create comparison dictionaries automatically
comparison_dicts = {
    var: {
        f"week {i+1}": {
            "Reference": ACC_ds[var].isel(leadtime=i).values,
            "Rollout": ACC_ds_rollout[var].isel(leadtime=i).values
        }
        for i in range(6)
    }
    for var in var_list
}

# Unpack for easier access
boxplot_acc_2t_comparison = comparison_dicts['2t']
boxplot_acc_tp_comparison = comparison_dicts['tp']
boxplot_acc_10u_comparison = comparison_dicts['10u']
boxplot_acc_10v_comparison = comparison_dicts['10v']

boxplot_comparison_list = [
    boxplot_acc_2t_comparison,
    boxplot_acc_tp_comparison,
    boxplot_acc_10u_comparison,
    boxplot_acc_10v_comparison
]

# Colors for the two models
color_models = {"Reference": "#1f77b4", "Rollout": "#ff7f0e"}

for boxplot_comparison, var in zip(boxplot_comparison_list, var_list):
    pf.plot_boxplots(
        data_dict=boxplot_comparison,
        title=f'ACC comparison for Weekly Forecasts of {var} over 12 Runs',
        colors=color_models,
        savename=f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/acc_comparison_{var}_boxplot.png',
        ylabel='Anomaly Correlation Coefficient',
        sharey = True
    )

# Load R_t data
Rt_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/Rt_weekly_anomalies.nc"
Rt_ds = xr.open_dataset(Rt_path)


Rt_rollout_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/Rt_weekly_anomalies_10k_rollout.nc"
Rt_ds_rollout = xr.open_dataset(Rt_rollout_path)

Rt_list = [Rt_ds, Rt_ds_rollout, Rt_package_ds]
Rt_labels = ['Reference Model', 'Rollout Model', 'Reference Model Package']

pf.plot_weekly_spatial_maps(Rt_ds, ['2t', 'tp', '10u', '10v'], [0, 2, 4, 6], "Tempoaral CC",
                            'Temporal Correlation Coefficient of Reference Model across 12 Runs',
                            '/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/test')



############################################################################################################
#Same with the outputs of the package
#ACC outputs
acc_package_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/ACC_weekly_anomalies_refmodel_package.nc"
acc_package_ds = xr.open_dataset(acc_package_path)
#R_t outputs
Rt_package_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/Rt_weekly_anomalies_refmodel_package.nc"
Rt_package_ds = xr.open_dataset(Rt_package_path)

#SEDI outputs
SEDI_weekly_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/SEDI_weekly.nc"
SEDI_weekly_ds = xr.open_dataset(SEDI_weekly_path)

SEDI_spatial_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/SEDI_spatial_weekly.nc"
SEDI_spatial_ds = xr.open_dataset(SEDI_spatial_path)

#RMSE outputs
RMSE_weekly_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/RMSE_weekly.nc"
RMSE_weekly_ds = xr.open_dataset(RMSE_weekly_path)

RMSE_spatial_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/RMSE_spatial_weekly.nc"
RMSE_spatial_ds = xr.open_dataset(RMSE_spatial_path)

#boxplot for package outputs
#Acc
boxplot_acc_package = {
    "2m Temperature": {
        f"week {i+1}": acc_package_ds["2t"].isel(leadtime=i) for i in range(8)},
    "Total Precipitation": {
        f"week {i+1}": acc_package_ds["tp"].isel(leadtime=i) for i in range(8)},
    "10m U Wind": {
        f"week {i+1}": acc_package_ds["10u"].isel(leadtime=i) for i in range(8)},
    "10m V Wind": {
        f"week {i+1}": acc_package_ds["10v"].isel(leadtime=i) for i in range(8)}
}
#RMSE

boxplot_RMSE = {
    "2m Temperature": {
        f"week {i+1}": RMSE_weekly_ds["2t"].isel(leadtime=i) for i in range(8)},
    "Total Precipitation": {
        f"week {i+1}": RMSE_weekly_ds["tp"].isel(leadtime=i) for i in range(8)},
    "10m U Wind": {
        f"week {i+1}": RMSE_weekly_ds["10u"].isel(leadtime=i) for i in range(8)},
    "10m V Wind": {
        f"week {i+1}": RMSE_weekly_ds["10v"].isel(leadtime=i) for i in range(8)}
}

for var in var_list:
    SEDI_weekly_ds[var].values = SEDI_weekly_ds[var].values * -1

#SEDI
boxplot_SEDI ={
    "2m Temperature": {
        f"week {i+1}": SEDI_weekly_ds["2t"].isel(leadtime=i) for i in range(8)},
    "Total Precipitation": {
        f"week {i+1}": SEDI_weekly_ds["tp"].isel(leadtime=i) for i in range(8)},
    "10m U Wind": {
        f"week {i+1}": SEDI_weekly_ds["10u"].isel(leadtime=i) for i in range(8)},
    "10m V Wind": {
        f"week {i+1}": SEDI_weekly_ds["10v"].isel(leadtime=i) for i in range(8)}
} 

#Boxplots
#ACC
pf.plot_boxplots(
    data_dict=boxplot_acc_package,
    title='ACC for Weekly Forecasts of reference Model over 12 Runs, computed with package',
    colors=color_weeks,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/acc_regional_boxplot_package.png',
    ylabel='Anomaly Correlation Coefficient',
    sharey = True
)

#RMSE
pf.plot_boxplots(
    data_dict=boxplot_RMSE,
    title="RMSE for Weekly Forecasts of reference Model over 12 Runs",
    colors=color_weeks,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/rmse_weekly.png',
    ylabel='Root Mean Square Error',
    sharey = False
)

#SEDI
pf.plot_boxplots(
    data_dict=boxplot_SEDI,
    title="SEDI for Weekly Forecasts of reference Model over 12 Runs",
    colors=color_weeks,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/sedi_weekly.png',
    ylabel='SEDI',
    sharey = True
)




SEDI_spatial_ds = SEDI_spatial_ds.assign_coords(
    latitude=("values", ds_dataset.latitudes.values),
    longitude=("values", ds_dataset.longitudes.values),)

    
pf.plot_weekly_spatial_maps(SEDI_spatial_ds , ['2t', 'tp', '10u', '10v'], [0, 2, 4, 6], "SEDI", "Spatial SEDI for Reference Model across 12 Runs", '/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/spatial_sedi_refmodel')

