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
ACC_ds = xr.open_dataset(ACC_spatial_path)


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

# Load R_t data
Rt_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/Rt_weekly_anomalies.nc"
Rt_ds = xr.open_dataset(Rt_path)

# Plot R_t maps for weeks 3-6
fig = plt.figure(figsize=(20, 16))

variables = ['2t', 'tp', '10u', '10v']
var_names_full = ['2m Temperature', 'Total Precipitation', '10m U Wind', '10m V Wind']
weeks = [0 , 2, 4, 6]  # Weeks 1, 3, 5, 7 (0-indexed)

proj = ccrs.PlateCarree()
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

for i, (var, var_name) in enumerate(zip(variables, var_names_full)):
    for j, week in enumerate(weeks):
        ax = fig.add_subplot(4, 4, i*4 + j + 1, projection=proj)
        ax.set_global()
        
        # Get R_t data for this variable and week
        rt_data = Rt_ds[var].isel(leadtime=week).values.ravel()
        lons = Rt_ds['longitude'].values.ravel()
        lats = Rt_ds['latitude'].values.ravel()
        
        # Handle antimeridian
        lons = np.where(lons > 180, lons - 360, lons)
        
        # Remove NaNs/infs
        mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(rt_data)
        lons_plot, lats_plot, rt_plot = lons[mask], lats[mask], rt_data[mask]
        
        # Plot using tricontourf
        im = ax.tricontourf(lons_plot, lats_plot, rt_plot, 40, 
                           transform=proj, norm=norm, cmap='RdBu_r')
        
        # Add coastlines
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        # Set title
        ax.set_title(f'{var_name}\nWeek {week+1}', fontsize=10, fontweight='bold')

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('R_t (Temporal Correlation)', fontsize=12)

plt.suptitle('Temporal Correlation Coefficient (R_t) across 12 Runs', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 0.91, 0.97])

plt.savefig('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_rollout_training/metrics/images/rt_spatial_maps.png', 
            dpi=300, bbox_inches='tight')
print("R_t spatial maps saved!")



