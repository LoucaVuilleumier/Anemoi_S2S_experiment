import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm, Normalize
import os
import sys
import glob
import warnings
import json
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
WEEKLY_MEANS_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init"
HW_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/heat_wave_analysis_final"
HW_DIR_SCRATCH ="/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_metrics"
OBS_WEEKLY_FILE = os.path.join(WEEKLY_MEANS_DIR, "observations_weekly_27_0.nc")
REF_WEEKLY_FILE = os.path.join(WEEKLY_MEANS_DIR, "reference_model_weekly_27_0.nc")
WEEKLY_FINETUNED_FILE = os.path.join(WEEKLY_MEANS_DIR, "weekly_finetuned_model_27_0.nc")
DAILY_FINETUNED_FILE = os.path.join(WEEKLY_MEANS_DIR, "daily_finetuned_model_weekly_3_0.nc")

OUTPUT_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/heat_wave_final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variables for comprehensive physics analysis
SURFACE_VARS = ['2t', '2d', 'msl', 'sp', 'skt', 'tp', '10u', '10v']
UPPER_VARS = ['t_500', 't_700', 't_850', 't_925', 'q_500', 'q_700', 'q_850', 'q_925',"z_500", "z_700", "z_850", "z_925"]
ALL_VARS = SURFACE_VARS + UPPER_VARS

list_modes = ["Reference", "Weekly Finetuned", "Daily Finetuned"]

# Load datasets
obs_weekly_ds = xr.open_dataset(OBS_WEEKLY_FILE)
ref_weekly_ds = xr.open_dataset(REF_WEEKLY_FILE)
weekly_finetuned_ds = xr.open_dataset(WEEKLY_FINETUNED_FILE)
daily_finetuned_ds = xr.open_dataset(DAILY_FINETUNED_FILE)

#Load metrics
ds_hw_metrics_spatial_ref = xr.open_dataset(os.path.join(HW_DIR,"heat_wave_metrics_spatial_reference.nc"))
ds_hw_metrics_spatial_weekly_finetuned = xr.open_dataset(os.path.join(HW_DIR,"heat_wave_metrics_spatial_weekly_finetuned.nc"))
ds_hw_metrics_spatial_daily_finetuned = xr.open_dataset(os.path.join(HW_DIR,"heat_wave_metrics_spatial_daily_finetuned.nc"))
list_ds_hw_metrics_spatial = [ds_hw_metrics_spatial_ref, ds_hw_metrics_spatial_weekly_finetuned, ds_hw_metrics_spatial_daily_finetuned]

ds_hw_metrics_aggregated_ref = xr.open_dataset(os.path.join(HW_DIR,"heat_wave_metrics_aggregated_reference.nc"))
ds_hw_metrics_aggregated_weekly_finetuned = xr.open_dataset(os.path.join(HW_DIR,"heat_wave_metrics_aggregated_weekly_finetuned.nc"))
ds_hw_metrics_aggregated_daily_finetuned = xr.open_dataset(os.path.join(HW_DIR,"heat_wave_metrics_aggregated_daily_finetuned.nc"))
list_ds_hw_metrics_aggregated = [ds_hw_metrics_aggregated_ref, ds_hw_metrics_aggregated_weekly_finetuned, ds_hw_metrics_aggregated_daily_finetuned]

#Metrcis with new thresholds
ds_hw_metrics_spatial_ref_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH,"heat_wave_metrics_spatial_reference_new.nc")).squeeze("variable")
ds_hw_metrics_spatial_weekly_finetuned_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH,"heat_wave_metrics_spatial_weekly_finetuned_new.nc")).squeeze("variable")
ds_hw_metrics_spatial_daily_finetuned_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH,"heat_wave_metrics_spatial_daily_finetuned_new.nc")).squeeze("variable")
list_ds_hw_metrics_spatial_new = [ds_hw_metrics_spatial_ref_new, ds_hw_metrics_spatial_weekly_finetuned_new, ds_hw_metrics_spatial_daily_finetuned_new]

ds_hw_metrics_aggregated_ref_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH,"heat_wave_metrics_aggregated_reference_new.nc"))
ds_hw_metrics_aggregated_weekly_finetuned_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH,"heat_wave_metrics_aggregated_weekly_finetuned_new.nc"))
ds_hw_metrics_aggregated_daily_finetuned_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH,"heat_wave_metrics_aggregated_daily_finetuned_new.nc"))
list_ds_hw_metrics_aggregated_new = [ds_hw_metrics_aggregated_ref_new, ds_hw_metrics_aggregated_weekly_finetuned_new, ds_hw_metrics_aggregated_daily_finetuned_new]

#load ROC data
ds_hw_roc_ref = xr.open_dataset(os.path.join(HW_DIR, "heat_wave_roc_reference.nc"))["roc"]
ds_hw_roc_weekly_finetuned = xr.open_dataset(os.path.join(HW_DIR, "heat_wave_roc_weekly_finetuned.nc"))["roc"]
ds_hw_roc_daily_finetuned = xr.open_dataset(os.path.join(HW_DIR, "heat_wave_roc_daily_finetuned.nc"))["roc"]

#load ROC data with new thresholds
ds_hw_roc_ref_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH, "heat_wave_roc_reference_new.nc"))["roc"].squeeze("variable")
ds_hw_roc_weekly_finetuned_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH, "heat_wave_roc_weekly_finetuned_new.nc"))["roc"].squeeze("variable")
ds_hw_roc_daily_finetuned_new = xr.open_dataset(os.path.join(HW_DIR_SCRATCH, "heat_wave_roc_daily_finetuned_new.nc"))["roc"].squeeze("variable")

# Load reliability diagram data
with open(os.path.join(HW_DIR, "heat_wave_reliability_diagrams_reference.json"), 'r') as f:
    reliability_data_ref = json.load(f)

with open(os.path.join(HW_DIR, "heat_wave_reliability_diagrams_weekly_finetuned.json"), 'r') as f:
    reliability_data_weekly_finetuned = json.load(f)

with open(os.path.join(HW_DIR, "heat_wave_reliability_diagrams_daily_finetuned.json"), 'r') as f:
    reliability_data_daily_finetuned = json.load(f)
    
list_reliability_data = [reliability_data_ref, reliability_data_weekly_finetuned, reliability_data_daily_finetuned]

#load reliability diagram data with new thresholds
with open(os.path.join(HW_DIR_SCRATCH, "heat_wave_reliability_diagrams_reference_new.json"), 'r') as f: 
    reliability_data_ref_new = json.load(f)
    
with open(os.path.join(HW_DIR_SCRATCH, "heat_wave_reliability_diagrams_weekly_finetuned_new.json"), 'r') as f:
    reliability_data_weekly_finetuned_new = json.load(f)

with open(os.path.join(HW_DIR_SCRATCH, "heat_wave_reliability_diagrams_daily_finetuned_new.json"), 'r') as f:
    reliability_data_daily_finetuned_new = json.load(f)
    
#Load anomalies
obs_anomalies_ds = xr.open_zarr("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/observations_anomalies_weekly_27_0.zarr")
ref_anomalies_ds = xr.open_zarr("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/reference_model_anomalies_weekly_27_0.zarr")
weekly_finetuned_anomalies_ds = xr.open_zarr("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/weekly_finetuned_model_anomalies_27_0.zarr")
daily_finetuned_anomalies_ds = xr.open_zarr("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/daily_finetuned_model_weekly_anomalies_3_0.zarr")

obs_anomalies_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/observations_anomalies_weekly_27_0_ROBUST.zarr")
ref_anomalies_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/reference_model_anomalies_weekly_27_0_ROBUST.zarr")
weekly_finetuned_anomalies_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/weekly_finetuned_model_anomalies_27_0_ROBUST.zarr")
daily_finetuned_anomalies_ROBUST = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/daily_finetuned_model_weekly_anomalies_3_0_ROBUST.zarr")

#load climatology
obs_climatology_ds = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/observations_climatology_weekly_27_0.nc")
ref_climatology_ds = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/reference_model_climatology_weekly_27_0.nc")
weekly_finetuned_climatology_ds = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/weekly_finetuned_model_climatology_27_0.nc")
daily_finetuned_climatology_ds = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init/daily_finetuned_model_weekly_climatology_3_0.nc")

obs_climatology_robust = xr.open_dataset("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/observations_climatology_weekly_27_0_ROBUST.nc")
ref_climatology_robust = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/reference_model_climatology_weekly_27_0_ROBUST.zarr")
weekly_finetuned_climatology_robust = xr.open_dataset("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/weekly_finetuned_model_climatology_27_0_ROBUST.nc")
daily_finetuned_climatology_robust = xr.open_zarr("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/daily_finetuned_model_weekly_climatology_3_0_ROBUST.zarr")

#create mask region
# Note: latitude and longitude are flattened (1D with 'values' dimension)
lat = obs_weekly_ds['latitude']
lon = obs_weekly_ds['longitude']
# Expanded analysis region to match map domain: 60W to 90E, 20N to 80N
# Handle longitude wrapping: convert lon > 180 to negative values
lon_adjusted = lon.where(lon <= 180, lon - 360)
region_lat_mask = (lat >= 20) & (lat <= 80)
region_lon_mask = (lon_adjusted >= -60) & (lon_adjusted <= 90)
region_mask = region_lat_mask & region_lon_mask

heatwave_lon_mask = (lon_adjusted >= 10) & (lon_adjusted <= 50)  # Focus on European heatwave region
heatwave_lat_mask = (lat >= 40) & (lat <= 60)
heatwave_mask = heatwave_lat_mask & heatwave_lon_mask

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def find_best_members(datasets_dict, anomaly_datasets_dict, init_date, region_mask):
    """
    Find the ensemble member closest to ground truth for 2t variable.
    Uses RMSE averaged over all lead times and spatial points.
    
    Parameters:
    -----------
    datasets_dict : dict
        Dictionary with raw datasets (to check for member dimension)
    anomaly_datasets_dict : dict
        Dictionary with anomaly datasets to compute RMSE
    init_date : str
        Initialization date to select
    region_mask : xarray.DataArray
        Boolean mask for the region of interest
    
    Returns:
    --------
    dict : Dictionary with best member index for each model dataset
    """
    print("\nFinding best ensemble members (closest to ground truth for 2t)...")
    
    best_members = {}
    model_names = ['ref_anomalies_ds', 'weekly_finetuned_anomalies_ds', 'daily_finetuned_anomalies_ds']
    model_labels = ['Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    
    # Get observation anomalies for 2t
    obs_2t_anom = anomaly_datasets_dict['obs_anomalies_ds']['2t'].sel(init_date=init_date)
    
    for model_name, model_label in zip(model_names, model_labels):
        model_2t_anom = anomaly_datasets_dict[model_name]['2t'].sel(init_date=init_date)
        
        # Check if member dimension exists
        if 'member' in model_2t_anom.dims:
            # Calculate RMSE for each member, averaged over lead times and spatial points
            rmse_per_member = []
            
            for member_idx in range(model_2t_anom.member.size):
                member_data = model_2t_anom.isel(member=member_idx)
                # Calculate squared differences
                sq_diff = (member_data - obs_2t_anom) ** 2
                # Average over space (applying mask) and lead times
                rmse = np.sqrt(sq_diff.where(region_mask).mean(dim=['values', 'week_lead_time']).values)
                rmse_per_member.append(rmse)
            
            # Find member with minimum RMSE
            best_member_idx = int(np.argmin(rmse_per_member))
            best_members[model_name] = best_member_idx
            
            print(f"  {model_label}: Member {best_member_idx} (RMSE={rmse_per_member[best_member_idx]:.4f} K)")
        else:
            best_members[model_name] = None
            print(f"  {model_label}: No ensemble members")
    
    return best_members


def plot_climatology_grid(var_name, climatology_dict, init_date, leadtimes, region_mask, lat_flat, lon_flat, output_dir, best_members=None):
    """
    Create a grid plot for climatology with weeks as rows and datasets as columns.
    For models with ensemble members, uses the best members identified by find_best_members.
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., '2t', 'sp', 'tp', 'msl', 't_850')
    climatology_dict : dict
        Dictionary with keys as dataset names and values as xarray climatology datasets
    init_date : str
        Initialization date to select (climatology for the corresponding week of year)
    leadtimes : list
        List of leadtimes (in weeks) to plot [1, 2, 3, 4, 5, 6, 7, 8]
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    lat_flat, lon_flat : xarray.DataArray
        Flattened latitude and longitude arrays
    output_dir : str
        Directory to save the output figure
    best_members : dict, optional
        Dictionary with best member index for each model dataset
    """
    
    fig = plt.figure(figsize=(20, 32))
    
    # Dataset order for columns (including observations)
    dataset_names = ['obs_climatology_ds', 'ref_climatology_ds', 'weekly_finetuned_climatology_ds', 'daily_finetuned_climatology_ds']
    dataset_labels = ['ERA5', 'Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    
    # Define projection for geographical plots
    projection = ccrs.PlateCarree()
    
    # Convert to numpy arrays for plotting (accept xarray.DataArray or numpy.ndarray)
    lat_np = lat_flat.values if hasattr(lat_flat, 'values') else np.asarray(lat_flat)
    lon_np = lon_flat.values if hasattr(lon_flat, 'values') else np.asarray(lon_flat)
    region_mask_np = region_mask.values if hasattr(region_mask, 'values') else np.asarray(region_mask)
    
    # Set map extent to supervisor's requested domain: 60W to 90E, 20N to 80N
    map_extent = [-60, 90, 20, 80]
    
    # Use provided best_members (already computed based on 2t ground truth)
    # If not provided, default to None for all datasets
    if best_members is None:
        best_members = {ds_name: None for ds_name in dataset_names}
    
    # Calculate single colorbar range for entire plot (all datasets and all leadtimes)
    all_values_global = []
    for leadtime_week in leadtimes:
        for ds_name in dataset_names:
            try:
                ds = climatology_dict[ds_name]
                data_sel = ds[var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
                if 'member' in data_sel.dims:
                    # Use best member if available
                    if best_members.get(ds_name) is not None:
                        data_sel = data_sel.isel(member=best_members[ds_name])
                    else:
                        data_sel = data_sel.mean(dim='member')
                # Convert temperature from Kelvin to Celsius
                if var_name == '2t':
                    data_sel = data_sel - 273.15
                field = data_sel.values
                # For precipitation, clamp to 0 (no negative values)
                if var_name == 'tp':
                    field = np.maximum(field, 0)
                field_masked = np.where(region_mask_np, field, np.nan)
                valid_values = field_masked[np.isfinite(field_masked)]
                if len(valid_values) > 0:
                    all_values_global.extend(valid_values)
            except:
                pass
    
    if len(all_values_global) > 0:
        vmin_global = float(np.nanmin(all_values_global))
        vmax_global = float(np.nanmax(all_values_global))
        # For precipitation, ensure vmin is 0
        if var_name == 'tp':
            vmin_global = 0.0
        # Create explicit contour levels to ensure consistency across all subplots
        levels_global = np.linspace(vmin_global, vmax_global, 41)
    else:
        vmin_global = None
        vmax_global = None
        levels_global = None
    
    # Create subplots (rows=weeks, columns=datasets)
    for row_idx, leadtime_week in enumerate(leadtimes):
        for col_idx, (ds_name, ds_label) in enumerate(zip(dataset_names, dataset_labels)):
            
            subplot_idx = row_idx * 4 + col_idx + 1
            ax = fig.add_subplot(len(leadtimes), 4, subplot_idx, projection=projection)
            
            try:
                # Get the dataset
                ds = climatology_dict[ds_name]
                
                # Select data for the specific init_date and week_lead_time
                data_sel = ds[var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
                
                # Handle member dimension if present (use best member)
                if 'member' in data_sel.dims:
                    if best_members.get(ds_name) is not None:
                        data_sel = data_sel.isel(member=best_members[ds_name])
                    else:
                        data_sel = data_sel.mean(dim='member')
                
                # Convert temperature from Kelvin to Celsius
                if var_name == '2t':
                    data_sel = data_sel - 273.15
                
                # Convert to numpy array
                field = data_sel.values
                
                # For precipitation, clamp to 0 (no negative values)
                if var_name == 'tp':
                    field = np.maximum(field, 0)
                
                # Apply region mask - only plot data within the region of interest
                field_masked = np.where(region_mask_np, field, np.nan)
                
                # Create mask for finite values
                mask = np.isfinite(lon_np) & np.isfinite(lat_np) & np.isfinite(field_masked)
                
                # Adjust longitude to [-180, 180] if needed
                lon_plot = np.where(lon_np[mask] > 180, lon_np[mask] - 360, lon_np[mask])
                lat_plot = lat_np[mask]
                field_plot = field_masked[mask]
                
                # Use consistent colorbar range for entire plot (all models and leadtimes)
                if len(field_plot) > 0 and np.any(np.isfinite(field_plot)):
                    # Use sequential colormap for precipitation (white at 0, darker with more precip)
                    if var_name == 'tp':
                        cmap_to_use = 'YlGnBu'
                        extend_opt = 'max'  # Only extend on high end for precipitation
                    else:
                        cmap_to_use = 'RdYlBu_r'
                        extend_opt = 'both'
                    
                    if levels_global is not None:
                        # Use tricontourf with explicit levels to ensure consistent colorbars
                        cf = ax.tricontourf(lon_plot, lat_plot, field_plot, levels=levels_global,
                                           transform=projection,
                                           cmap=cmap_to_use,
                                           vmin=vmin_global, vmax=vmax_global,
                                           extend=extend_opt)
                    else:
                        # Fallback if colorbar range couldn't be determined
                        cf = ax.tricontourf(lon_plot, lat_plot, field_plot, 40,
                                           transform=projection,
                                           cmap=cmap_to_use)
                    
                    # Add colorbar
                    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
                    cbar.ax.tick_params(labelsize=8)
                else:
                    ax.text(0.5, 0.5, 'No valid data', 
                           transform=ax.transAxes,
                           ha='center', va='center', fontsize=12)
                
                # Add coastlines and borders
                ax.coastlines(resolution='50m', linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                
                # Set extent to show the full domain
                ax.set_extent(map_extent, crs=projection)
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                if col_idx > 0:
                    gl.left_labels = False
                if row_idx < len(leadtimes) - 1:
                    gl.bottom_labels = False
                
                # Add column title (model name)
                if row_idx == 0:
                    ax.set_title(ds_label, fontsize=14, fontweight='bold')
                
                # Add row labels (week)
                if col_idx == 0:
                    ax.text(-0.15, 0.5, f'Week {leadtime_week}', 
                           transform=ax.transAxes,
                           fontsize=14, fontweight='bold',
                           rotation=90, va='center', ha='center')
                
            except Exception as e:
                # If there's an error (e.g., variable not found), add text to subplot
                ax.text(0.5, 0.5, f'Data not available\n{str(e)[:50]}', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_extent(map_extent, crs=projection)
                ax.coastlines(resolution='50m', linewidth=0.5)
    
    # Add main title
    fig.suptitle(f'Climatology: {var_name} - Init Date: {pd.to_datetime(init_date).strftime("%Y-%m-%dT%H:%M")}', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_climatology_grid_plot_{init_date.split("T")[0]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved climatology plot for {var_name}: {output_file}")


def plot_variable_grid(var_name, datasets_dict, init_date, leadtimes, region_mask, lat_flat, lon_flat, output_dir, best_members=None):
    """
    Create a grid plot for a single variable (raw fields) with weeks as rows and datasets as columns.
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., '2t', 'sp', 'tp', 'msl', 't_850')
    datasets_dict : dict
        Dictionary with keys as dataset names and values as xarray datasets
    init_date : str
        Initialization date to select
    leadtimes : list
        List of leadtimes (in weeks) to plot [1, 2, 3, 4, 5, 6, 7, 8]
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    lat_flat, lon_flat : xarray.DataArray
        Flattened latitude and longitude arrays
    output_dir : str
        Directory to save the output figure
    best_members : dict, optional
        Dictionary with best member index for each model dataset
    """
    
    fig = plt.figure(figsize=(20, 32))
    
    # Dataset order for columns (including observations)
    dataset_names = ['obs_weekly_ds', 'ref_weekly_ds', 'weekly_finetuned_ds', 'daily_finetuned_ds']
    dataset_labels = ['Observations', 'Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    
    # Define projection for geographical plots
    projection = ccrs.PlateCarree()
    
    # Convert to numpy arrays for plotting (accept xarray.DataArray or numpy.ndarray)
    lat_np = lat_flat.values if hasattr(lat_flat, 'values') else np.asarray(lat_flat)
    lon_np = lon_flat.values if hasattr(lon_flat, 'values') else np.asarray(lon_flat)
    region_mask_np = region_mask.values if hasattr(region_mask, 'values') else np.asarray(region_mask)
    
    # Set map extent to supervisor's requested domain: 60W to 90E, 20N to 80N
    map_extent = [-60, 90, 20, 80]
    
    # Use provided best_members (already computed based on 2t ground truth)
    # If not provided, default to None for all datasets
    if best_members is None:
        best_members = {ds_name: None for ds_name in dataset_names}
    
    # Calculate single colorbar range for entire plot (all datasets and all leadtimes)
    all_values_global = []
    for leadtime_week in leadtimes:
        for ds_name in dataset_names:
            try:
                ds = datasets_dict[ds_name]
                data_sel = ds[var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
                if 'member' in data_sel.dims:
                    # Use best member if available
                    if best_members.get(ds_name) is not None:
                        data_sel = data_sel.isel(member=best_members[ds_name])
                    else:
                        data_sel = data_sel.mean(dim='member')
                # Convert temperature from Kelvin to Celsius
                if var_name == '2t':
                    data_sel = data_sel - 273.15
                field = data_sel.values
                # For precipitation, clamp to 0 (no negative values)
                if var_name == 'tp':
                    field = np.maximum(field, 0)
                field_masked = np.where(region_mask_np, field, np.nan)
                valid_values = field_masked[np.isfinite(field_masked)]
                if len(valid_values) > 0:
                    all_values_global.extend(valid_values)
            except:
                pass
    
    if len(all_values_global) > 0:
        vmin_global = float(np.nanmin(all_values_global))
        vmax_global = float(np.nanmax(all_values_global))
        # For precipitation, ensure vmin is 0
        if var_name == 'tp':
            vmin_global = 0.0
        # Create explicit contour levels to ensure consistency across all subplots
        levels_global = np.linspace(vmin_global, vmax_global, 41)
    else:
        vmin_global = None
        vmax_global = None
        levels_global = None
            
    
    # Create subplots (rows=weeks, columns=datasets)
    for row_idx, leadtime_week in enumerate(leadtimes):
        for col_idx, (ds_name, ds_label) in enumerate(zip(dataset_names, dataset_labels)):
            
            subplot_idx = row_idx * 4 + col_idx + 1
            ax = fig.add_subplot(len(leadtimes), 4, subplot_idx, projection=projection)
            
            try:
                # Get the dataset
                ds = datasets_dict[ds_name]
                
                # Select data for the specific init_date and week_lead_time
                data_sel = ds[var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
                
                # Handle member dimension if present (use best member)
                if 'member' in data_sel.dims:
                    if best_members.get(ds_name) is not None:
                        data_sel = data_sel.isel(member=best_members[ds_name])
                    else:
                        data_sel = data_sel.mean(dim='member')
                
                # Convert temperature from Kelvin to Celsius
                if var_name == '2t':
                    data_sel = data_sel - 273.15
                
                # Convert to numpy array
                field = data_sel.values
                
                # For precipitation, clamp to 0 (no negative values)
                if var_name == 'tp':
                    field = np.maximum(field, 0)
                
                # Apply region mask - only plot data within the region of interest
                field_masked = np.where(region_mask_np, field, np.nan)
                
                # Create mask for finite values
                mask = np.isfinite(lon_np) & np.isfinite(lat_np) & np.isfinite(field_masked)
                
                # Adjust longitude to [-180, 180] if needed
                lon_plot = np.where(lon_np[mask] > 180, lon_np[mask] - 360, lon_np[mask])
                lat_plot = lat_np[mask]
                field_plot = field_masked[mask]
                
                # Use consistent colorbar range for entire plot (all models and leadtimes)
                if len(field_plot) > 0 and np.any(np.isfinite(field_plot)):
                    # Use sequential colormap for precipitation (white at 0, darker with more precip)
                    if var_name == 'tp':
                        cmap_to_use = 'YlGnBu'
                        extend_opt = 'max'  # Only extend on high end for precipitation
                    else:
                        cmap_to_use = 'RdYlBu_r'
                        extend_opt = 'both'
                    
                    if levels_global is not None:
                        # Use tricontourf with explicit levels to ensure consistent colorbars
                        cf = ax.tricontourf(lon_plot, lat_plot, field_plot, levels=levels_global,
                                           transform=projection,
                                           cmap=cmap_to_use,
                                           vmin=vmin_global, vmax=vmax_global,
                                           extend=extend_opt)
                    else:
                        # Fallback if colorbar range couldn't be determined
                        cf = ax.tricontourf(lon_plot, lat_plot, field_plot, 40,
                                           transform=projection,
                                           cmap=cmap_to_use)
                    
                    # Add colorbar
                    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
                    cbar.ax.tick_params(labelsize=8)
                else:
                    ax.text(0.5, 0.5, 'No valid data', 
                           transform=ax.transAxes,
                           ha='center', va='center', fontsize=12)
                
                # Add coastlines and borders
                ax.coastlines(resolution='50m', linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                
                # Set extent to show the full domain
                ax.set_extent(map_extent, crs=projection)
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                if col_idx > 0:
                    gl.left_labels = False
                if row_idx < len(leadtimes) - 1:
                    gl.bottom_labels = False
                
                # Add column title (model name)
                if row_idx == 0:
                    ax.set_title(ds_label, fontsize=14, fontweight='bold')
                
                # Add row labels (week)
                if col_idx == 0:
                    ax.text(-0.15, 0.5, f'Week {leadtime_week}', 
                           transform=ax.transAxes,
                           fontsize=14, fontweight='bold',
                           rotation=90, va='center', ha='center')
                
            except Exception as e:
                # If there's an error (e.g., variable not found), add text to subplot
                ax.text(0.5, 0.5, f'Data not available\n{str(e)[:50]}', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_extent(map_extent, crs=projection)
                ax.coastlines(resolution='50m', linewidth=0.5)
    
    # Add main title
    fig.suptitle(f'Variable: {var_name} - Init Date: {pd.to_datetime(init_date).strftime("%Y-%m-%dT%H:%M")}', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_grid_plot_{init_date.split("T")[0]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {var_name}: {output_file}")


def plot_variable_grid_anomalies(var_name, datasets_dict, init_date, leadtimes, region_mask, lat_flat, lon_flat, output_dir, best_members=None):
    """
    Create a grid plot for a single variable (anomalies) with weeks as rows and datasets as columns.
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., 't2m_anomaly', 'msl_anomaly', etc.)
    datasets_dict : dict
        Dictionary with keys as dataset names and values as xarray anomaly datasets
    init_date : str
        Initialization date to select
    leadtimes : list
        List of leadtimes (in weeks) to plot [1, 2, 3, 4, 5, 6, 7, 8]
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    lat_flat, lon_flat : xarray.DataArray
        Flattened latitude and longitude arrays
    output_dir : str
        Directory to save the output figure
    best_members : dict, optional
        Dictionary with best member index for each model dataset
    """
    
    fig = plt.figure(figsize=(20, 32))
    
    # Dataset order for columns (including observations, using anomaly datasets)
    dataset_names = ['obs_anomalies_ds', 'ref_anomalies_ds', 'weekly_finetuned_anomalies_ds', 'daily_finetuned_anomalies_ds']
    dataset_labels = ['Observations', 'Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    
    # Define projection for geographical plots
    projection = ccrs.PlateCarree()
    
    # Convert to numpy arrays for plotting (accept xarray.DataArray or numpy.ndarray)
    lat_np = lat_flat.values if hasattr(lat_flat, 'values') else np.asarray(lat_flat)
    lon_np = lon_flat.values if hasattr(lon_flat, 'values') else np.asarray(lon_flat)
    region_mask_np = region_mask.values if hasattr(region_mask, 'values') else np.asarray(region_mask)
    
    # Set map extent to supervisor's requested domain: 60W to 90E, 20N to 80N
    map_extent = [-60, 90, 20, 80]
    
    # Use provided best_members (already computed based on 2t ground truth)
    # If not provided, default to None for all datasets
    if best_members is None:
        best_members = {ds_name: None for ds_name in dataset_names}
    
    # Calculate single colorbar range for entire plot (all datasets and all leadtimes)
    all_values_global = []
    for leadtime_week in leadtimes:
        for ds_name in dataset_names:
            try:
                ds = datasets_dict[ds_name]
                data_sel = ds[var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
                if 'member' in data_sel.dims:
                    # Use best member if available
                    if best_members.get(ds_name) is not None:
                        data_sel = data_sel.isel(member=best_members[ds_name])
                    else:
                        data_sel = data_sel.mean(dim='member')
                field = data_sel.values
                field_masked = np.where(region_mask_np, field, np.nan)
                valid_values = field_masked[np.isfinite(field_masked)]
                if len(valid_values) > 0:
                    all_values_global.extend(valid_values)
            except:
                pass
    
    if len(all_values_global) > 0:
        vmin_global = float(np.nanmin(all_values_global))
        vmax_global = float(np.nanmax(all_values_global))
        # Create explicit contour levels to ensure consistency across all subplots
        levels_global = np.linspace(vmin_global, vmax_global, 41)
    else:
        vmin_global = None
        vmax_global = None
        levels_global = None
            
    
    # Create subplots (rows=weeks, columns=datasets)
    for row_idx, leadtime_week in enumerate(leadtimes):
        for col_idx, (ds_name, ds_label) in enumerate(zip(dataset_names, dataset_labels)):
            
            subplot_idx = row_idx * 4 + col_idx + 1
            ax = fig.add_subplot(len(leadtimes), 4, subplot_idx, projection=projection)
            
            try:
                # Get the dataset
                ds = datasets_dict[ds_name]
                
                # Select data for the specific init_date and week_lead_time
                data_sel = ds[var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
                
                # Handle member dimension if present (use best member)
                if 'member' in data_sel.dims:
                    if best_members.get(ds_name) is not None:
                        data_sel = data_sel.isel(member=best_members[ds_name])
                    else:
                        data_sel = data_sel.mean(dim='member')
                
                # Convert to numpy array
                field = data_sel.values
                
                # Apply region mask - only plot data within the region of interest
                field_masked = np.where(region_mask_np, field, np.nan)
                
                # Create mask for finite values
                mask = np.isfinite(lon_np) & np.isfinite(lat_np) & np.isfinite(field_masked)
                
                # Adjust longitude to [-180, 180] if needed
                lon_plot = np.where(lon_np[mask] > 180, lon_np[mask] - 360, lon_np[mask])
                lat_plot = lat_np[mask]
                field_plot = field_masked[mask]
                
                # Use consistent colorbar range for entire plot (all models and leadtimes)
                if len(field_plot) > 0 and np.any(np.isfinite(field_plot)):
                    # Use diverging colormap for anomalies (centered at 0)
                    cmap_to_use = 'RdBu_r'
                    extend_opt = 'both'
                    
                    if levels_global is not None:
                        # Use tricontourf with explicit levels to ensure consistent colorbars
                        cf = ax.tricontourf(lon_plot, lat_plot, field_plot, levels=levels_global,
                                           transform=projection,
                                           cmap=cmap_to_use,
                                           vmin=vmin_global, vmax=vmax_global,
                                           extend=extend_opt)
                    else:
                        # Fallback if colorbar range couldn't be determined
                        cf = ax.tricontourf(lon_plot, lat_plot, field_plot, 40,
                                           transform=projection,
                                           cmap=cmap_to_use)
                    
                    # Add colorbar
                    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
                    cbar.ax.tick_params(labelsize=8)
                else:
                    ax.text(0.5, 0.5, 'No valid data', 
                           transform=ax.transAxes,
                           ha='center', va='center', fontsize=12)
                
                # Add coastlines and borders
                ax.coastlines(resolution='50m', linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                
                # Set extent to show the full domain
                ax.set_extent(map_extent, crs=projection)
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                if col_idx > 0:
                    gl.left_labels = False
                if row_idx < len(leadtimes) - 1:
                    gl.bottom_labels = False
                
                # Add column title (model name)
                if row_idx == 0:
                    ax.set_title(ds_label, fontsize=14, fontweight='bold')
                
                # Add row labels (week)
                if col_idx == 0:
                    ax.text(-0.15, 0.5, f'Week {leadtime_week}', 
                           transform=ax.transAxes,
                           fontsize=14, fontweight='bold',
                           rotation=90, va='center', ha='center')
                
            except Exception as e:
                # If there's an error (e.g., variable not found), add text to subplot
                ax.text(0.5, 0.5, f'Data not available\n{str(e)[:50]}', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_extent(map_extent, crs=projection)
                ax.coastlines(resolution='50m', linewidth=0.5)
    
    # Add main title
    fig.suptitle(f'Variable Anomaly: {var_name} - Init Date: {pd.to_datetime(init_date).strftime("%Y-%m-%dT%H:%M")}', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_anomaly_grid_plot_{init_date.split("T")[0]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {var_name}: {output_file}")


def plot_combined_anomaly_fields(base_var, overlay_var, datasets_dict, init_date, leadtimes, region_mask, 
                                  lat_flat, lon_flat, output_dir, best_members=None, overlay_levels=None):
    """
    Create a grid plot combining two anomaly fields: base variable as filled contours and overlay variable as contour lines.
    Only positive anomalies are shown for the overlay variable (contour lines).
    
    Parameters:
    -----------
    base_var : str
        Base variable name for filled contours (e.g., '2t')
    overlay_var : str
        Overlay variable name for contour lines (e.g., 'z_500'). Only positive anomalies plotted.
    datasets_dict : dict
        Dictionary with keys as dataset names and values as xarray anomaly datasets
    init_date : str
        Initialization date to select
    leadtimes : list
        List of leadtimes (in weeks) to plot [1, 2, 3, 4, 5, 6, 7, 8]
    region_mask : xarray.DataArray or numpy.ndarray
        Boolean mask for the region of interest (flattened)
    lat_flat, lon_flat : xarray.DataArray or numpy.ndarray
        Flattened latitude and longitude arrays
    output_dir : str
        Directory to save the output figure
    best_members : dict, optional
        Dictionary with best member index for each model dataset
    overlay_levels : int or array-like, optional
        Number of contour levels or specific levels for overlay variable (positive only)
    """
    
    fig = plt.figure(figsize=(20, 32))
    
    # Dataset order for columns
    dataset_names = ['obs_anomalies_ds', 'ref_anomalies_ds', 'weekly_finetuned_anomalies_ds', 'daily_finetuned_anomalies_ds']
    dataset_labels = ['ERA5', 'Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    
    # Define projection for geographical plots
    projection = ccrs.PlateCarree()
    
    # Convert to numpy arrays for plotting (handle both xarray and numpy inputs)
    lat_np = lat_flat.values if hasattr(lat_flat, 'values') else lat_flat
    lon_np = lon_flat.values if hasattr(lon_flat, 'values') else lon_flat
    region_mask_np = region_mask.values if hasattr(region_mask, 'values') else region_mask
    
    # Set map extent
    map_extent = [-60, 90, 20, 80]
    
    # Use provided best_members
    if best_members is None:
        best_members = {ds_name: None for ds_name in dataset_names}
    
    # Calculate colorbar range for base variable
    all_values_base = []
    for leadtime_week in leadtimes:
        for ds_name in dataset_names:
            try:
                ds = datasets_dict[ds_name]
                data_sel = ds[base_var].sel(init_date=init_date, week_lead_time=leadtime_week)
                if 'member' in data_sel.dims:
                    if best_members.get(ds_name) is not None:
                        data_sel = data_sel.isel(member=best_members[ds_name])
                    else:
                        data_sel = data_sel.mean(dim='member')
                field = data_sel.values
                field_masked = np.where(region_mask_np, field, np.nan)
                valid_values = field_masked[np.isfinite(field_masked)]
                if len(valid_values) > 0:
                    all_values_base.extend(valid_values)
            except:
                pass
    
    if len(all_values_base) > 0:
        vmin_base = float(np.nanmin(all_values_base))
        vmax_base = float(np.nanmax(all_values_base))
        levels_base = np.linspace(vmin_base, vmax_base, 41)
    else:
        vmin_base = None
        vmax_base = None
        levels_base = None
    
    # Create subplots
    for row_idx, leadtime_week in enumerate(leadtimes):
        for col_idx, (ds_name, ds_label) in enumerate(zip(dataset_names, dataset_labels)):
            
            subplot_idx = row_idx * 4 + col_idx + 1
            ax = fig.add_subplot(len(leadtimes), 4, subplot_idx, projection=projection)
            
            try:
                ds = datasets_dict[ds_name]
                
                # --- BASE VARIABLE (filled contours) ---
                base_data = ds[base_var].sel(init_date=init_date, week_lead_time=leadtime_week)
                if 'member' in base_data.dims:
                    if best_members.get(ds_name) is not None:
                        base_data = base_data.isel(member=best_members[ds_name])
                    else:
                        base_data = base_data.mean(dim='member')
                
                base_field = base_data.values
                base_masked = np.where(region_mask_np, base_field, np.nan)
                
                # --- OVERLAY VARIABLE (contour lines) ---
                overlay_data = ds[overlay_var].sel(init_date=init_date, week_lead_time=leadtime_week)
                if 'member' in overlay_data.dims:
                    if best_members.get(ds_name) is not None:
                        overlay_data = overlay_data.isel(member=best_members[ds_name])
                    else:
                        overlay_data = overlay_data.mean(dim='member')
                
                overlay_field = overlay_data.values
                overlay_masked = np.where(region_mask_np, overlay_field, np.nan)
                
                # Create mask for finite values
                mask_base = np.isfinite(lon_np) & np.isfinite(lat_np) & np.isfinite(base_masked)
                mask_overlay = np.isfinite(lon_np) & np.isfinite(lat_np) & np.isfinite(overlay_masked)
                
                # Adjust longitude
                lon_plot_base = np.where(lon_np[mask_base] > 180, lon_np[mask_base] - 360, lon_np[mask_base])
                lat_plot_base = lat_np[mask_base]
                field_plot_base = base_masked[mask_base]
                
                lon_plot_overlay = np.where(lon_np[mask_overlay] > 180, lon_np[mask_overlay] - 360, lon_np[mask_overlay])
                lat_plot_overlay = lat_np[mask_overlay]
                field_plot_overlay = overlay_masked[mask_overlay]
                
                # Plot base variable (filled contours)
                if len(field_plot_base) > 0 and np.any(np.isfinite(field_plot_base)):
                    if levels_base is not None:
                        cf = ax.tricontourf(lon_plot_base, lat_plot_base, field_plot_base, 
                                           levels=levels_base,
                                           transform=projection,
                                           cmap='RdBu_r',
                                           vmin=vmin_base, vmax=vmax_base,
                                           extend='both')
                    else:
                        cf = ax.tricontourf(lon_plot_base, lat_plot_base, field_plot_base, 40,
                                           transform=projection, cmap='RdBu_r')
                    
                    # Add colorbar for base variable
                    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
                    cbar.ax.tick_params(labelsize=8)
                    cbar.set_label(f'{base_var} anomaly', fontsize=14)
                
                # Overlay contour lines for second variable (positive anomalies only)
                if len(field_plot_overlay) > 0 and np.any(np.isfinite(field_plot_overlay)):
                    # Only plot positive anomalies
                    max_overlay = np.nanmax(field_plot_overlay)
                    
                    if max_overlay > 0:  # Only plot if there are positive anomalies
                        # Determine contour levels for overlay (positive values only)
                        if overlay_levels is None:
                            # Auto-determine reasonable contour levels
                            # Start from a small positive value to avoid zero
                            min_positive = max(0.1 * max_overlay, 10)  # Start at 10% of max or 10, whichever is larger
                            overlay_levels = np.linspace(min_positive, max_overlay, 8)
                        elif isinstance(overlay_levels, int):
                            # If integer provided, create that many levels from min positive to max
                            min_positive = max(0.1 * max_overlay, 10)
                            overlay_levels = np.linspace(min_positive, max_overlay, overlay_levels)
                        # else: use provided levels array as-is
                        
                        cs = ax.tricontour(lon_plot_overlay, lat_plot_overlay, field_plot_overlay,
                                          levels=overlay_levels,
                                          transform=projection,
                                          colors='black',
                                          linewidths=1.2,
                                          alpha=0.8)
                        
                        # Add contour labels
                        ax.clabel(cs, inline=True, fontsize=14, fmt='%1.0f')
                
                # Add coastlines and borders
                ax.coastlines(resolution='50m', linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.set_extent(map_extent, crs=projection)
                
                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                if col_idx > 0:
                    gl.left_labels = False
                if row_idx < len(leadtimes) - 1:
                    gl.bottom_labels = False
                
                # Add column title
                if row_idx == 0:
                    ax.set_title(ds_label, fontsize=14, fontweight='bold')
                
                # Add row labels
                if col_idx == 0:
                    ax.text(-0.15, 0.5, f'Week {leadtime_week}', 
                           transform=ax.transAxes,
                           fontsize=14, fontweight='bold',
                           rotation=90, va='center', ha='center')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Data not available\n{str(e)[:50]}', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_extent(map_extent, crs=projection)
                ax.coastlines(resolution='50m', linewidth=0.5)
    
    # Add main title
    fig.suptitle(f'Combined Anomalies: 2 meter temperature (filled) + Z500 (positive contours only)\n' + 
                f'Init Date: {pd.to_datetime(init_date).strftime("%Y-%m-%dT%H:%M")}', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{base_var}_{overlay_var}_combined_anomaly_{init_date.split("T")[0]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined anomaly plot: {output_file}")


def plot_error_maps(var_name, datasets_dict, init_date, leadtimes, region_mask, lat_flat, lon_flat, output_dir, best_members=None):
    """
    Create error maps showing (model - observation) with weeks as rows and models as columns.
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., '2t', 'sp', 'tp', 'msl', 't_850')
    datasets_dict : dict
        Dictionary with keys as dataset names and values as xarray datasets
    init_date : str
        Initialization date to select
    leadtimes : list
        List of leadtimes (in weeks) to plot [1, 2, 3, 4, 5, 6, 7, 8]
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    lat_flat, lon_flat : xarray.DataArray
        Flattened latitude and longitude arrays
    output_dir : str
        Directory to save the output figure
    best_members : dict, optional
        Dictionary with best member index for each model dataset
    """
    
    fig = plt.figure(figsize=(15, 32))
    
    # Model order for columns
    model_names = ['ref_weekly_ds', 'weekly_finetuned_ds', 'daily_finetuned_ds']
    model_labels = ['Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    
    # Define projection for geographical plots
    projection = ccrs.PlateCarree()
    
    # Convert to numpy arrays for plotting (accept xarray.DataArray or numpy.ndarray)
    lat_np = lat_flat.values if hasattr(lat_flat, 'values') else np.asarray(lat_flat)
    lon_np = lon_flat.values if hasattr(lon_flat, 'values') else np.asarray(lon_flat)
    region_mask_np = region_mask.values if hasattr(region_mask, 'values') else np.asarray(region_mask)
    
    # Set map extent to supervisor's requested domain: 60W to 90E, 20N to 80N
    map_extent = [-60, 90, 20, 80]
    
    # Use provided best_members
    if best_members is None:
        best_members = {model_name: None for model_name in model_names}
    
    # Calculate single colorbar range for entire error plot (all models and leadtimes)
    all_errors_global = []
    for leadtime_week in leadtimes:
        for model_name in model_names:

            obs_ds_key = 'obs_weekly_ds'

            # Get observation data
            obs_data = datasets_dict[obs_ds_key][var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
            # Convert temperature from Kelvin to Celsius
            if var_name == '2t':
                obs_data = obs_data - 273.15
            obs_field = obs_data.values

            # Get model data
            model_data = datasets_dict[model_name][var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
            if 'member' in model_data.dims:
                if best_members.get(model_name) is not None:
                    model_data = model_data.isel(member=best_members[model_name])
                else:
                    model_data = model_data.mean(dim='member')
            # Convert temperature from Kelvin to Celsius
            if var_name == '2t':
                model_data = model_data - 273.15
            model_field = model_data.values

            # Calculate error: model - observation
            error_field = model_field - obs_field
            error_masked = np.where(region_mask_np, error_field, np.nan)
            valid_errors = error_masked[np.isfinite(error_masked)]
            if len(valid_errors) > 0:
                all_errors_global.extend(valid_errors)

    
    if len(all_errors_global) > 0:
        # Use symmetric colorbar range centered at 0
        abs_max = float(np.nanmax(np.abs(all_errors_global)))
        # For precipitation, clamp error range to reasonable values for better detail
        if var_name == 'tp':
            abs_max = min(abs_max, 0.15)  # Clamp to ±0.15 m for precipitation errors
        vmin_global = -abs_max
        vmax_global = abs_max
        # Create explicit contour levels to ensure consistency
        levels_global = np.linspace(vmin_global, vmax_global, 41)
    else:
        vmin_global = None
        vmax_global = None
        levels_global = None
    
    # Create subplots (rows=weeks, columns=models)
    for row_idx, leadtime_week in enumerate(leadtimes):
        for col_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            
            subplot_idx = row_idx * 3 + col_idx + 1
            ax = fig.add_subplot(len(leadtimes), 3, subplot_idx, projection=projection)
            

            obs_ds_key = 'obs_weekly_ds'
            
            # Get observation data
            obs_data = datasets_dict[obs_ds_key][var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
            # Convert temperature from Kelvin to Celsius
            if var_name == '2t':
                obs_data = obs_data - 273.15
            obs_field = obs_data.values
            
            # Get model data
            model_data = datasets_dict[model_name][var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
            if 'member' in model_data.dims:
                if best_members.get(model_name) is not None:
                    model_data = model_data.isel(member=best_members[model_name])
                else:
                    model_data = model_data.mean(dim='member')
            # Convert temperature from Kelvin to Celsius
            if var_name == '2t':
                model_data = model_data - 273.15
            model_field = model_data.values
            
            # Calculate error: model - observation
            error_field = model_field - obs_field
            
            # Apply region mask
            error_masked = np.where(region_mask_np, error_field, np.nan)
            
            # Create mask for finite values
            mask = np.isfinite(lon_np) & np.isfinite(lat_np) & np.isfinite(error_masked)
            
            # Adjust longitude to [-180, 180] if needed
            lon_plot = np.where(lon_np[mask] > 180, lon_np[mask] - 360, lon_np[mask])
            lat_plot = lat_np[mask]
            error_plot = error_masked[mask]
            
            # Calculate mean error over the domain for the title
            mean_error = np.nanmean(error_masked)
            
            # Use consistent colorbar range for entire plot
            if len(error_plot) > 0 and np.any(np.isfinite(error_plot)):
                if levels_global is not None:
                    # Use tricontourf with explicit levels and diverging colormap (RdBu_r)
                    cf = ax.tricontourf(lon_plot, lat_plot, error_plot, levels=levels_global,
                                       transform=projection,
                                       cmap='RdBu_r',
                                       vmin=vmin_global, vmax=vmax_global,
                                       extend='both')
                else:
                    # Fallback if colorbar range couldn't be determined
                    cf = ax.tricontourf(lon_plot, lat_plot, error_plot, 40,
                                       transform=projection,
                                       cmap='RdBu_r')
                
                # Add colorbar
                cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
                cbar.ax.tick_params(labelsize=8)
            else:
                ax.text(0.5, 0.5, 'No valid data', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
            
            # Add coastlines and borders
            ax.coastlines(resolution='50m', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            
            # Set extent to show the full domain
            ax.set_extent(map_extent, crs=projection)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            if col_idx > 0:
                gl.left_labels = False
            if row_idx < len(leadtimes) - 1:
                gl.bottom_labels = False
            
            # Add column title (model name) and mean error
            if row_idx == 0:
                ax.set_title(f'{model_label}\nMean: {mean_error:.4f}', 
                            fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'Mean: {mean_error:.4f}', fontsize=12)
            
            # Add row labels (week)
            if col_idx == 0:
                ax.text(-0.15, 0.5, f'Week {leadtime_week}', 
                       transform=ax.transAxes,
                       fontsize=14, fontweight='bold',
                       rotation=90, va='center', ha='center')
    
    # Add main title
    fig.suptitle(f'Error (Model - Obs): {var_name} - Init Date: {pd.to_datetime(init_date).strftime("%Y-%m-%dT%H:%M")}', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_error_map_{init_date.split("T")[0]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error map for {var_name}: {output_file}")


def plot_error_maps_anomalies(var_name, datasets_dict, init_date, leadtimes, region_mask, lat_flat, lon_flat, output_dir, best_members=None):
    """
    Create error maps showing (model - observation) anomalies with weeks as rows and models as columns.
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., 't2m_anomaly', 'msl_anomaly', etc.)
    datasets_dict : dict
        Dictionary with keys as dataset names and values as xarray anomaly datasets
    init_date : str
        Initialization date to select
    leadtimes : list
        List of leadtimes (in weeks) to plot [1, 2, 3, 4, 5, 6, 7, 8]
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    lat_flat, lon_flat : xarray.DataArray
        Flattened latitude and longitude arrays
    output_dir : str
        Directory to save the output figure
    best_members : dict, optional
        Dictionary with best member index for each model dataset
    """
    
    fig = plt.figure(figsize=(15, 32))
    
    # Model order for columns (use anomaly datasets)
    model_names = ['ref_anomalies_ds', 'weekly_finetuned_anomalies_ds', 'daily_finetuned_anomalies_ds']
    model_labels = ['Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    
    # Define projection for geographical plots
    projection = ccrs.PlateCarree()
    
    # Convert to numpy arrays for plotting (accept xarray.DataArray or numpy.ndarray)
    lat_np = lat_flat.values if hasattr(lat_flat, 'values') else np.asarray(lat_flat)
    lon_np = lon_flat.values if hasattr(lon_flat, 'values') else np.asarray(lon_flat)
    region_mask_np = region_mask.values if hasattr(region_mask, 'values') else np.asarray(region_mask)
    
    # Set map extent to supervisor's requested domain: 60W to 90E, 20N to 80N
    map_extent = [-60, 90, 20, 80]
    
    # Use provided best_members
    if best_members is None:
        best_members = {model_name: None for model_name in model_names}
    
    # Calculate single colorbar range for entire error plot (all models and leadtimes)
    all_errors_global = []
    for leadtime_week in leadtimes:
        for model_name in model_names:

            obs_ds_key = 'obs_anomalies_ds'

            # Get observation data
            obs_data = datasets_dict[obs_ds_key][var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
            obs_field = obs_data.values

            # Get model data
            model_data = datasets_dict[model_name][var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
            if 'member' in model_data.dims:
                if best_members.get(model_name) is not None:
                    model_data = model_data.isel(member=best_members[model_name])
                else:
                    model_data = model_data.mean(dim='member')
            model_field = model_data.values

            # Calculate error: model - observation
            error_field = model_field - obs_field
            error_masked = np.where(region_mask_np, error_field, np.nan)
            valid_errors = error_masked[np.isfinite(error_masked)]
            if len(valid_errors) > 0:
                all_errors_global.extend(valid_errors)

    
    if len(all_errors_global) > 0:
        # Use symmetric colorbar range centered at 0
        abs_max = float(np.nanmax(np.abs(all_errors_global)))
        vmin_global = -abs_max
        vmax_global = abs_max
        # Create explicit contour levels to ensure consistency
        levels_global = np.linspace(vmin_global, vmax_global, 41)
    else:
        vmin_global = None
        vmax_global = None
        levels_global = None
    
    # Create subplots (rows=weeks, columns=models)
    for row_idx, leadtime_week in enumerate(leadtimes):
        for col_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            
            subplot_idx = row_idx * 3 + col_idx + 1
            ax = fig.add_subplot(len(leadtimes), 3, subplot_idx, projection=projection)
            

            obs_ds_key = 'obs_anomalies_ds'
            
            # Get observation data
            obs_data = datasets_dict[obs_ds_key][var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
            obs_field = obs_data.values
            
            # Get model data
            model_data = datasets_dict[model_name][var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
            if 'member' in model_data.dims:
                if best_members.get(model_name) is not None:
                    model_data = model_data.isel(member=best_members[model_name])
                else:
                    model_data = model_data.mean(dim='member')
            model_field = model_data.values
            
            # Calculate error: model - observation
            error_field = model_field - obs_field
            
            # Apply region mask
            error_masked = np.where(region_mask_np, error_field, np.nan)
            
            # Create mask for finite values
            mask = np.isfinite(lon_np) & np.isfinite(lat_np) & np.isfinite(error_masked)
            
            # Adjust longitude to [-180, 180] if needed
            lon_plot = np.where(lon_np[mask] > 180, lon_np[mask] - 360, lon_np[mask])
            lat_plot = lat_np[mask]
            error_plot = error_masked[mask]
            
            # Calculate mean error over the domain for the title
            mean_error = np.nanmean(error_masked)
            
            # Use consistent colorbar range for entire plot
            if len(error_plot) > 0 and np.any(np.isfinite(error_plot)):
                if levels_global is not None:
                    # Use tricontourf with explicit levels and diverging colormap (RdBu_r)
                    cf = ax.tricontourf(lon_plot, lat_plot, error_plot, levels=levels_global,
                                       transform=projection,
                                       cmap='RdBu_r',
                                       vmin=vmin_global, vmax=vmax_global,
                                       extend='both')
                else:
                    # Fallback if colorbar range couldn't be determined
                    cf = ax.tricontourf(lon_plot, lat_plot, error_plot, 40,
                                       transform=projection,
                                       cmap='RdBu_r')
                
                # Add colorbar
                cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
                cbar.ax.tick_params(labelsize=8)
            else:
                ax.text(0.5, 0.5, 'No valid data', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
            
            # Add coastlines and borders
            ax.coastlines(resolution='50m', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            
            # Set extent to show the full domain
            ax.set_extent(map_extent, crs=projection)
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            if col_idx > 0:
                gl.left_labels = False
            if row_idx < len(leadtimes) - 1:
                gl.bottom_labels = False
            
            # Add column title (model name) and mean error
            if row_idx == 0:
                ax.set_title(f'{model_label}\nMean: {mean_error:.4f}', 
                            fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'Mean: {mean_error:.4f}', fontsize=12)
            
            # Add row labels (week)
            if col_idx == 0:
                ax.text(-0.15, 0.5, f'Week {leadtime_week}', 
                       transform=ax.transAxes,
                       fontsize=14, fontweight='bold',
                       rotation=90, va='center', ha='center')
    
    # Add main title
    fig.suptitle(f'Error (Model - Obs) Anomaly: {var_name} - Init Date: {pd.to_datetime(init_date).strftime("%Y-%m-%dT%H:%M")}', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_anomaly_error_map_{init_date.split("T")[0]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error map for {var_name}: {output_file}")


def plot_ensemble_plumes(var_name, datasets_dict, init_date, region_mask, output_dir, best_members=None):
    """
    Create ensemble plume plots showing temporal evolution for different models.
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., '2t', 'sp', 'tp', 'msl', 't_850')
    datasets_dict : dict
        Dictionary with keys as dataset names and values as xarray datasets
    init_date : str
        Single initialization date to plot
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    output_dir : str
        Directory to save the output figure
    best_members : dict, optional
        Dictionary with best member index for each model dataset
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Model configuration
    model_names = ['ref_weekly_ds', 'weekly_finetuned_ds', 'daily_finetuned_ds']
    model_labels = ['Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    model_colors = {
        'ref_weekly_ds': '#1f77b4',      # blue
        'weekly_finetuned_ds': '#2ca02c',    # green
        'daily_finetuned_ds': '#E69F00', # orange
    }
    
    # Variable units for y-axis labels
    var_units = {
        "2t": "°C", 
        "sp": "Pa", 
        "tp": "m", 
        "msl": "Pa", 
        "t_850": "K",
        "z_500": "m²/s²",
        "2d": "K",
        "skt": "K",
        "10u": "m/s",
        "10v": "m/s"
    }
    
    # Week lead times (assuming 1-8 weeks)
    weeks = np.arange(1, 9)
    
    # Calculate global y-limits across all models for this init_date
    all_values_global = []
    
    # Get observation data
    obs_data = datasets_dict['obs_weekly_ds'][var_name].sel(init_date=init_date)
    # Convert temperature from Kelvin to Celsius
    if var_name == '2t':
        obs_data = obs_data - 273.15
    obs_regional_mean = obs_data.where(region_mask).mean(dim='values')
    all_values_global.append(obs_regional_mean.values)
    
    # Get model data
    for model_name in model_names:
        model_data = datasets_dict[model_name][var_name].sel(init_date=init_date)
        # Convert temperature from Kelvin to Celsius
        if var_name == '2t':
            model_data = model_data - 273.15
        model_regional = model_data.where(region_mask).mean(dim='values')
        
        if 'member' in model_regional.dims:
            all_values_global.append(model_regional.values.flatten())
        else:
            all_values_global.append(model_regional.values.flatten())
    
    all_values_global = np.concatenate([v.flatten() for v in all_values_global])
    y_min = np.nanmin(all_values_global)
    y_max = np.nanmax(all_values_global)
    y_margin = (y_max - y_min) * 0.05
    y_limits = [y_min - y_margin, y_max + y_margin]
    
    # Process each model (columns)
    for col_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
        
        ax = axes[col_idx]
            

        obs_ds_key = 'obs_weekly_ds'
            
        # Get observation data for this init_date (spatial mean over region)
        obs_data = datasets_dict[obs_ds_key][var_name].sel(init_date=init_date)
        # Convert temperature from Kelvin to Celsius
        if var_name == '2t':
            obs_data = obs_data - 273.15
        obs_regional_mean = obs_data.where(region_mask).mean(dim='values')
        
        # Get model data
        model_data = datasets_dict[model_name][var_name].sel(init_date=init_date)
        # Convert temperature from Kelvin to Celsius
        if var_name == '2t':
            model_data = model_data - 273.15
        
        # Take spatial mean over region
        model_regional = model_data.where(region_mask).mean(dim='values')
        
        # Plot ensemble members if present
        if 'member' in model_regional.dims:
            for member in range(model_regional.member.size):
                member_data = model_regional.isel(member=member)
                ax.plot(weeks, member_data.values, 
                       color=model_colors[model_name], 
                       alpha=0.3, linewidth=1)
            
            # Highlight the selected best member (used in other plots)
            if best_members is not None and best_members.get(model_name) is not None:
                best_member_idx = best_members[model_name]
                best_member_data = model_regional.isel(member=best_member_idx)
                ax.plot(weeks, best_member_data.values, 
                       color=model_colors[model_name], 
                       alpha=0.9, linewidth=2.5, linestyle='--',
                       marker='s', markersize=5,
                       label='Selected Member')
            
            # Plot ensemble mean (thick line)
            ensemble_mean = model_regional.mean(dim='member')
            ax.plot(weeks, ensemble_mean.values, 
                   color=model_colors[model_name], 
                   linewidth=3, label='Ensemble Mean')
        else:
            # No member dimension (e.g., observations)
            ax.plot(weeks, model_regional.values, 
                   color=model_colors[model_name], 
                   linewidth=3, label='Model')
        

        obs_label = 'Observation'
        ax.plot(weeks, obs_regional_mean.values, 
               'k-', linewidth=2, marker='o', 
               markersize=4, label=obs_label)
        
        # Formatting
        ax.set_ylim(y_limits)
        ax.grid(True, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Week Lead Time', fontsize=14)
        if col_idx == 0:  # First subplot
            ax.set_ylabel(f'{var_units.get(var_name, "")}', fontsize=14)
        
        # Model name as title
        ax.set_title(model_label, fontsize=14, fontweight='bold')
        
        # Legend on first subplot
        if col_idx == 0:
            ax.legend(loc='best', fontsize=12)
                
    
    # Overall title with init_date
    init_date_str = pd.to_datetime(init_date).strftime("%Y-%m-%d")
    fig.suptitle(
        f'Ensemble Plumes: {var_name} - Init Date: {init_date_str}\n'
        'Regional mean over: Latitude [20°N, 80°N], Longitude [60°W, 90°E]',
        fontsize=14,
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_ensemble_plumes.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ensemble plume plot for {var_name}: {output_file}")
    

def plot_ensemble_plumes_anomalies(var_name, anomalies_dict, init_date, region_mask, output_dir, best_members=None):
    """
    Create ensemble plume plots showing temporal evolution for different models (anomalies).
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., '2t', 'sp', 'tp', 'msl', 't_850')
    anomalies_dict : dict
        Dictionary with keys as anomaly dataset names and values as xarray datasets
    init_date : str
        Single initialization date to plot
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    output_dir : str
        Directory to save the output figure
    best_members : dict, optional
        Dictionary with best member index for each model dataset (using anomaly dataset keys)
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Model configuration (using anomaly dataset keys)
    model_names = ['ref_anomalies_ds', 'weekly_finetuned_anomalies_ds', 'daily_finetuned_anomalies_ds']
    model_labels = ['Reference Model', 'Weekly Finetuned', 'Daily Finetuned']
    model_colors = {
        'ref_anomalies_ds': '#1f77b4',      # blue
        'weekly_finetuned_anomalies_ds': '#2ca02c',    # green
        'daily_finetuned_anomalies_ds':  '#E69F00', # orange
    }
    
    # Variable units for y-axis labels (anomalies have same units as raw fields)
    var_units = {
        "2t": "°C", 
        "sp": "Pa", 
        "tp": "m", 
        "msl": "Pa", 
        "t_850": "K",
        "z_500": "m²/s²",
        "2d": "K",
        "skt": "K",
        "10u": "m/s",
        "10v": "m/s"
    }
    
    # Week lead times (assuming 1-8 weeks)
    weeks = np.arange(1, 9)
    
    # Calculate global y-limits across all models for this init_date
    all_values_global = []
    
    # Get observation anomaly data
    obs_data = anomalies_dict['obs_anomalies_ds'][var_name].sel(init_date=init_date)
    # Note: No temperature conversion needed - anomalies are already difference values
    obs_regional_mean = obs_data.where(region_mask).mean(dim='values')
    all_values_global.append(obs_regional_mean.values)
    
    # Get model anomaly data
    for model_name in model_names:
        model_data = anomalies_dict[model_name][var_name].sel(init_date=init_date)
        # Note: No temperature conversion needed - anomalies are already difference values
        model_regional = model_data.where(region_mask).mean(dim='values')
        
        if 'member' in model_regional.dims:
            all_values_global.append(model_regional.values.flatten())
        else:
            all_values_global.append(model_regional.values.flatten())
    
    all_values_global = np.concatenate([v.flatten() for v in all_values_global])
    y_min = np.nanmin(all_values_global)
    y_max = np.nanmax(all_values_global)
    y_margin = (y_max - y_min) * 0.05
    y_limits = [y_min - y_margin, y_max + y_margin]
    
    # Process each model (columns)
    for col_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
        
        ax = axes[col_idx]
            
        obs_ds_key = 'obs_anomalies_ds'
            
        # Get observation anomaly data for this init_date (spatial mean over region)
        obs_data = anomalies_dict[obs_ds_key][var_name].sel(init_date=init_date)
        # Note: No temperature conversion needed - anomalies are already difference values
        obs_regional_mean = obs_data.where(region_mask).mean(dim='values')
        
        # Get model anomaly data
        model_data = anomalies_dict[model_name][var_name].sel(init_date=init_date)
        # Note: No temperature conversion needed - anomalies are already difference values
        
        # Take spatial mean over region
        model_regional = model_data.where(region_mask).mean(dim='values')
        
        # Plot ensemble members if present
        if 'member' in model_regional.dims:
            for member in range(model_regional.member.size):
                member_data = model_regional.isel(member=member)
                ax.plot(weeks, member_data.values, 
                       color=model_colors[model_name], 
                       alpha=0.3, linewidth=1)
            
            # Highlight the selected best member (used in other plots)
            if best_members is not None and best_members.get(model_name) is not None:
                best_member_idx = best_members[model_name]
                best_member_data = model_regional.isel(member=best_member_idx)
                ax.plot(weeks, best_member_data.values, 
                       color=model_colors[model_name], 
                       alpha=0.9, linewidth=2.5, linestyle='--',
                       marker='s', markersize=5,
                       label='Selected Member')
            
            # Plot ensemble mean (thick line)
            ensemble_mean = model_regional.mean(dim='member')
            ax.plot(weeks, ensemble_mean.values, 
                   color=model_colors[model_name], 
                   linewidth=3, label='Ensemble Mean')
        else:
            # No member dimension (e.g., observations)
            ax.plot(weeks, model_regional.values, 
                   color=model_colors[model_name], 
                   linewidth=3, label='Model')
        
        obs_label = 'Observation'
        ax.plot(weeks, obs_regional_mean.values, 
               'k-', linewidth=2, marker='o', 
               markersize=4, label=obs_label)
        
        # Formatting
        ax.set_ylim(y_limits)
        ax.grid(True, alpha=0.3)
        
        # Labels
        ax.set_xlabel('Week Lead Time', fontsize=14)
        if col_idx == 0:  # First subplot
            ax.set_ylabel(f'{var_units.get(var_name, "")}', fontsize=14)
        
        # Model name as title
        ax.set_title(model_label, fontsize=14, fontweight='bold')
        
        # Legend on first subplot
        if col_idx == 0:
            ax.legend(loc='best', fontsize=12)
                
    
    # Overall title with init_date
    init_date_str = pd.to_datetime(init_date).strftime("%Y-%m-%d")
    fig.suptitle(
        f'Ensemble Plumes: {var_name} Anomaly - Init Date: {init_date_str}\n'
        'Regional mean over: Latitude [20°N, 80°N], Longitude [60°W, 90°E]',
        fontsize=14,
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_ensemble_plumes_anomalies.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ensemble plume plot (anomalies) for {var_name}: {output_file}")
    
    
def plot_probabilistic_metrics(list_ds_hw_metrics_aggregated, list_reliability_data, list_model_names, output_dir):
    """
    Create visualization of probabilistic metrics:
    1. Brier Skill Score evolution with lead time (3 severities per subplot, 3 model subplots)
    2. Reliability diagrams for each lead time and model
    """
    print("\nCreating probabilistic metric plots...")
    
    # 1. Brier Skill Score evolution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Heat Wave Prediction Skill over Europe: Brier Skill Score by Lead Time', 
                 fontsize=18, fontweight='bold')
    
    # Color scheme for models (matching the reliability diagram colors)
    model_colors = {
        0: '#1f77b4',    # Blue - Reference
        1: '#2ca02c',    # Green - Weekly Finetuned  
        2: '#E69F00'     # Orange - Daily Finetuned
    }
    
    severities = ['moderate', 'severe', 'extreme']
    severity_labels = {'moderate': 'Moderate', 'severe': 'Severe', 'extreme': 'Extreme'}
    
    for idx, severity in enumerate(severities):
        ax = axes[idx]
        
        # Plot each model's BSS for this severity
        for model_idx, (metrics_by_severity, model_name) in enumerate(zip(list_ds_hw_metrics_aggregated, list_model_names)):
            metrics = metrics_by_severity.sel(severity=severity)
            lead_times = metrics.week_lead_time.values
            bss_values = metrics['brier_skill_score'].values
            
            # Plot BSS for this model
            ax.plot(lead_times, bss_values, 'o-', linewidth=2.5, markersize=8, 
                   color=model_colors[model_idx], label=model_name)
        
        # Reference line at BSS=0
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Lead Time (weeks)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Brier Skill Score', fontsize=14, fontweight='bold')
        ax.set_title(f'{severity_labels[severity]} Heat Waves', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(lead_times)
        
        # Set y-limits for consistency
        ax.set_ylim([-.25, 0.75])
    
    # Add shared legend below subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=min(len(labels), 3),
        fontsize=14,
        frameon=True
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    output_file = os.path.join(output_dir, 'brier_skill_score_evolution_all_models.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # 2. Reliability diagrams (all models on same subplots, different colors per model)
    # Color scheme for models (matching BSS plot but for models)
    model_colors = {
        0: '#1f77b4',    # Blue - Reference
        1: '#2ca02c',    # Green - Weekly Finetuned
        2: '#E69F00'     # Orange - Daily Finetuned
    }
    
    for severity in ['moderate', 'severe', 'extreme']:
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        fig.suptitle(f'Reliability Diagrams: {severity.capitalize()} Heat Waves - All Models\n' +
                    '(Circle size proportional to number of samples)', 
                     fontsize=18, fontweight='bold')
        
        # Get all weeks from first model
        first_metrics = list_reliability_data[0][severity]
        weeks_sorted = sorted(first_metrics.keys(), key=lambda x: int(x.split('_')[1]))
        
        for idx, week in enumerate(weeks_sorted):
            ax = axes[idx // 4, idx % 4]
            week_num = int(week.split('_')[1])
            
            # Plot perfect reliability line (only once)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect', zorder=1)
            
            # Plot climatology line (using first model - should be same for all)
            clim = float(list_ds_hw_metrics_aggregated[0].sel(severity=severity).isel(week_lead_time=idx)['climatology_frequency'].values)
            ax.axhline(clim, color='blue', linestyle=':', linewidth=1, alpha=0.75, 
                      label=f'Climatology ({clim:.2f})', zorder=1)
            
            # Collect all counts to normalize marker sizes across all models in this subplot
            all_counts = []
            for model_idx, (metrics_by_severity, reliability_data, model_name) in enumerate(
                zip(list_ds_hw_metrics_aggregated, list_reliability_data, list_model_names)):
                rel_data = reliability_data[severity][week]
                valid_mask = ~np.isnan(rel_data['observed_freq'])
                if valid_mask.sum() > 0:
                    all_counts.extend(np.array(rel_data['counts'])[valid_mask])
            
            # Determine size scaling based on counts
            if len(all_counts) > 0:
                max_count = np.max(all_counts)
                min_count = np.min(all_counts)
                # Scale marker sizes: minimum 20, maximum 200
                size_min, size_max = 50, 1000
            
            # Plot reliability curves for each model
            for model_idx, (metrics_by_severity, reliability_data, model_name) in enumerate(
                zip(list_ds_hw_metrics_aggregated, list_reliability_data, list_model_names)):
                
                rel_data = reliability_data[severity][week]
                
                # Plot reliability curve
                valid_mask = ~np.isnan(rel_data['observed_freq'])
                if valid_mask.sum() > 0:
                    forecast_probs = np.array(rel_data['forecast_prob'])[valid_mask]
                    observed_freqs = np.array(rel_data['observed_freq'])[valid_mask]
                    counts = np.array(rel_data['counts'])[valid_mask]
                    
                    # Calculate marker sizes based on sample counts
                    if max_count > min_count:
                        # Linear scaling
                        marker_sizes = size_min + (counts - min_count) / (max_count - min_count) * (size_max - size_min)
                    else:
                        # All same count
                        marker_sizes = np.ones_like(counts) * ((size_min + size_max) / 2)
                    
                    scatter = ax.scatter(
                        forecast_probs, 
                        observed_freqs,
                        s=marker_sizes, 
                        alpha=0.7, 
                        c=model_colors[model_idx],
                        marker='o',  # Circles for all models
                        edgecolors='black', 
                        linewidth=1, 
                        zorder=3,
                        label=model_name
                    )
                    
                    # Connect points with line
                    ax.plot(
                        forecast_probs, 
                        observed_freqs,
                        '-', 
                        linewidth=1.5, 
                        color=model_colors[model_idx], 
                        alpha=0.6,
                        zorder=2
                    )
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('Forecast Probability', fontsize=14)
            ax.set_ylabel('Observed Frequency', fontsize=14)
            
            # Simplified title with just week number
            ax.set_title(f'Week {week_num} Lead', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            ax.set_aspect('equal')
        
        # Add shared legend below all subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center',
            ncol=min(len(labels), 5),
            fontsize=14,
            frameon=True,
            markerscale=0.5
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        output_file = os.path.join(output_dir, f'reliability_diagram_{severity}_all_models.png')
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()


def plot_bss_spatial_maps(list_metrics_spatial, list_model_names, europe_mask, lat, lon, output_dir):
    """
    Create spatial maps of Brier Skill Score over Europe
    One figure per severity, with 3 rows (one per model) and 4 columns (Week 1, 3, 5, 7)
    
    Note: White patches indicate grid points with no climatological variability
    (p=0 or p=1), where BSS is undefined (bs_clim=0). This is expected for:
    - Rare events at longer lead times
    - More extreme heat wave definitions
    """
    print("\nCreating spatial BSS comparison maps (one per severity)...")
    
    # Convert lon to -180 to 180 for plotting
    lon_plot = np.where(lon > 180, lon - 360, lon)
    
    proj = ccrs.PlateCarree()
    
    lead_weeks = [1, 2, 3, 4]
    severities = ['moderate', 'severe', 'extreme']
    
    # Create one figure per severity
    for severity in severities:
        fig = plt.figure(figsize=(22, 15))
        
        # Store axes and contour plots for each row to add colorbar later
        row_axes = {0: [], 1: [], 2: []}
        row_cf = {0: None, 1: None, 2: None}
        
        # Loop over models (rows) and lead times (columns)
        for model_idx, (metrics_spatial, model_name) in enumerate(zip(list_metrics_spatial, list_model_names)):
            
            for lead_idx, week_lead in enumerate(lead_weeks):
                ax = fig.add_subplot(3, 4, model_idx * 4 + lead_idx + 1, projection=proj)
                row_axes[model_idx].append(ax)
                
                # Get BSS data for this severity and lead time
                # Assuming the dataset has dimensions: severity, week_lead_time, values
                bss_data = metrics_spatial['brier_skill_score'].sel(severity=severity, week_lead_time=week_lead)
                bss = bss_data.values
                
                # Diagnostic: count valid values before masking
                n_total = len(bss)
                n_finite = np.isfinite(bss).sum()
                n_europe = europe_mask.sum()
                
                # Only plot Europe points with finite BSS values
                mask = np.isfinite(lon_plot) & np.isfinite(lat) & np.isfinite(bss) & europe_mask
                n_plot_points = mask.sum()
                
                # Enhanced diagnostics
                if model_idx == 0 and lead_idx == 0:  # Print diagnostics for first panel
                    print(f"  [{severity.capitalize()}, {model_name}, Week {week_lead}] Diagnostics:")
                    print(f"    Total grid points: {n_total}")
                    print(f"    Finite BSS values: {n_finite}")
                    print(f"    Europe mask points: {n_europe}")
                    print(f"    Points to plot: {n_plot_points}")
                    if n_plot_points > 0:
                        print(f"    BSS range: [{np.nanmin(bss[mask]):.3f}, {np.nanmax(bss[mask]):.3f}]")
                        print(f"    BSS mean (Europe): {np.nanmean(bss[europe_mask]):.3f}")
                
                vmin, vmax = -0.5, 1.0
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                
                if n_plot_points > 0:
                    # Use tricontourf with RdBu (red=negative/worse, blue=positive/better, colorblind-friendly)
                    cf = ax.tricontourf(lon_plot[mask], lat[mask], bss[mask],
                                       levels=np.linspace(vmin, vmax, 31),
                                       cmap='RdBu', norm=norm, transform=proj, extend='both')
                    # Store the contour plot for colorbar
                    if row_cf[model_idx] is None:
                        row_cf[model_idx] = cf
                
                ax.coastlines(linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
                ax.set_extent([-10, 40, 35, 70], crs=proj)
                
                # Calculate mean BSS over Europe
                bss_mean = float(np.nanmean(bss[europe_mask]))
                
                # Title: just week and BSS value
                title = f'Week {week_lead}\nMean BSS={bss_mean:.3f}'
                ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Apply tight_layout first to position subplots correctly
        plt.suptitle(f'Brier Skill Score: {severity.capitalize()} Heat Waves - All Models\n' +
                    '(Blue: skillful forecasts, Red: worse than climatology)',
                    fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0.05, 0, 0.98, 0.96])
        
        # Now add colorbars for each row (spanning all 4 subplots) - after tight_layout
        for model_idx in range(3):
            if row_cf[model_idx] is not None:
                # Create colorbar that spans the row
                cbar = fig.colorbar(row_cf[model_idx], ax=row_axes[model_idx], 
                                   orientation='vertical', pad=0.01, shrink=.75, aspect=20)
                cbar.set_label('BSS', fontsize=14, fontweight='bold')
                cbar.ax.axhline(0, color='black', linewidth=1.5, alpha=0.7)
        
        # Add model names on the left side using subplot positions
        for model_idx, model_name in enumerate(list_model_names):
            # Get the position of the first subplot in this row
            ax_pos = row_axes[model_idx][0].get_position()
            # Place text at the vertical center of the row, to the left of the subplots
            y_center = (ax_pos.y0 + ax_pos.y1) / 2
            fig.text(0.01, y_center, model_name, fontsize=14, fontweight='bold', 
                    rotation=90, va='center', ha='center', transform=fig.transFigure)
        
        output_file = os.path.join(output_dir, f'bss_spatial_{severity}_all_models.png')
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()
    



# ============================================================================
# ROC PLOTTING
# ============================================================================
def plot_roc_curves_by_severity(ref_roc, weekly_roc, daily_roc, output_dir): 

    """
    Plot ROC curves for each heatwave severity.

    Parameters
    ----------
    ref_roc : xr.DataArray
        ROC DataArray for reference model.
    weekly_roc : xr.DataArray
        ROC DataArray for weekly finetuned model.
    daily_roc : xr.DataArray
        ROC DataArray for daily finetuned model.
    save_dir : str, optional
        Directory to save figures.
    """

    model_data = {
        "Reference": ref_roc,
        "Weekly FT": weekly_roc,
        "Daily FT": daily_roc,
    }
    
    model_colors = {
        'Reference': '#1f77b4',      # blue
        'Weekly FT': '#2ca02c',    # green
        'Daily FT':  '#E69F00', # orange
    }

    severities = ref_roc.severity.values

    for severity in severities:

        fig, axes = plt.subplots(
            2, 4,
            figsize=(18, 9),
            sharex=True,
            sharey=True
        )

        axes = axes.flatten()

        for lead in range(1, 9):

            ax = axes[lead - 1]
            auc_text = []

            for model_name, roc_da in model_data.items():

                roc_lead = roc_da.sel(
                    severity=severity,
                    week_lead_time=lead
                )

                fpr = roc_lead.sel(metric="false positive rate")
                tpr = roc_lead.sel(metric="true positive rate")

                auc = float(
                    roc_lead.sel(metric="area under curve")
                    .isel(probability_bin=0)
                )
                auc_text.append(f"AUC {model_name}={auc:.2f}\n")

                ax.plot(fpr, tpr, color=model_colors[model_name], lw=2, label=f"{model_name}")
                

            # No-skill line
            ax.plot([0, 1],[0, 1],"k--",alpha=0.5)
            ax.set_title(f"Week {lead}", fontsize=14)

            ax.text(
               0.97, 0.03,
                "".join(auc_text),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=12,
                bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none"
                )
            )
            ax.grid(True, alpha=0.3)

            if lead > 4:
                ax.set_xlabel("False Positive Rate")

            if lead in [1, 5]:
                ax.set_ylabel("True Positive Rate")

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(handles,labels,loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f"ROC Curves: {severity.capitalize()} Heat Waves - All Models (NEW THRESHOLDS)",
            fontsize=18,
            y=0.98
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        output_file = os.path.join(output_dir, f'roc_curves_{severity}.png')
        print(f"  Saved: {output_file}")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')




# ============================================================================
# MAIN PLOTTING EXECUTION
# ============================================================================


init_date = "2010-06-11T12:00:00.000000000"

# Variables to plot - raw fields
variables_to_plot_raw = ['2t', 'tp', 'msl', 'z_500']

# Variables to plot - anomalies
variables_to_plot_anomalies = ['2t', 'tp', 'msl', 'z_500']

# Leadtimes in weeks (all 8 weeks)
leadtimes = [1, 2, 3, 4, 5, 6, 7, 8]

# Organize datasets in a dictionary (for raw fields and checking member dimensions)
datasets_dict = {
    'obs_weekly_ds': obs_weekly_ds,
    'ref_weekly_ds': ref_weekly_ds,
    'weekly_finetuned_ds': weekly_finetuned_ds,
    'daily_finetuned_ds': daily_finetuned_ds
}

# Organize anomaly datasets in a dictionary
anomalies_dict = {
    'obs_anomalies_ds': obs_anomalies_ds,
    'ref_anomalies_ds': ref_anomalies_ds,
    'weekly_finetuned_anomalies_ds': weekly_finetuned_anomalies_ds,
    'daily_finetuned_anomalies_ds': daily_finetuned_anomalies_ds
}

anomalies_robust_dict = {
    'obs_anomalies_ds': obs_anomalies_ROBUST,
    'ref_anomalies_ds': ref_anomalies_ROBUST,
    'weekly_finetuned_anomalies_ds': weekly_finetuned_anomalies_ROBUST,
    'daily_finetuned_anomalies_ds': daily_finetuned_anomalies_ROBUST
}

# Find best ensemble members (closest to ground truth for 2t)
best_members_robust = find_best_members(datasets_dict, anomalies_robust_dict, init_date, region_mask)
# Map best members from anomaly dataset names to raw dataset names
best_members_raw = {
    'ref_weekly_ds': best_members_robust.get('ref_anomalies_ds'),
    'weekly_finetuned_ds': best_members_robust.get('weekly_finetuned_anomalies_ds'),
    'daily_finetuned_ds': best_members_robust.get('daily_finetuned_anomalies_ds')
}

# Create plots for raw fields
print(f"\nCreating raw field plots for init_date: {init_date}")
print(f"Region: lat [{lat.where(region_mask).min().values:.1f}, {lat.where(region_mask).max().values:.1f}], "
      f"lon [{lon.where(region_mask).min().values:.1f}, {lon.where(region_mask).max().values:.1f}]")
print("="*80)

for var in variables_to_plot_raw:
    print(f"\nProcessing raw field variable: {var}")
    try:
        plot_variable_grid(var, datasets_dict, init_date, leadtimes, region_mask, lat, lon, OUTPUT_DIR, best_members_raw)
    except Exception as e:
        print(f"Error plotting {var}: {str(e)}")
        import traceback
        traceback.print_exc()

# Create plots for anomalies
print(f"\n\nCreating anomaly plots for init_date: {init_date}")
print("="*80)

for var in variables_to_plot_anomalies:
    print(f"\nProcessing anomaly variable: {var}")
    try:
        plot_variable_grid_anomalies(var, anomalies_dict, init_date, leadtimes, region_mask, lat, lon, OUTPUT_DIR, best_members)
    except Exception as e:
        print(f"Error plotting {var}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("All plots completed!")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# CLIMATOLOGY PLOTS
# ============================================================================

# Create climatology dictionary
climatology_dict = {
    'obs_climatology_ds': obs_climatology_ds,
    'ref_climatology_ds': ref_climatology_ds,
    'weekly_finetuned_climatology_ds': weekly_finetuned_climatology_ds,
    'daily_finetuned_climatology_ds': daily_finetuned_climatology_ds
}

# Map best members to climatology dataset names
best_members_climatology = {
    'ref_climatology_ds': best_members_robust.get('ref_anomalies_ds'),
    'weekly_finetuned_climatology_ds': best_members_robust.get('weekly_finetuned_anomalies_ds'),
    'daily_finetuned_climatology_ds': best_members_robust.get('daily_finetuned_anomalies_ds')
}

print("\n" + "="*80)
print("Creating climatology plots...")
print("="*80)


plot_climatology_grid("2t", climatology_dict, init_date, leadtimes, region_mask, lat, lon, OUTPUT_DIR, best_members_climatology)


print("\n" + "="*80)
print("All climatology plots completed!")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# ENSEMBLE PLUME PLOTS
# ============================================================================

print("\n" + "="*80)
print("Creating ensemble plume plots (raw fields)...")
print("="*80)

for var in variables_to_plot_raw:
    print(f"\nProcessing ensemble plumes for variable: {var}")
    try:
        plot_ensemble_plumes(var, datasets_dict, init_date, region_mask, OUTPUT_DIR, best_members_raw)
        plot_ensemble_plumes_anomalies(var, anomalies_dict, init_date, region_mask, OUTPUT_DIR, best_members)
    except Exception as e:
        print(f"Error plotting ensemble plumes for {var}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("All ensemble plume plots completed!")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# ERROR MAPS
# ============================================================================

print("\n" + "="*80)
print("Creating error maps (Model - Observation) for raw fields...")
print("="*80)

for var in variables_to_plot_raw:
    print(f"\nProcessing error maps for raw field variable: {var}")
    try:
        plot_error_maps(var, datasets_dict, init_date, leadtimes, region_mask, lat, lon, OUTPUT_DIR, best_members_raw)
    except Exception as e:
        print(f"Error plotting error maps for {var}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("Creating error maps (Model - Observation) for anomalies...")
print("="*80)

for var in variables_to_plot_anomalies:
    print(f"\nProcessing error maps for anomaly variable: {var}")
    try:
        plot_error_maps_anomalies(var, anomalies_dict, init_date, leadtimes, region_mask, lat, lon, OUTPUT_DIR, best_members)
    except Exception as e:
        print(f"Error plotting error maps for {var}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("All error maps completed!")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# PROBABILISTIC METRIC PLOTS
# ============================================================================
print("\n" + "="*80)
print("Creating probabilistic metric plots...")
print("="*80)

list_model_names = ['Reference Model', 'Weekly Finetuned', "Daily Finetuned"]
list_reliability_data_new = [reliability_data_ref_new, reliability_data_weekly_finetuned_new, reliability_data_daily_finetuned_new] 
plot_probabilistic_metrics(list_ds_hw_metrics_aggregated_new, list_reliability_data_new, list_model_names, OUTPUT_DIR)
print("\n" + "="*80)
print("All probabilistic metric plots completed!")
print(f"Output directory: {OUTPUT_DIR}")

#europe mask
lat = obs_weekly_ds.latitude.values
lon = obs_weekly_ds.longitude.values

lon_adjusted = np.where(lon <= 180, lon, lon - 360)
region_lat_mask = (lat >= 20) & (lat <= 80)
region_lon_mask = (lon_adjusted >= -60) & (lon_adjusted <= 90)
region_mask = region_lat_mask & region_lon_mask
europe_lat_mask = (lat >= 35) & (lat <= 70)
europe_lon_mask = (lon_adjusted >= -10) & (lon_adjusted <= 40)
europe_mask = europe_lat_mask & europe_lon_mask

print("\n" + "="*80)
print("Creating probabilistic spatial metric plots...")
print("="*80)

plot_bss_spatial_maps(list_ds_hw_metrics_spatial_new, list_model_names, europe_mask, lat, lon, OUTPUT_DIR)
print("\n" + "="*80)
print("All probabilistic spatial metric plots completed!")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
#Roc plots
# ============================================================================

plot_roc_curves_by_severity(ds_hw_roc_ref_new , ds_hw_roc_weekly_finetuned_new, ds_hw_roc_daily_finetuned_new, OUTPUT_DIR)


# ============================================================================
#Combine plots of anomalies and geopotential height
# ============================================================================
print("\n" + "="*80)
print("Creating combined anomaly plots...")
print("="*80)

plot_combined_anomaly_fields(
    base_var='2t',           # Temperature anomalies as filled contours
    overlay_var='z_500',     # Geopotential height as contour lines
    datasets_dict=anomalies_robust_dict,
    init_date=init_date,
    leadtimes=leadtimes,
    region_mask=region_mask,
    lat_flat=lat,
    lon_flat=lon,
    output_dir=OUTPUT_DIR,
    best_members=best_members_robust,
    overlay_levels=7)


