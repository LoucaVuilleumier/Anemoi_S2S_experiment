import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm, Normalize
import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
WEEKLY_MEANS_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed"
OBS_WEEKLY_FILE = os.path.join(WEEKLY_MEANS_DIR, "observations_weekly.nc")
OBS_WEEKLY_FROM_DAILY_FILE = os.path.join(WEEKLY_MEANS_DIR, "observations_weekly_from_daily.nc")
REF_WEEKLY_FILE = os.path.join(WEEKLY_MEANS_DIR, "reference_model_weekly.nc")
DAILY_WEEKLY_FILE = os.path.join(WEEKLY_MEANS_DIR, "daily_finetuned_model_weekly.nc")
WEEKLY_FINETUNED_FILE = os.path.join(WEEKLY_MEANS_DIR, "Weekly_Means_18k_lr_0.625e-6_weekly.nc")

OUTPUT_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/heat_wave_simple_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variables for comprehensive physics analysis
SURFACE_VARS = ['2t', '2d', 'msl', 'sp', 'skt', 'tp', '10u', '10v']
UPPER_VARS = ['t_500', 't_700', 't_850', 't_925', 'q_500', 'q_700', 'q_850', 'q_925',"z_500", "z_700", "z_850", "z_925"]
ALL_VARS = SURFACE_VARS + UPPER_VARS

# Load datasets
obs_weekly_ds = xr.open_dataset(OBS_WEEKLY_FILE)
obs_weekly_from_daily_ds = xr.open_dataset(OBS_WEEKLY_FROM_DAILY_FILE)
ref_weekly_ds = xr.open_dataset(REF_WEEKLY_FILE)
daily_weekly_ds = xr.open_dataset(DAILY_WEEKLY_FILE)
weekly_finetuned_ds = xr.open_dataset(WEEKLY_FINETUNED_FILE)

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

init_date = "2007-07-01T12:00:00.000000000"

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_variable_grid(var_name, datasets_dict, init_date, leadtimes, region_mask, lat_flat, lon_flat, output_dir):
    """
    Create a 4x4 grid plot for a single variable.
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., '2t', 'sp', 'tp', 'msl', 't_850')
    datasets_dict : dict
        Dictionary with keys as dataset names and values as xarray datasets
    init_date : str
        Initialization date to select
    leadtimes : list
        List of leadtimes (in weeks) to plot [1, 3, 5, 7]
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    lat_flat, lon_flat : xarray.DataArray
        Flattened latitude and longitude arrays
    output_dir : str
        Directory to save the output figure
    """
    
    fig = plt.figure(figsize=(20, 16))
    
    # Dataset order for rows
    dataset_names = ['obs_weekly_ds', 'ref_weekly_ds', 'daily_weekly_ds', 'weekly_finetuned_ds']
    dataset_labels = ['Observations', 'Reference Model', 'Daily Finetuned', 'Weekly Finetuned']
    
    # Define projection for geographical plots
    projection = ccrs.PlateCarree()
    
    # Convert to numpy arrays for plotting
    lat_np = lat_flat.values
    lon_np = lon_flat.values
    region_mask_np = region_mask.values
    
    # Set map extent to supervisor's requested domain: 60W to 90E, 20N to 80N
    map_extent = [-60, 90, 20, 80]
    
    # Calculate single colorbar range for entire plot (all datasets and all leadtimes)
    all_values_global = []
    for leadtime_week in leadtimes:
        for ds_name in dataset_names:
            try:
                ds = datasets_dict[ds_name]
                data_sel = ds[var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
                if 'member' in data_sel.dims:
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
    
    # Create subplots
    for row_idx, (ds_name, ds_label) in enumerate(zip(dataset_names, dataset_labels)):
        for col_idx, leadtime_week in enumerate(leadtimes):
            
            subplot_idx = row_idx * 4 + col_idx + 1
            ax = fig.add_subplot(4, 4, subplot_idx, projection=projection)
            
            try:
                # Get the dataset
                ds = datasets_dict[ds_name]
                
                # Select data for the specific init_date and week_lead_time
                data_sel = ds[var_name].sel(init_date=init_date, week_lead_time=leadtime_week)
                
                # Handle member dimension if present (take ensemble mean)
                if 'member' in data_sel.dims:
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
                           ha='center', va='center', fontsize=8)
                
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
                if row_idx < 3:
                    gl.bottom_labels = False
                
                # Add title
                if row_idx == 0:
                    ax.set_title(f'Leadtime: {leadtime_week} week{"s" if leadtime_week > 1 else ""}', 
                                fontsize=12, fontweight='bold')
                
                # Add row labels
                if col_idx == 0:
                    ax.text(-0.15, 0.5, ds_label, 
                           transform=ax.transAxes,
                           fontsize=12, fontweight='bold',
                           rotation=90, va='center', ha='center')
                
            except Exception as e:
                # If there's an error (e.g., variable not found), add text to subplot
                ax.text(0.5, 0.5, f'Data not available\n{str(e)[:50]}', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=8)
                ax.set_extent(map_extent, crs=projection)
                ax.coastlines(resolution='50m', linewidth=0.5)
    
    # Add main title
    fig.suptitle(f'Variable: {var_name} - Init Date: {init_date}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_grid_plot_{init_date.split("T")[0]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for {var_name}: {output_file}")


def plot_error_maps(var_name, datasets_dict, init_date, leadtimes, region_mask, lat_flat, lon_flat, output_dir):
    """
    Create error maps showing (model - observation) for different models.
    
    Parameters:
    -----------
    var_name : str
        Variable name to plot (e.g., '2t', 'sp', 'tp', 'msl', 't_850')
    datasets_dict : dict
        Dictionary with keys as dataset names and values as xarray datasets
    init_date : str
        Initialization date to select
    leadtimes : list
        List of leadtimes (in weeks) to plot [1, 3, 5, 7]
    region_mask : xarray.DataArray
        Boolean mask for the region of interest (flattened)
    lat_flat, lon_flat : xarray.DataArray
        Flattened latitude and longitude arrays
    output_dir : str
        Directory to save the output figure
    """
    
    fig = plt.figure(figsize=(20, 12))
    
    # Model order for rows (skip observations)
    model_names = ['ref_weekly_ds', 'daily_weekly_ds', 'weekly_finetuned_ds']
    model_labels = ['Reference Model', 'Daily Finetuned', 'Weekly Finetuned']
    
    # Define projection for geographical plots
    projection = ccrs.PlateCarree()
    
    # Convert to numpy arrays for plotting
    lat_np = lat_flat.values
    lon_np = lon_flat.values
    region_mask_np = region_mask.values
    
    # Set map extent to supervisor's requested domain: 60W to 90E, 20N to 80N
    map_extent = [-60, 90, 20, 80]
    
    # Calculate single colorbar range for entire error plot (all models and leadtimes)
    all_errors_global = []
    for leadtime_week in leadtimes:
        for model_name in model_names:
            try:
                # Select appropriate observation dataset based on model and variable
                if var_name == 'tp':
                    obs_ds_key = 'obs_weekly_ds'
                elif model_name == 'daily_weekly_ds':
                    obs_ds_key = 'obs_weekly_from_daily_ds'
                else:
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
            except:
                pass
    
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
    
    # Create subplots
    for row_idx, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
        for col_idx, leadtime_week in enumerate(leadtimes):
            
            subplot_idx = row_idx * 4 + col_idx + 1
            ax = fig.add_subplot(3, 4, subplot_idx, projection=projection)
            
            try:
                # Select appropriate observation dataset based on model and variable
                if var_name == 'tp':
                    obs_ds_key = 'obs_weekly_ds'
                elif model_name == 'daily_weekly_ds':
                    obs_ds_key = 'obs_weekly_from_daily_ds'
                else:
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
                           ha='center', va='center', fontsize=8)
                
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
                if row_idx < 2:
                    gl.bottom_labels = False
                
                # Add title with mean error
                if row_idx == 0:
                    ax.set_title(f'Leadtime: {leadtime_week} week{"s" if leadtime_week > 1 else ""}\nMean: {mean_error:.4f}', 
                                fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f'Mean: {mean_error:.4f}', fontsize=10)
                
                # Add row labels
                if col_idx == 0:
                    ax.text(-0.15, 0.5, model_label, 
                           transform=ax.transAxes,
                           fontsize=12, fontweight='bold',
                           rotation=90, va='center', ha='center')
                
            except Exception as e:
                # If there's an error, add text to subplot
                ax.text(0.5, 0.5, f'Data not available\n{str(e)[:50]}', 
                       transform=ax.transAxes,
                       ha='center', va='center', fontsize=8)
                ax.set_extent(map_extent, crs=projection)
                ax.coastlines(resolution='50m', linewidth=0.5)
    
    # Add main title
    fig.suptitle(f'Error (Model - Obs): {var_name} - Init Date: {init_date}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_error_map_{init_date.split("T")[0]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error map for {var_name}: {output_file}")


def plot_ensemble_plumes(var_name, datasets_dict, init_date, region_mask, output_dir):
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
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Model configuration
    model_names = ['ref_weekly_ds', 'daily_weekly_ds', 'weekly_finetuned_ds']
    model_labels = ['Reference Model', 'Daily Finetuned', 'Weekly Finetuned']
    model_colors = {
        'ref_weekly_ds': '#1f77b4',      # blue
        'daily_weekly_ds': '#2ca02c',    # green
        'weekly_finetuned_ds': '#d62728' # red
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
    
    # Get observation data from both sources
    obs_data_weekly = datasets_dict['obs_weekly_ds'][var_name].sel(init_date=init_date)
    # Convert temperature from Kelvin to Celsius
    if var_name == '2t':
        obs_data_weekly = obs_data_weekly - 273.15
    obs_regional_mean_weekly = obs_data_weekly.where(region_mask).mean(dim='values')
    all_values_global.append(obs_regional_mean_weekly.values)
    
    obs_data_daily = datasets_dict['obs_weekly_from_daily_ds'][var_name].sel(init_date=init_date)
    # Convert temperature from Kelvin to Celsius
    if var_name == '2t':
        obs_data_daily = obs_data_daily - 273.15
    obs_regional_mean_daily = obs_data_daily.where(region_mask).mean(dim='values')
    all_values_global.append(obs_regional_mean_daily.values)
    
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
            
        try:
            # Select appropriate observation dataset based on model and variable
            # For tp (accumulated), use obs_weekly_ds for all models
            # For other variables, use obs_weekly_from_daily_ds for Daily Finetuned
            if var_name == 'tp':
                obs_ds_key = 'obs_weekly_ds'
            elif model_name == 'daily_weekly_ds':
                obs_ds_key = 'obs_weekly_from_daily_ds'
            else:
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
            
            # Plot observations (black line)
            if var_name == 'tp':
                obs_label = 'Observation (weekly)'
            elif model_name == 'daily_weekly_ds':
                obs_label = 'Observation (from daily)'
            else:
                obs_label = 'Observation (weekly)'
            ax.plot(weeks, obs_regional_mean.values, 
                   'k-', linewidth=2, marker='o', 
                   markersize=4, label=obs_label)
            
            # Formatting
            ax.set_ylim(y_limits)
            ax.grid(True, alpha=0.3)
            
            # Labels
            ax.set_xlabel('Week Lead Time', fontsize=11)
            if col_idx == 0:  # First subplot
                ax.set_ylabel(f'{var_units.get(var_name, "")}', fontsize=11)
            
            # Model name as title
            ax.set_title(model_label, fontsize=12, fontweight='bold')
            
            # Legend on first subplot
            if col_idx == 0:
                ax.legend(loc='best', fontsize=9)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Data not available\n{str(e)[:40]}', 
                   transform=ax.transAxes,
                   ha='center', va='center', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Overall title with init_date
    init_date_str = pd.to_datetime(init_date).strftime("%Y-%m-%d")
    fig.suptitle(f'Ensemble Plumes: {var_name} - Init Date: {init_date_str}\n(Regional mean: 60W-90E, 20N-80N)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save figure
    output_file = os.path.join(output_dir, f'{var_name}_ensemble_plumes.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ensemble plume plot for {var_name}: {output_file}")


# ============================================================================
# MAIN PLOTTING EXECUTION
# ============================================================================

# Variables to plot
variables_to_plot = ['2t', 'sp', 'tp', 'msl', 't_850', 'z_500']

# Leadtimes in weeks
leadtimes = [1, 3, 5, 7]

# Organize datasets in a dictionary
datasets_dict = {
    'obs_weekly_ds': obs_weekly_ds,
    'obs_weekly_from_daily_ds': obs_weekly_from_daily_ds,
    'ref_weekly_ds': ref_weekly_ds,
    'daily_weekly_ds': daily_weekly_ds,
    'weekly_finetuned_ds': weekly_finetuned_ds
}

# Create plots for each variable
print(f"\nCreating plots for init_date: {init_date}")
print(f"Region: lat [{lat.where(region_mask).min().values:.1f}, {lat.where(region_mask).max().values:.1f}], "
      f"lon [{lon.where(region_mask).min().values:.1f}, {lon.where(region_mask).max().values:.1f}]")
print("="*80)

for var in variables_to_plot:
    print(f"\nProcessing variable: {var}")
    try:
        plot_variable_grid(var, datasets_dict, init_date, leadtimes, region_mask, lat, lon, OUTPUT_DIR)
    except Exception as e:
        print(f"Error plotting {var}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("All plots completed!")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# ENSEMBLE PLUME PLOTS
# ============================================================================

print("\n" + "="*80)
print("Creating ensemble plume plots...")
print("="*80)

for var in variables_to_plot:
    print(f"\nProcessing ensemble plumes for variable: {var}")
    try:
        plot_ensemble_plumes(var, datasets_dict, init_date, region_mask, OUTPUT_DIR)
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
print("Creating error maps (Model - Observation)...")
print("="*80)

for var in variables_to_plot:
    print(f"\nProcessing error maps for variable: {var}")
    try:
        plot_error_maps(var, datasets_dict, init_date, leadtimes, region_mask, lat, lon, OUTPUT_DIR)
    except Exception as e:
        print(f"Error plotting error maps for {var}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("All error maps completed!")
print(f"Output directory: {OUTPUT_DIR}")