
import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
import warnings
from dask.diagnostics import ProgressBar
import dask
warnings.filterwarnings('ignore')

# Configure dask for parallel processing
# Use environment variable if available, otherwise default to 8
n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
dask.config.set(scheduler='threads', num_workers=n_workers)
print(f"Dask configured with {n_workers} workers")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
RAW_OBS_PATH = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-6h-v1-for-single-v2.zarr"
FCST_DIR = "/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/AIFS"
OUTPUT_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/weekly_means_physics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variables for comprehensive physics analysis
SURFACE_VARS = ['2t', '2d', 'msl', 'sp', 'skt', 'tp', '10u', '10v']
UPPER_VARS = ['t_500', 't_700', 't_850', 't_925', 'q_500', 'q_700', 'q_850', 'q_925']
ALL_VARS = SURFACE_VARS + UPPER_VARS

print("="*80)
print("PREPROCESSING HEAT WAVE PHYSICS DATA")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nVariables to process:")
print(f"  Surface: {SURFACE_VARS}")
print(f"  Upper levels: {UPPER_VARS}")
print(f"  Total: {len(ALL_VARS)} variables")
print()

# ============================================================================
# FORECAST PROCESSING (Easier - already in NetCDF)
# ============================================================================

def process_forecasts():

    print("\n" + "="*80)
    print("PROCESSING FORECASTS")
    print("="*80)
    
    # Find all initialization dates
    paths_init = sorted(glob.glob(f"{FCST_DIR}/aifs-subs-pretrain-*"))
    print(f"\nFound {len(paths_init)} initialization dates")
    
    reforecast_datasets = []
    init_dates = []
    
    for path_init in paths_init:
        init_date_str = path_init.split('-')[-3:]
        init_date_str = '-'.join(init_date_str)
        print(f"\nProcessing {init_date_str}...")
        
        # Get all member paths
        paths_member = sorted(glob.glob(f"{path_init}/*.nc"))
        
        # Filter out empty or corrupted files
        valid_paths = [p for p in paths_member if os.path.getsize(p) > 0]
        if len(valid_paths) < len(paths_member):
            skipped = len(paths_member) - len(valid_paths)
            skipped_files = [os.path.basename(p) for p in paths_member if os.path.getsize(p) == 0]
            print(f"  ⚠️  WARNING: Skipping {skipped} empty/corrupted file(s): {', '.join(skipped_files)}")
        
        if not valid_paths:
            print(f"  ❌ ERROR: No valid files found, skipping this initialization date")
            continue
            
        print(f"  Loading {len(valid_paths)} ensemble members...")
        
        # Load all members with chunking for parallel processing
        ds_init = xr.open_mfdataset(
            valid_paths,
            combine="nested",
            concat_dim="member",
            parallel=True,
            chunks={'time': 100, 'values': 10000},
            engine='netcdf4'
        )
        
        # Check which variables are available
        available_vars = [v for v in ALL_VARS if v in ds_init.data_vars]
        print(f"  Found {len(available_vars)}/{len(ALL_VARS)} variables")
        missing = [v for v in ALL_VARS if v not in ds_init.data_vars]
        if missing:
            print(f"  Missing: {missing}")
        
        # Select only available variables
        ds_init = ds_init[available_vars]
        
        # Extract initialization date
        init_date = ds_init.time.values[0].astype('datetime64[ns]')
        init_dates.append(init_date)
        
        # Store absolute time
        ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
        
        # Replace time with relative lead_time (in days)
        lead_times = (ds_init.time.values - init_date) / np.timedelta64(1, 'D')
        ds_init = ds_init.assign_coords(time=lead_times).rename({'time': 'lead_time'})
        
        # Remove first timestep (initialization)
        ds_init = ds_init.isel(lead_time=slice(1, None))
        
        print(f"  Computing weekly means (7-day averages)...")
        # Compute weekly means
        ds_init_weekly = ds_init.coarsen(lead_time=7, boundary='trim').mean()
        ds_init_weekly = ds_init_weekly.rename({"lead_time": "week_lead_time"})
        ds_init_weekly = ds_init_weekly.assign_coords(week_lead_time=np.arange(1, 9))
        
        # Add init_date as coordinate
        ds_init_weekly = ds_init_weekly.expand_dims(init_date=[init_date])
        
        # Compute immediately to free memory
        print(f"  Computing results...")
        with ProgressBar():
            ds_init_weekly = ds_init_weekly.compute()
        
        reforecast_datasets.append(ds_init_weekly)
    
    # Concatenate all init dates
    print("\nConcatenating all initialization dates...")
    ds_fcst_weekly = xr.concat(reforecast_datasets, dim='init_date')
    
    print(f"\nForecast weekly dataset shape: {dict(ds_fcst_weekly.dims)}")
    print(f"Variables: {list(ds_fcst_weekly.data_vars)}")
    
    # Save to NetCDF
    output_file = os.path.join(OUTPUT_DIR, "Forecasts_weekly_AIFS_physics.nc")
    print(f"\nSaving to: {output_file}")
    print("  (This may take a few minutes...)")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Actually save the file with compression
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_fcst_weekly.data_vars}
    ds_fcst_weekly.to_netcdf(output_file, encoding=encoding)
    
    print(f"Saved forecast weekly means ({os.path.getsize(output_file) / 1e9:.2f} GB)")
    
    return init_dates, available_vars


# ============================================================================
# OBSERVATION PROCESSING (Memory-intensive - process one init date at a time)
# ============================================================================

def process_observations_one_init(obs_raw, init_date, available_vars):
    """
    Process observations for a single initialization date
    This reduces memory usage by processing one init date at a time
    """
    times = obs_raw.time.values
    
    print(f"\n  Processing init date: {pd.Timestamp(init_date).strftime('%Y-%m-%d')}")
    
    # Find time index
    t0 = np.where(times == init_date)[0][0]
    # 56 days = 8 weeks, 4 timesteps per day (6-hourly)
    t_end = t0 + 56 * 4
    
    print(f"    Extracting time slice ({t0} to {t_end})...")
    ds_init = obs_raw.isel(time=slice(t0, t_end + 1))
    ds_init = ds_init.squeeze("ensemble")
    
    # Store absolute time
    ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
    
    # Process each variable according to its nature
    var_datasets = []
    
    # Get reference time coordinate from first non-tp variable
    ref_var = [v for v in available_vars if v != 'tp'][0]
    print(f"    Using {ref_var} as reference for daily timesteps...")
    ds_ref = ds_init['data'].sel(variable=ref_var)
    ds_ref_12h = ds_ref.sel(time=ds_ref.time.dt.hour == 12)
    ds_ref_daily = ds_ref_12h.isel(time=slice(1, None))
    ref_times = ds_ref_daily.time.values
    
    print(f"    Processing {len(available_vars)} variables to daily means...")
    for var in available_vars:
        ds_var = ds_init['data'].sel(variable=var)
        
        if var == 'tp':
            # Total precipitation: daily accumulation
            ds_var_daily = ds_var.resample(time="1D").sum()
            # Align to ref_times
            ds_var_daily = ds_var_daily.isel(time=slice(1, None))
            ds_var_daily = ds_var_daily.assign_coords(time=ref_times)
        else:
            # Other variables: select 12:00 UTC values
            ds_var_12h = ds_var.sel(time=ds_var.time.dt.hour == 12)
            ds_var_daily = ds_var_12h.isel(time=slice(1, None))
        
        var_datasets.append(ds_var_daily.expand_dims(variable=[var]))
    
    # Combine all variables
    print(f"    Combining variables...")
    ds_init_daily = xr.concat(var_datasets, dim='variable', coords='minimal')
    
    # Add forecast_time coordinate
    ds_init_daily = ds_init_daily.assign_coords(forecast_time=('time', ref_times))
    
    # Replace time with lead_time in days
    lead_times = (ds_init_daily.time.values - init_date) / np.timedelta64(1, 'D')
    ds_init_daily = ds_init_daily.assign_coords(time=lead_times).rename({'time': 'lead_time'})
    
    # Convert to Dataset with separate variables
    xr_ds_init = xr.Dataset({
        var: ds_init_daily.sel(variable=var).drop_vars('variable') 
        for var in available_vars
    })
    
    # Compute weekly means
    print(f"    Computing weekly means...")
    ds_init_weekly = xr_ds_init.coarsen(lead_time=7, boundary='trim').mean()
    ds_init_weekly = ds_init_weekly.rename({"lead_time": "week_lead_time"})
    ds_init_weekly = ds_init_weekly.assign_coords(week_lead_time=np.arange(1, 9))
    
    # Add init_date dimension
    ds_init_weekly = ds_init_weekly.expand_dims(init_date=[init_date])
    
    # Compute and load into memory immediately
    print(f"    Computing results...")
    with ProgressBar():
        ds_init_weekly = ds_init_weekly.compute()
    
    print(f"    Completed ({dict(ds_init_weekly.dims)})")
    
    return ds_init_weekly


def process_observations(init_dates, available_vars):
    """
    Load and process observations to weekly means
    Process one initialization date at a time to reduce memory usage
    """
    print("\n" + "="*80)
    print("PROCESSING OBSERVATIONS")
    print("="*80)
    
    print("\nOpening zarr file with chunking...")
    obs = xr.open_zarr(RAW_OBS_PATH, chunks={'time': 500, 'values': 10000})
    times = obs.dates.values
    obs = obs.assign_coords(time=times).rename({"cell": "values"})
    
    # Get all variable names from dataset
    all_var_names = obs.attrs["variables"]
    
    # Find indices for variables we need
    var_indices = []
    for var in available_vars:
        if var in all_var_names:
            var_indices.append(all_var_names.index(var))
        else:
            raise ValueError(f"Variable {var} not found in observations!")
    
    print(f"Found all {len(available_vars)} variables in observations")
    
    obs = obs.isel(variable=var_indices)
    obs = obs.assign_coords(variable=available_vars)
    
    # Process one init date at a time
    print(f"\nProcessing {len(init_dates)} initialization dates (one at a time to save memory)...")
    
    obs_weekly_list = []
    for init_date in init_dates:
        ds_weekly = process_observations_one_init(obs, init_date, available_vars)
        obs_weekly_list.append(ds_weekly)
    
    # Concatenate all init dates
    print("\nConcatenating all initialization dates...")
    ds_obs_weekly = xr.concat(obs_weekly_list, dim='init_date')
    
    print(f"\nObservation weekly dataset shape: {dict(ds_obs_weekly.dims)}")
    print(f"Variables: {list(ds_obs_weekly.data_vars)}")
    
    # Save to NetCDF
    output_file = os.path.join(OUTPUT_DIR, "Observations_weekly_AIFS_physics.nc")
    print(f"\nSaving to: {output_file}")
    print("  (This may take a few minutes...)")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Actually save the file with compression
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_obs_weekly.data_vars}
    ds_obs_weekly.to_netcdf(output_file, encoding=encoding)
    
    print(f"Saved observation weekly means ({os.path.getsize(output_file) / 1e9:.2f} GB)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main preprocessing pipeline
    """
    # Process forecasts first (easier, determines available variables)
    print("\nStep 1: Processing forecasts...")
    init_dates, available_vars = process_forecasts()
    
    # Process observations (memory-intensive, one init date at a time)
    print("\nStep 2: Processing observations...")
    process_observations(init_dates, available_vars)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  1. Forecasts_weekly_AIFS_physics.nc")
    print(f"  2. Observations_weekly_AIFS_physics.nc")
    print("\nYou can now run the enhanced heat wave analysis script with these files.")
    print("="*80)


if __name__ == "__main__":
    main()
