"""
Standalone script to recompute ONLY the NEW reference model (27-0) with corrected tp handling.
This avoids re-running the entire preprocessing pipeline.
"""

import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import re
import gc

print("=" * 80)
print("RECOMPUTING NEW REFERENCE MODEL (27-0) WITH CORRECTED TP HANDLING")
print("=" * 80)

# Paths
preprocessed_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed"
var_of_interest = ["2t", "tp", "10u", "10v"]

# Load observations (27-0) - already correctly computed
print("Loading existing observations (27-0)...")
ds_obs_weekly_27_0 = xr.open_dataset(os.path.join(preprocessed_dir, "observations_weekly_27_0.nc"))
ds_climatology_weekly_obs_27_0 = xr.open_dataset(os.path.join(preprocessed_dir, "observations_climatology_weekly_27_0.nc"))
observed_anomalies_27_0 = xr.open_dataset(os.path.join(preprocessed_dir, "observations_anomalies_weekly_27_0.nc"))

weekly_init_dates = ds_obs_weekly_27_0.init_date.values
print(f"Found {len(weekly_init_dates)} init dates")

# Load raw 6-hourly reference model data
print("\nLoading raw 6-hourly reference model data...")
paths_init_ref = sorted(glob.glob(
    "/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/AIFS-6hourly/"
    "aifs-subs-pretrain-*"
))

print(f"Found {len(paths_init_ref)} directories for reference model")

# Recompute reference model with correct tp handling
print("\nReprocessing reference model with corrected tp computation...")
ds_inf_ref_weekly_27_0_list = []
ref_init_dates_list = []  # Collect init dates from actual data

for idx, path_init in enumerate(paths_init_ref, 1):
    print(f"[{idx}/{len(paths_init_ref)}] Processing {os.path.basename(path_init)}...")
    
    all_files = sorted(glob.glob(f"{path_init}/*.nc"))
    if not all_files:
        print(f"  No files found, skipping...")
        continue
    
    # Group files by member
    member_files = {}
    for filepath in all_files:
        filename = os.path.basename(filepath)
        match = re.match(r'member-(\d+)-(\d{4})\.nc', filename)
        if match:
            member_num = int(match.group(1))
            if member_num not in member_files:
                member_files[member_num] = []
            member_files[member_num].append(filepath)
    
    if not member_files:
        print(f"  No matching files found, skipping...")
        continue
    
    member_weekly_datasets = []
    init_date_for_this_path = None  # Store init date for this directory
    
    for member_num in sorted(member_files.keys()):
        time_files = sorted(member_files[member_num])
        
        # Separate by initialization hour
        files_by_hour = {}
        for filepath in time_files:
            filename = os.path.basename(filepath)
            match = re.match(r'member-\d+-(\d{4})\.nc', filename)
            if match:
                hour = match.group(1)
                files_by_hour[hour] = filepath
        
        required_hours = ['0000', '0600', '1200', '1800']
        if not all(h in files_by_hour for h in required_hours):
            print(f"  Member {member_num}: Missing required hours, skipping...")
            continue
        
        # Load each initialization hour's forecast
        forecasts_by_hour = {}
        for hour in required_hours:
            ds = xr.open_dataset(files_by_hour[hour])
            ds = ds[var_of_interest]
            forecasts_by_hour[hour] = ds
        
        # Get base init date from 12:00 initialization (extract from actual data)
        if member_num == 0:
            base_time = pd.to_datetime(forecasts_by_hour['1200'].time.values[0])
            init_date = pd.Timestamp(year=base_time.year, month=base_time.month, 
                                    day=base_time.day, hour=12).to_datetime64()
            init_date_for_this_path = init_date  # Save for later use
        
        # Create continuous 6-hourly series
        continuous_timesteps = []
        n_days = len(forecasts_by_hour['0000'].time)
        
        # Day 0: 12:00 and 18:00
        continuous_timesteps.append(forecasts_by_hour['1200'].isel(time=0).drop_vars('time'))
        continuous_timesteps.append(forecasts_by_hour['1800'].isel(time=0).drop_vars('time'))
        
        # Days 1 onwards: all 4 hours
        for day in range(1, n_days):
            continuous_timesteps.append(forecasts_by_hour['0000'].isel(time=day).drop_vars('time'))
            continuous_timesteps.append(forecasts_by_hour['0600'].isel(time=day).drop_vars('time'))
            continuous_timesteps.append(forecasts_by_hour['1200'].isel(time=day).drop_vars('time'))
            continuous_timesteps.append(forecasts_by_hour['1800'].isel(time=day).drop_vars('time'))
        
        ds_member_continuous = xr.concat(continuous_timesteps, dim='time')
        
        # Assign proper time coordinates
        start_time = pd.to_datetime(forecasts_by_hour['1200'].time.values[0])
        time_coords = pd.date_range(start=start_time, periods=len(ds_member_continuous.time), freq='6h')
        ds_member_continuous = ds_member_continuous.assign_coords(time=time_coords)
        
        # Compute weekly means with backward-looking 28-timestep windows
        n_weeks = 8
        n_timesteps_per_week = 28  # 7 days * 4 timesteps
        i_start = 27  # lookback for rolling average [-27, 0, 'frequency']
        week_means_member = []
        
        for week_num in range(1, n_weeks + 1):
            # Backward-looking window
            forecast_idx = week_num * n_timesteps_per_week
            start_idx = forecast_idx - i_start  # = forecast_idx - 27
            end_idx = forecast_idx + 1          # +1 to include forecast timestep
            
            if end_idx > len(ds_member_continuous.time):
                break
            
            week_slice = ds_member_continuous.isel(time=slice(start_idx, end_idx))
            
            # Compute mean for each variable
            week_mean_dict = {}
            for var in var_of_interest:
                if var == "tp":
                    # CORRECTED: tp values are 24h accumulations
                    # Compute daily means (average 4 6-hourly values per day), then sum 7 daily means
                    daily_means = []
                    
                    for day in range(7):
                        start_idx_day = day * 4
                        end_idx_day = start_idx_day + 4
                        if end_idx_day <= len(week_slice.time):
                            day_mean = week_slice[var].isel(time=slice(start_idx_day, end_idx_day)).mean(dim='time')
                            daily_means.append(day_mean)
                    
                    # Sum daily means to get weekly accumulation
                    week_mean_dict[var] = xr.concat(daily_means, dim='temp_day').sum(dim='temp_day')
                else:
                    week_mean_dict[var] = week_slice[var].mean(dim='time')
            
            week_mean_ds = xr.Dataset(week_mean_dict)
            week_means_member.append(week_mean_ds)
        
        # Stack all weeks for this member
        member_weekly = xr.concat(
            week_means_member,
            dim=xr.DataArray(range(1, len(week_means_member) + 1), dims='week_lead_time', name='week_lead_time')
        )
        
        member_weekly_datasets.append(member_weekly)
        
        # Free memory
        del forecasts_by_hour, ds_member_continuous, continuous_timesteps, week_means_member
        gc.collect()
    
    if not member_weekly_datasets:
        print(f"  Warning: No valid data processed, skipping...")
        continue
    
    # Concatenate members for this init date
    num_members = len(member_weekly_datasets)
    init_ds = xr.concat(
        member_weekly_datasets,
        dim=xr.DataArray(range(num_members), dims='member', name='member')
    )
    
    ds_inf_ref_weekly_27_0_list.append(init_ds)
    ref_init_dates_list.append(init_date_for_this_path)  # Use init_date extracted from data
    
    # Free memory
    del member_weekly_datasets, init_ds
    gc.collect()
    
    print(f"  Processed {num_members} members for init_date: {init_date_for_this_path}")

# Concatenate all init dates
print("\nConcatenating all init dates...")
print(f"Collected {len(ref_init_dates_list)} init dates from processed data")

ds_inf_weekly_27_0_ref = xr.concat(
    ds_inf_ref_weekly_27_0_list, 
    dim=xr.DataArray(ref_init_dates_list, dims='init_date', name='init_date')
)

ds_inf_weekly_27_0_ref = ds_inf_weekly_27_0_ref[var_of_interest]
print(f"Reference model (27-0) dimensions: {dict(ds_inf_weekly_27_0_ref.dims)}")

# Compute climatology for reference model
print("\nComputing climatology for reference model (27-0)...")
ds_clim_multiple_ref_27_0_init = []

for init_date in weekly_init_dates:
    init_date_pd = pd.to_datetime(init_date)
    year = init_date_pd.year
    month = init_date_pd.month
    day = init_date_pd.day
    
    mask_27_0 = (ds_inf_weekly_27_0_ref.init_date.dt.month == month) & \
                (ds_inf_weekly_27_0_ref.init_date.dt.day == day) & \
                (ds_inf_weekly_27_0_ref.init_date.dt.year != year)
    
    climatology_weekly_ref_27_0 = ds_inf_weekly_27_0_ref.isel(init_date=mask_27_0).mean(dim=["init_date", "member"])
    ds_clim_multiple_ref_27_0_init.append(climatology_weekly_ref_27_0)

ds_climatology_weekly_inf_27_0_ref = xr.concat(
    ds_clim_multiple_ref_27_0_init, 
    dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date')
)

print(f"Reference model (27-0) climatology computed: {dict(ds_climatology_weekly_inf_27_0_ref.dims)}")

# Compute anomalies
print("\nComputing anomalies for reference model (27-0)...")
predicted_anomalies_27_0_ref = ds_inf_weekly_27_0_ref - ds_climatology_weekly_inf_27_0_ref
print(f"Reference model (27-0) anomalies computed: {dict(predicted_anomalies_27_0_ref.dims)}")

# Save corrected data
print("\nSaving corrected reference model (27-0) data...")

# Assign coordinates
ds_inf_weekly_27_0_ref = ds_inf_weekly_27_0_ref.assign_coords(
    latitude=("values", ds_obs_weekly_27_0.latitude.values),
    longitude=("values", ds_obs_weekly_27_0.longitude.values)
)
ds_climatology_weekly_inf_27_0_ref = ds_climatology_weekly_inf_27_0_ref.assign_coords(
    latitude=("values", ds_obs_weekly_27_0.latitude.values),
    longitude=("values", ds_obs_weekly_27_0.longitude.values)
)
predicted_anomalies_27_0_ref = predicted_anomalies_27_0_ref.assign_coords(
    latitude=("values", ds_obs_weekly_27_0.latitude.values),
    longitude=("values", ds_obs_weekly_27_0.longitude.values)
)

# Save files
ref_path_27_0 = os.path.join(preprocessed_dir, "reference_model_weekly_27_0.nc")
ref_clim_path_27_0 = os.path.join(preprocessed_dir, "reference_model_climatology_weekly_27_0.nc")
ref_anom_path_27_0 = os.path.join(preprocessed_dir, "reference_model_anomalies_weekly_27_0.nc")

if os.path.exists(ref_path_27_0):
    os.remove(ref_path_27_0)
ds_inf_weekly_27_0_ref.to_netcdf(ref_path_27_0, mode='w')
print(f"Saved: {ref_path_27_0}")

if os.path.exists(ref_clim_path_27_0):
    os.remove(ref_clim_path_27_0)
ds_climatology_weekly_inf_27_0_ref.to_netcdf(ref_clim_path_27_0, mode='w')
print(f"Saved: {ref_clim_path_27_0}")

if os.path.exists(ref_anom_path_27_0):
    os.remove(ref_anom_path_27_0)
predicted_anomalies_27_0_ref.to_netcdf(ref_anom_path_27_0, mode='w')
print(f"Saved: {ref_anom_path_27_0}")

print("\n" + "=" * 80)
print("RECOMPUTATION COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("1. Run metrics_weekly_means.py to recompute metrics with corrected data")
print("2. Run plots_weekly_means.py to generate updated plots")
