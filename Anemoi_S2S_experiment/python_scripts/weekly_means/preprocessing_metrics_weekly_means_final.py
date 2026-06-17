#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CRITICAL: Set environment variables BEFORE importing any libraries
import os
import sys

# Configure HDF5 to avoid file handle issues - MUST be set before importing xarray/h5py
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Force unbuffered output so prints appear immediately in logs
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

print("=" * 80, flush=True)
print("SCRIPT STARTING - Imports beginning...", flush=True)
print("=" * 80, flush=True)

# Now import the rest
import xarray as xr
import numpy as np
import pandas as pd
import glob
import gc
import importlib

sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import metrics_function as mf
importlib.reload(mf)

print("Imports complete!", flush=True)

# Configure Dask to use all available threads
import dask
n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
dask.config.set(scheduler='threads', num_workers=n_workers)
print(f"Dask configured to use {n_workers} threads", flush=True)

############################################################################################################################################################

print("=" * 80, flush=True)
print("PREPROCESSING WEEKLY MEANS DATA - 75 INIT DATES", flush=True)
print("=" * 80, flush=True)

#loading and preprocessing the data
#load dataset with observations
print("Loading observations...", flush=True)
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-6h-v1-for-single-v2.zarr"
# Use consolidated=False to avoid the slow fallback warning
obs = xr.open_zarr(dataset_path, consolidated=False)
times = obs.dates.values
obs = obs.assign_coords(time=times).rename({"cell": "values"})

#select only a subset of variable
var_names = obs.attrs["variables"]
var_of_interest = ['2t', '2d', 'msl', 'sp', 'skt', 'tp', '10u', '10v','t_500', 't_700', 't_850', 't_925', 'q_500', 'q_700', 'q_850', 'q_925', "z_500", "z_700", "z_850", "z_925"]
var_indices = [var_names.index(var) for var in var_of_interest]
obs = obs.isel(variable=var_indices)

#create latitudinal weights for the metrics
lat_lon_coords = {
            'latitude': obs['latitudes'],
            'longitude': obs['longitudes']
        }
lat_weights = np.cos(np.radians(lat_lon_coords["latitude"].values))
lat_weights = xr.DataArray(lat_weights, dims=["values"])

############################################################################################################################################################
# NEW: Create reference model with [-27, 0, 'frequency'] structure (28 timesteps, backward-looking, no overlap)
print("=" * 80)
print("Creating NEW reference model weekly means matching [-27, 0, 'frequency'] structure...")
print("This uses 28 timesteps per week (backward-looking window, no boundary overlap)")
print("=" * 80)

import re

paths_init_ref = sorted(glob.glob(
    "/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/AIFS-6hourly-75-init/"
    "aifs-subs-pretrain-*"
))

if not paths_init_ref:
    print("Warning: No directories found for reference model 6-hourly data, skipping...")
    ds_inf_weekly_27_0_ref = None
    ref_init_dates_list = []
else:
    print(f"Found {len(paths_init_ref)} directories for reference model 6-hourly data")
    
    ds_inf_ref_weekly_27_0_list = []
    ref_init_dates_list = []
    
    for idx, path_init in enumerate(paths_init_ref, 1):
        print(f"[{idx}/{len(paths_init_ref)}] Processing {os.path.basename(path_init)}...")
        
        all_files = sorted(glob.glob(f"{path_init}/*.nc"))
        if not all_files:
            print(f"  Warning: No .nc files found, skipping...")
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
            print(f"  Warning: No valid member files found, skipping...")
            continue
        
        member_weekly_datasets = []
        
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
            
            # We need all 4 hours to create continuous series
            required_hours = ['0000', '0600', '1200', '1800']
            if not all(h in files_by_hour for h in required_hours):
                print(f"  Warning: Missing some initialization hours for member {member_num}, skipping...")
                continue
            
            # Load each initialization hour's forecast
            forecasts_by_hour = {}
            for hour in required_hours:
                ds = xr.open_dataset(files_by_hour[hour])
                ds = ds[var_of_interest]  # Select only variables of interest
                forecasts_by_hour[hour] = ds
            
            # Extract the base date from the 12:00 initialization
            if member_num == 0:
                base_time = pd.to_datetime(forecasts_by_hour['1200'].time.values[0])
                init_date = pd.Timestamp(year=base_time.year, month=base_time.month, 
                                        day=base_time.day, hour=12).to_datetime64()
                print(f"  Base init_date set to: {pd.to_datetime(init_date).strftime('%Y-%m-%d %H:%M')}")
            
            # Create continuous 6-hourly time series
            continuous_timesteps = []
            
            # Check forecast length
            forecast_lengths = {hour: len(forecasts_by_hour[hour].time) for hour in required_hours}
            if len(set(forecast_lengths.values())) > 1:
                print(f"  ERROR: Member {member_num} has inconsistent forecast lengths: {forecast_lengths}")
                continue
            
            n_days = forecast_lengths['0000']
            print(f"  Member {member_num}: Each forecast has {n_days} timesteps")
            
            # Build continuous series starting at 12:00
            continuous_timesteps.append(forecasts_by_hour['1200'].isel(time=0).drop_vars('time'))
            continuous_timesteps.append(forecasts_by_hour['1800'].isel(time=0).drop_vars('time'))
            
            for day in range(1, n_days):
                continuous_timesteps.append(forecasts_by_hour['0000'].isel(time=day).drop_vars('time'))
                continuous_timesteps.append(forecasts_by_hour['0600'].isel(time=day).drop_vars('time'))
                continuous_timesteps.append(forecasts_by_hour['1200'].isel(time=day).drop_vars('time'))
                continuous_timesteps.append(forecasts_by_hour['1800'].isel(time=day).drop_vars('time'))
            
            # Concatenate all timesteps
            ds_member_continuous = xr.concat(continuous_timesteps, dim='time')
            
            # Assign proper 6-hourly time coordinates
            start_time = pd.to_datetime(forecasts_by_hour['1200'].time.values[0])
            time_coords = pd.date_range(start=start_time, periods=len(ds_member_continuous.time), freq='6h')
            ds_member_continuous = ds_member_continuous.assign_coords(time=time_coords)
            
            if member_num == 0:
                print(f"  Total timesteps in continuous series: {len(ds_member_continuous.time)}")
            
            # NEW: Compute weekly means using [-27, 0] structure (28 timesteps, backward-looking)
            n_weeks = 8
            n_timesteps_per_week = 28
            i_start = 27  # lookback for rolling average [-27, 0, 'frequency']
            week_means_member = []
            
            for week_num in range(1, n_weeks + 1):
                # Week k: forecast at index k*28, uses rolling avg of [(k-1)*28+1 : k*28+1]
                forecast_idx = week_num * n_timesteps_per_week
                start_idx = forecast_idx - i_start  # = forecast_idx - 27
                end_idx = forecast_idx + 1          # +1 to include the forecast timestep itself
                
                if end_idx > len(ds_member_continuous.time):
                    break
                
                if member_num == 0 and week_num == 1:
                    start_time_w = pd.to_datetime(ds_member_continuous.time.values[start_idx])
                    end_time_w = pd.to_datetime(ds_member_continuous.time.values[end_idx - 1])
                    print(f"  Week 1 uses timesteps [{start_idx}:{end_idx}] (28 timesteps)")
                    print(f"    From: {start_time_w.strftime('%Y-%m-%d %H:%M')}")
                    print(f"    To:   {end_time_w.strftime('%Y-%m-%d %H:%M')}")
                
                week_slice = ds_member_continuous.isel(time=slice(start_idx, end_idx))
                
                week_mean_dict = {}
                for var in var_of_interest:
                    if var == "tp":
                        # For tp: compute daily means then sum
                        daily_means = []
                        
                        # Day 0: 2 timesteps (12:00, 18:00)
                        if len(week_slice.time) >= 2:
                            day0_mean = week_slice[var].isel(time=slice(0, 2)).mean(dim='time')
                            daily_means.append(day0_mean)
                        
                        # Days 1-6: 4 timesteps each
                        for day in range(6):
                            start_idx_day = 2 + day * 4
                            end_idx_day = start_idx_day + 4
                            if end_idx_day <= len(week_slice.time):
                                day_mean = week_slice[var].isel(time=slice(start_idx_day, end_idx_day)).mean(dim='time')
                                daily_means.append(day_mean)
                        
                        # Day 7: remaining timesteps
                        remaining_start = 2 + 6 * 4
                        if remaining_start < len(week_slice.time):
                            day7_mean = week_slice[var].isel(time=slice(remaining_start, None)).mean(dim='time')
                            daily_means.append(day7_mean)
                        
                        week_mean_dict[var] = xr.concat(daily_means, dim='temp_day').sum(dim='temp_day')
                    else:
                        week_mean_dict[var] = week_slice[var].mean(dim='time')
                
                week_mean_ds = xr.Dataset(week_mean_dict)
                week_means_member.append(week_mean_ds)
            
            member_weekly = xr.concat(
                week_means_member,
                dim=xr.DataArray(range(1, len(week_means_member) + 1), dims='week_lead_time', name='week_lead_time')
            )
            
            member_weekly_datasets.append(member_weekly)
            
            del forecasts_by_hour, ds_member_continuous, continuous_timesteps, week_means_member
            gc.collect()
        
        if not member_weekly_datasets:
            print(f"  Warning: No valid data processed, skipping...")
            continue
        
        init_weekly = xr.concat(
            member_weekly_datasets,
            dim=xr.DataArray(range(len(member_weekly_datasets)), dims='member', name='member')
        )
        
        ds_inf_ref_weekly_27_0_list.append(init_weekly)
        ref_init_dates_list.append(init_date)
        
        del member_weekly_datasets, init_weekly
        gc.collect()
        
        print(f"  Completed: {pd.to_datetime(init_date).strftime('%Y-%m-%d')}")
    
    if ds_inf_ref_weekly_27_0_list:
        ds_inf_weekly_27_0_ref = xr.concat(
            ds_inf_ref_weekly_27_0_list,
            dim=xr.DataArray(ref_init_dates_list, dims='init_date', name='init_date')
        )
        print(f"Processed {len(ref_init_dates_list)} init dates for reference model")
        print(f"Reference model (27-0) dimensions: {dict(ds_inf_weekly_27_0_ref.dims)}")
    else:
        ds_inf_weekly_27_0_ref = None

############################################################################################################################################################
# NEW: Load weekly means model with rolling average [-27, 0, 'frequency'] (28 timesteps, no boundary overlap)
print("=" * 80)
print("Loading NEW weekly means model with rolling average [-27, 0, 'frequency']...")
print("This model uses 28 timesteps per week (no duplicate at boundaries)")
print("=" * 80)

paths_init_weekly_27_0 = sorted(glob.glob(
    "/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/WeeklyMeans-27-0-freq-75-init/"
    "aifs-subs-pretrain-*"
))

if not paths_init_weekly_27_0:
    print(f"Warning: No directories found for WeeklyMeans-27-0-freq-75-init, skipping...")
    ds_inf_weekly_27_0 = None
    weekly_init_dates = None
else:
    print(f"Found {len(paths_init_weekly_27_0)} directories for WeeklyMeans-27-0-freq-75-init")
    
    reforecast_datasets_27_0 = []
    init_dates_27_0 = []
    
    for path_init in paths_init_weekly_27_0:
        paths_member = sorted(glob.glob(f"{path_init}/*.nc"))
        
        if not paths_member:
            print(f"Warning: No .nc files found in {path_init}, skipping...")
            continue
        
        print(f"Loading {len(paths_member)} members from {os.path.basename(path_init)}")
        
        ds_init = xr.open_mfdataset(
            paths_member,
            combine="nested",
            concat_dim="member",
            parallel=False,
            chunks={}
        )
        
        init_date = ds_init.time.values[0].astype('datetime64[ns]')
        init_dates_27_0.append(init_date)
        
        ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
        lead_times = (ds_init.time.values - init_date) / np.timedelta64(7, 'D')
        ds_init = ds_init.assign_coords(time=lead_times).rename({'time': 'week_lead_time'})
        ds_init = ds_init.isel(week_lead_time=slice(1, None))
        
        reforecast_datasets_27_0.append(ds_init)
    
    if not reforecast_datasets_27_0:
        print(f"Warning: No valid data loaded for WeeklyMeans-27-0-freq-75-init")
        ds_inf_weekly_27_0 = None
        weekly_init_dates = None
    else:
        ds_inf_weekly_27_0 = xr.concat(
            reforecast_datasets_27_0,
            dim=xr.DataArray(init_dates_27_0, dims='init_date', name='init_date')
        )
        ds_inf_weekly_27_0 = ds_inf_weekly_27_0[var_of_interest]
        ds_inf_weekly_27_0 = ds_inf_weekly_27_0.transpose('week_lead_time', 'init_date', 'member', 'values')
        weekly_init_dates = init_dates_27_0
        print(f"Loaded weekly means model (27-0): {dict(ds_inf_weekly_27_0.dims)}")

############################################################################################################################################################
# NEW: Load daily means model with rolling average [-3, 0, 'frequency'] (4 timesteps, no boundary overlap)
print("=" * 80)
print("Loading NEW daily means model with rolling average [-3, 0, 'frequency']...")
print("This model uses 4 timesteps per day (no duplicate at boundaries)")
print("=" * 80)

paths_init_daily_3_0 = sorted(glob.glob(
    "/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/DailyMeans-3-0-freq-75-init/"
    "aifs-subs-pretrain-*"
))

if not paths_init_daily_3_0:
    print(f"Warning: No directories found for DailyMeans-3-0-freq-75-init, skipping...")
    ds_inf_daily_3_0 = None
else:
    print(f"Found {len(paths_init_daily_3_0)} directories for DailyMeans-3-0-freq-75-init")
    
    reforecast_datasets_daily_3_0 = []
    init_dates_daily_3_0 = []
    
    for path_init in paths_init_daily_3_0:
        paths_member = sorted(glob.glob(f"{path_init}/*.nc"))
        
        if not paths_member:
            print(f"Warning: No .nc files found in {path_init}, skipping...")
            continue
        
        ds_init = xr.open_mfdataset(
            paths_member,
            combine="nested",
            concat_dim="member",
            parallel=False,
            chunks={}
        )
        
        init_date = ds_init.time.values[0].astype('datetime64[ns]')
        init_dates_daily_3_0.append(init_date)
        
        ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
        lead_times = (ds_init.time.values - init_date) / np.timedelta64(1, 'D')
        ds_init = ds_init.assign_coords(time=lead_times).rename({'time': 'lead_time'})
        
        # Remove day 0 if present (not valid with rolling average window)
        if 0 in ds_init.lead_time.values:
            print(f"  Removing day 0 from daily finetuned model (not valid with rolling average window)")
            ds_init = ds_init.sel(lead_time=slice(1, None))
        
        reforecast_datasets_daily_3_0.append(ds_init)
    
    if not reforecast_datasets_daily_3_0:
        print(f"Warning: No valid data loaded for DailyMeans-3-0-freq-75-init")
        ds_inf_daily_3_0 = None
    else:
        ds_inf_daily_3_0 = xr.concat(
            reforecast_datasets_daily_3_0,
            dim=xr.DataArray(init_dates_daily_3_0, dims='init_date', name='init_date')
        )
        ds_inf_daily_3_0 = ds_inf_daily_3_0[var_of_interest]
        print(f"Loaded daily means model (3-0): {dict(ds_inf_daily_3_0.dims)}")

# NEW: Create weekly means for daily finetuned model with [-3, 0, 'frequency'] structure (no overlap)
print("=" * 80)
print("Creating weekly aggregation from daily model (3-0)...")
print("=" * 80)

if ds_inf_daily_3_0 is not None:
    # Aggregate daily data to weekly (7 days per week, no overlap)
    n_weeks = 8
    n_days_per_week = 7
    
    weekly_datasets = []
    
    for init_idx in range(len(ds_inf_daily_3_0.init_date)):
        daily_for_init = ds_inf_daily_3_0.isel(init_date=init_idx)
        
        week_means = []
        for week_num in range(1, n_weeks + 1):
            # Week 1: days [1:8] (7 days)
            # Week 2: days [8:15] (7 days), etc.
            start_day = 1 + (week_num - 1) * n_days_per_week
            end_day = start_day + n_days_per_week - 1  # Inclusive end
            
            # Select days for this week
            week_slice = daily_for_init.sel(lead_time=slice(start_day, end_day))
            
            if len(week_slice.lead_time) == 0:
                break
            
            # Compute mean for each variable
            week_mean_dict = {}
            for var in var_of_interest:
                if var == "tp":
                    week_mean_dict[var] = week_slice[var].sum(dim='lead_time')
                else:
                    week_mean_dict[var] = week_slice[var].mean(dim='lead_time')
            
            week_mean_ds = xr.Dataset(week_mean_dict)
            week_means.append(week_mean_ds)
        
        init_weekly = xr.concat(
            week_means,
            dim=xr.DataArray(range(1, len(week_means) + 1), dims='week_lead_time', name='week_lead_time')
        )
        weekly_datasets.append(init_weekly)
    
    ds_inf_daily_weekly_3_0 = xr.concat(
        weekly_datasets,
        dim=ds_inf_daily_3_0.init_date
    )
    
    print(f"Created weekly aggregation from daily model (3-0): {dict(ds_inf_daily_weekly_3_0.dims)}")
else:
    ds_inf_daily_weekly_3_0 = None

##################################################################################################################################
# NEW: Create ground truth for models with rolling average [-27, 0, 'frequency'] (28 timesteps, no boundary overlap)
print("=" * 80)
print("Creating NEW weekly observations matching [-27, 0, 'frequency'] structure...")
print("This uses 28 timesteps per week (backward-looking window)")
print("=" * 80)

if weekly_init_dates is None:
    print("Warning: No weekly init dates available, cannot create observations")
    ds_obs_weekly_27_0 = None
else:
    ds_obs_weekly_27_0_list = []
    
    for init_date in weekly_init_dates:
        t0 = np.where(times == init_date)[0][0]
        
        # NEW: 28 timesteps per week, backward-looking
        n_timesteps_per_week = 28
        n_weeks = 8
        i_start = 27  # lookback for rolling average [-27, 0, 'frequency']
        
        total_timesteps = n_timesteps_per_week * n_weeks + i_start
        
        t1 = t0 + total_timesteps
        
        ds_init = obs.isel(time=slice(t0, t1))
        ds_init = ds_init.squeeze("ensemble").assign_coords(variable=var_of_interest)
        
        week_means = []
        
        for week_num in range(1, n_weeks + 1):
            # Week k: forecast at index k*28, uses rolling avg of [(k-1)*28+1 : k*28+1]
            forecast_idx = week_num * n_timesteps_per_week
            start_idx = forecast_idx - i_start
            end_idx = forecast_idx + 1
            
            week_slice = ds_init.isel(time=slice(start_idx, end_idx))
            
            var_week_means = []
            for var in var_of_interest:
                var_data = week_slice.sel(variable=var)['data']
                
                if var == "tp":
                    var_week_mean = var_data.sum(dim='time')
                else:
                    var_week_mean = var_data.mean(dim='time')
                
                var_week_means.append(var_week_mean)
            
            week_data = xr.concat(var_week_means, dim=xr.DataArray(var_of_interest, dims='variable', name='variable'))
            week_means.append(week_data)
        
        obs_init = xr.concat(week_means, dim=xr.DataArray(range(1, n_weeks + 1), dims='week_lead_time', name='week_lead_time'))
        obs_init_ds = xr.Dataset({var: obs_init.sel(variable=var).drop_vars('variable') for var in var_of_interest})
        ds_obs_weekly_27_0_list.append(obs_init_ds)
    
    ds_obs_weekly_27_0 = xr.concat(ds_obs_weekly_27_0_list, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))
    
    print(f"Created weekly observations (27-0) for {len(weekly_init_dates)} init dates")
    print(f"Observation dataset (27-0) dimensions: {dict(ds_obs_weekly_27_0.dims)}")

##################################################################################################################################
# NEW: Climatology lists for new datasets with [-27, 0] and [-3, 0] structures
print("=" * 80)
print("Computing climatology for NEW models...")
print("=" * 80)

ds_clim_multiple_obs_27_0_init = []
ds_clim_multiple_inf_weekly_27_0_init = []
ds_clim_multiple_inf_weekly_27_0_ref_init = []
ds_clim_multiple_inf_daily_weekly_3_0_init = []

if weekly_init_dates is not None:
    for init_date in weekly_init_dates:
        year = pd.to_datetime(init_date).year
        month = pd.to_datetime(init_date).month
        day = pd.to_datetime(init_date).day
        
        # Climatology for observations (27-0)
        if ds_obs_weekly_27_0 is not None:
            mask_27_0 = (ds_obs_weekly_27_0.init_date.dt.month == month) & (ds_obs_weekly_27_0.init_date.dt.day == day) & (ds_obs_weekly_27_0.init_date.dt.year != year)
            climatology_weekly_obs_27_0 = ds_obs_weekly_27_0.isel(init_date=mask_27_0).mean(dim="init_date")
            ds_clim_multiple_obs_27_0_init.append(climatology_weekly_obs_27_0)
        
        # Climatology for weekly finetuned model (27-0)
        if ds_inf_weekly_27_0 is not None and init_date in ds_inf_weekly_27_0.init_date.values:
            mask_weekly_27_0 = (ds_inf_weekly_27_0.init_date.dt.month == month) & (ds_inf_weekly_27_0.init_date.dt.day == day) & (ds_inf_weekly_27_0.init_date.dt.year != year)
            climatology_weekly_inf_27_0 = ds_inf_weekly_27_0.isel(init_date=mask_weekly_27_0).mean(dim="init_date")
            ds_clim_multiple_inf_weekly_27_0_init.append(climatology_weekly_inf_27_0)
        
        # Climatology for daily finetuned model weekly aggregation (3-0)
        if ds_inf_daily_weekly_3_0 is not None and init_date in ds_inf_daily_weekly_3_0.init_date.values:
            mask_daily_3_0 = (ds_inf_daily_weekly_3_0.init_date.dt.month == month) & (ds_inf_daily_weekly_3_0.init_date.dt.day == day) & (ds_inf_daily_weekly_3_0.init_date.dt.year != year)
            climatology_weekly_inf_daily_3_0 = ds_inf_daily_weekly_3_0.isel(init_date=mask_daily_3_0).mean(dim="init_date")
            ds_clim_multiple_inf_daily_weekly_3_0_init.append(climatology_weekly_inf_daily_3_0)
        
        # Climatology for reference model (27-0)
        if ds_inf_weekly_27_0_ref is not None and init_date in ds_inf_weekly_27_0_ref.init_date.values:
            mask_ref_27_0 = (ds_inf_weekly_27_0_ref.init_date.dt.month == month) & (ds_inf_weekly_27_0_ref.init_date.dt.day == day) & (ds_inf_weekly_27_0_ref.init_date.dt.year != year)
            climatology_weekly_inf_27_0_ref = ds_inf_weekly_27_0_ref.isel(init_date=mask_ref_27_0).mean(dim="init_date")
            ds_clim_multiple_inf_weekly_27_0_ref_init.append(climatology_weekly_inf_27_0_ref)
    
    # Concatenate climatologies
    if ds_obs_weekly_27_0 is not None and ds_clim_multiple_obs_27_0_init:
        ds_climatology_weekly_obs_27_0 = xr.concat(ds_clim_multiple_obs_27_0_init, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))
    else:
        ds_climatology_weekly_obs_27_0 = None
    
    if ds_inf_weekly_27_0 is not None and ds_clim_multiple_inf_weekly_27_0_init:
        weekly_27_0_init_dates = [init_date for init_date in weekly_init_dates if init_date in ds_inf_weekly_27_0.init_date.values]
        ds_climatology_weekly_inf_27_0 = xr.concat(ds_clim_multiple_inf_weekly_27_0_init, dim=xr.DataArray(weekly_27_0_init_dates, dims='init_date', name='init_date'))
    else:
        ds_climatology_weekly_inf_27_0 = None
    
    if ds_inf_daily_weekly_3_0 is not None and ds_clim_multiple_inf_daily_weekly_3_0_init:
        daily_3_0_init_dates = [init_date for init_date in weekly_init_dates if init_date in ds_inf_daily_weekly_3_0.init_date.values]
        ds_climatology_weekly_inf_daily_3_0 = xr.concat(ds_clim_multiple_inf_daily_weekly_3_0_init, dim=xr.DataArray(daily_3_0_init_dates, dims='init_date', name='init_date'))
    else:
        ds_climatology_weekly_inf_daily_3_0 = None
    
    if ds_inf_weekly_27_0_ref is not None and ds_clim_multiple_inf_weekly_27_0_ref_init:
        ref_27_0_init_dates = [init_date for init_date in weekly_init_dates if init_date in ds_inf_weekly_27_0_ref.init_date.values]
        ds_climatology_weekly_inf_27_0_ref = xr.concat(ds_clim_multiple_inf_weekly_27_0_ref_init, dim=xr.DataArray(ref_27_0_init_dates, dims='init_date', name='init_date'))
    else:
        ds_climatology_weekly_inf_27_0_ref = None
    
    print(f"Computed climatology for {len(weekly_init_dates)} init dates")
    print(f"  Observations (27-0): {len(ds_clim_multiple_obs_27_0_init)} climatologies")
    print(f"  Weekly model (27-0): {len(ds_clim_multiple_inf_weekly_27_0_init)} climatologies")
    print(f"  Daily model weekly (3-0): {len(ds_clim_multiple_inf_daily_weekly_3_0_init)} climatologies")
    print(f"  Reference model (27-0): {len(ds_clim_multiple_inf_weekly_27_0_ref_init)} climatologies")
else:
    ds_climatology_weekly_obs_27_0 = None
    ds_climatology_weekly_inf_27_0 = None
    ds_climatology_weekly_inf_daily_3_0 = None
    ds_climatology_weekly_inf_27_0_ref = None

##################################################################################################################################
# NEW: Anomalies for new datasets
print("=" * 80)
print("Computing anomalies for NEW models...")
print("=" * 80)

if ds_obs_weekly_27_0 is not None and ds_climatology_weekly_obs_27_0 is not None:
    observed_anomalies_27_0 = ds_obs_weekly_27_0 - ds_climatology_weekly_obs_27_0
    print("Computed anomalies for observations (27-0)")
else:
    observed_anomalies_27_0 = None

if ds_inf_weekly_27_0 is not None and ds_climatology_weekly_inf_27_0 is not None:
    predicted_anomalies_weekly_27_0 = ds_inf_weekly_27_0 - ds_climatology_weekly_inf_27_0
    print("Computed anomalies for weekly model (27-0)")
else:
    predicted_anomalies_weekly_27_0 = None

if ds_inf_daily_weekly_3_0 is not None and ds_climatology_weekly_inf_daily_3_0 is not None:
    predicted_anomalies_daily_3_0 = ds_inf_daily_weekly_3_0 - ds_climatology_weekly_inf_daily_3_0
    print("Computed anomalies for daily model weekly (3-0)")
else:
    predicted_anomalies_daily_3_0 = None

if ds_inf_weekly_27_0_ref is not None and ds_climatology_weekly_inf_27_0_ref is not None:
    predicted_anomalies_ref_27_0 = ds_inf_weekly_27_0_ref - ds_climatology_weekly_inf_27_0_ref
    print("Computed anomalies for reference model (27-0)")
else:
    predicted_anomalies_ref_27_0 = None

############################################################################################################################################################
# Export preprocessed data
print("=" * 80)
print("EXPORTING PREPROCESSED DATA - 75 INIT DATES")
print("=" * 80)

output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init"
os.makedirs(output_dir, exist_ok=True)

# Export observations (27-0)
if ds_obs_weekly_27_0 is not None:
    ds_obs_weekly_27_0 = ds_obs_weekly_27_0.assign_coords(
        latitude=("values", lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values))
    
    obs_path_27_0 = os.path.join(output_dir, "observations_weekly_27_0.nc")
    if os.path.exists(obs_path_27_0):
        os.remove(obs_path_27_0)
    print(f"Saving observations (27-0) to {obs_path_27_0}")
    ds_obs_weekly_27_0.to_netcdf(obs_path_27_0, mode='w')
    
    if observed_anomalies_27_0 is not None:
        observed_anomalies_27_0 = observed_anomalies_27_0.assign_coords(
            latitude=("values", lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values))
        obs_anom_path_27_0 = os.path.join(output_dir, "observations_anomalies_weekly_27_0.nc")
        if os.path.exists(obs_anom_path_27_0):
            os.remove(obs_anom_path_27_0)
        print(f"Saving observation anomalies (27-0) to {obs_anom_path_27_0}")
        # Use Zarr format for robust handling of large dask arrays
        zarr_path = obs_anom_path_27_0.replace('.nc', '.zarr')
        if os.path.exists(zarr_path):
            import shutil
            shutil.rmtree(zarr_path)
        observed_anomalies_27_0.to_zarr(zarr_path, mode='w', consolidated=True)
        print(f"  Saved to Zarr format: {zarr_path}")
    
    if ds_climatology_weekly_obs_27_0 is not None:
        ds_climatology_weekly_obs_27_0 = ds_climatology_weekly_obs_27_0.assign_coords(
            latitude=("values", lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values))
        obs_clim_path_27_0 = os.path.join(output_dir, "observations_climatology_weekly_27_0.nc")
        if os.path.exists(obs_clim_path_27_0):
            os.remove(obs_clim_path_27_0)
        print(f"Saving observation climatology (27-0) to {obs_clim_path_27_0}")
        ds_climatology_weekly_obs_27_0.to_netcdf(obs_clim_path_27_0, mode='w')

# Force garbage collection to free memory and close file handles
gc.collect()

# Export weekly finetuned model (27-0)
if ds_inf_weekly_27_0 is not None:
    ds_inf_weekly_27_0 = ds_inf_weekly_27_0.assign_coords(
        latitude=("values", lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values))
    
    
    weekly_27_0_path = os.path.join(output_dir, "weekly_finetuned_model_27_0.nc")
    if os.path.exists(weekly_27_0_path):
        os.remove(weekly_27_0_path)
    print(f"Saving weekly finetuned model (27-0) to {weekly_27_0_path}")
    ds_inf_weekly_27_0.to_netcdf(weekly_27_0_path, mode='w')
    
    if predicted_anomalies_weekly_27_0 is not None:
        predicted_anomalies_weekly_27_0 = predicted_anomalies_weekly_27_0.assign_coords(
            latitude=("values", lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values))
        weekly_27_0_anom_path = os.path.join(output_dir, "weekly_finetuned_model_anomalies_27_0.nc")
        if os.path.exists(weekly_27_0_anom_path):
            os.remove(weekly_27_0_anom_path)
        print(f"Saving weekly finetuned model anomalies (27-0) to {weekly_27_0_anom_path}")
        # Use Zarr format for robust handling of large dask arrays
        zarr_path = weekly_27_0_anom_path.replace('.nc', '.zarr')
        if os.path.exists(zarr_path):
            import shutil
            shutil.rmtree(zarr_path)
        predicted_anomalies_weekly_27_0.to_zarr(zarr_path, mode='w', consolidated=True)
        print(f"  Saved to Zarr format: {zarr_path}")
    
    if ds_climatology_weekly_inf_27_0 is not None:
        ds_climatology_weekly_inf_27_0 = ds_climatology_weekly_inf_27_0.assign_coords(
            latitude=("values", lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values))
        weekly_27_0_clim_path = os.path.join(output_dir, "weekly_finetuned_model_climatology_27_0.nc")
        if os.path.exists(weekly_27_0_clim_path):
            os.remove(weekly_27_0_clim_path)
        print(f"Saving weekly finetuned model climatology (27-0) to {weekly_27_0_clim_path}")
        ds_climatology_weekly_inf_27_0.to_netcdf(weekly_27_0_clim_path, mode='w')

# Force garbage collection to free memory and close file handles
gc.collect()

# Export daily finetuned model weekly aggregation (3-0)
if ds_inf_daily_weekly_3_0 is not None:
    ds_inf_daily_weekly_3_0 = ds_inf_daily_weekly_3_0.assign_coords(
        latitude=("values", lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values))
    
    daily_3_0_weekly_path = os.path.join(output_dir, "daily_finetuned_model_weekly_3_0.nc")
    if os.path.exists(daily_3_0_weekly_path):
        os.remove(daily_3_0_weekly_path)
    print(f"Saving daily finetuned model weekly (3-0) to {daily_3_0_weekly_path}")
    ds_inf_daily_weekly_3_0.to_netcdf(daily_3_0_weekly_path, mode='w')
    
    if predicted_anomalies_daily_3_0 is not None:
        predicted_anomalies_daily_3_0 = predicted_anomalies_daily_3_0.assign_coords(
            latitude=("values", lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values))
        daily_3_0_weekly_anom_path = os.path.join(output_dir, "daily_finetuned_model_weekly_anomalies_3_0.nc")
        if os.path.exists(daily_3_0_weekly_anom_path):
            os.remove(daily_3_0_weekly_anom_path)
        print(f"Saving daily finetuned model weekly anomalies (3-0) to {daily_3_0_weekly_anom_path}")
        # Use Zarr format for robust handling of large dask arrays
        zarr_path = daily_3_0_weekly_anom_path.replace('.nc', '.zarr')
        if os.path.exists(zarr_path):
            import shutil
            shutil.rmtree(zarr_path)
        predicted_anomalies_daily_3_0.to_zarr(zarr_path, mode='w', consolidated=True)
        print(f"  Saved to Zarr format: {zarr_path}")
    
    if ds_climatology_weekly_inf_daily_3_0 is not None:
        ds_climatology_weekly_inf_daily_3_0 = ds_climatology_weekly_inf_daily_3_0.assign_coords(
            latitude=("values", lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values))
        daily_3_0_weekly_clim_path = os.path.join(output_dir, "daily_finetuned_model_weekly_climatology_3_0.nc")
        if os.path.exists(daily_3_0_weekly_clim_path):
            os.remove(daily_3_0_weekly_clim_path)
        print(f"Saving daily finetuned model weekly climatology (3-0) to {daily_3_0_weekly_clim_path}")
        ds_climatology_weekly_inf_daily_3_0.to_netcdf(daily_3_0_weekly_clim_path, mode='w')

# Force garbage collection to free memory and close file handles
gc.collect()

# Export reference model (27-0)
if ds_inf_weekly_27_0_ref is not None:
    print(f"Exporting reference model (27-0) data...")
    ds_inf_weekly_27_0_ref = ds_inf_weekly_27_0_ref.assign_coords(
        latitude=("values",lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values),)
    ref_path_27_0 = os.path.join(output_dir, "reference_model_weekly_27_0.nc")
    if os.path.exists(ref_path_27_0):
        os.remove(ref_path_27_0)
    print(f"Saving reference model (27-0) to {ref_path_27_0}")
    ds_inf_weekly_27_0_ref.to_netcdf(ref_path_27_0, mode='w')
    
    if predicted_anomalies_ref_27_0 is not None:
        print(f"Exporting reference model (27-0) anomalies...")
        predicted_anomalies_ref_27_0 = predicted_anomalies_ref_27_0.assign_coords(
            latitude=("values",lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values),)
        ref_anom_path_27_0 = os.path.join(output_dir, "reference_model_anomalies_weekly_27_0.nc")
        if os.path.exists(ref_anom_path_27_0):
            os.remove(ref_anom_path_27_0)
        print(f"Saving reference model (27-0) anomalies to {ref_anom_path_27_0}")
        # Use Zarr format for robust handling of large dask arrays
        zarr_path = ref_anom_path_27_0.replace('.nc', '.zarr')
        if os.path.exists(zarr_path):
            import shutil
            shutil.rmtree(zarr_path)
        predicted_anomalies_ref_27_0.to_zarr(zarr_path, mode='w', consolidated=True)
        print(f"  Saved to Zarr format: {zarr_path}")
    
    if ds_climatology_weekly_inf_27_0_ref is not None:
        print(f"Exporting reference model (27-0) climatology...")
        ds_climatology_weekly_inf_27_0_ref = ds_climatology_weekly_inf_27_0_ref.assign_coords(
            latitude=("values",lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values),)
        ref_clim_path_27_0 = os.path.join(output_dir, "reference_model_climatology_weekly_27_0.nc")
        if os.path.exists(ref_clim_path_27_0):
            os.remove(ref_clim_path_27_0)
        print(f"Saving reference model (27-0) climatology to {ref_clim_path_27_0}")
        ds_climatology_weekly_inf_27_0_ref.to_netcdf(ref_clim_path_27_0, mode='w')

# Force garbage collection to free memory and close file handles
gc.collect()

# Export lat_weights for use in metrics computation
lat_weights_path = os.path.join(output_dir, "lat_weights.nc")
if os.path.exists(lat_weights_path):
    os.remove(lat_weights_path)
print(f"Saving latitude weights to {lat_weights_path}")
lat_weights_ds = xr.Dataset({'lat_weights': lat_weights})
lat_weights_ds.to_netcdf(lat_weights_path, mode='w')

print("=" * 80)
print("PREPROCESSING COMPLETE - 75 INIT DATES")
print(f"All preprocessed data saved to: {output_dir}")
print("=" * 80)