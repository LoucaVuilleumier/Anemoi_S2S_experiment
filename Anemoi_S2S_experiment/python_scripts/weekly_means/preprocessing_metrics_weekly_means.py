import xarray as xr
import numpy as np
import pandas as pd
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import metrics_function as mf
import glob
import os
import gc
import importlib
from utils import metrics_function as mf
importlib.reload(mf)

# Configure Dask to use all available threads
import dask
n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
dask.config.set(scheduler='threads', num_workers=n_workers)
print(f"Dask configured to use {n_workers} threads")

############################################################################################################################################################

print("=" * 80)
print("PREPROCESSING WEEKLY MEANS DATA")
print("=" * 80)

#loading and preprocessing the data
#load dataset with observations
print("Loading observations...")
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-6h-v1-for-single-v2.zarr"
# Use consolidated=False to avoid the slow fallback warning
obs = xr.open_zarr(dataset_path, consolidated=False)
times = obs.dates.values
obs = obs.assign_coords(time=times).rename({"cell": "values"})

#select only a substet of variable
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

######################################################################
# Load reference model 6-hourly data and compute weekly means
# Strategy: Combine 4 forecasts initialized at 00:00, 06:00, 12:00, 18:00 on the same date
# Each forecast contributes only its matching hour to create a continuous 6-hourly time series
# This matches how the finetuned models were trained (on all 6-hourly data)
print("Loading reference model 6-hourly data and computing weekly means...")
print("Combining 00:00, 06:00, 12:00, 18:00 initializations to create continuous 6-hourly series")

paths_init = sorted(glob.glob(
    "/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_inference/AIFS-6hourly/"
    "aifs-subs-pretrain-*"
))

if not paths_init:
    print("Warning: No directories found for reference model 6-hourly data, skipping...")
    ds_inf_ref_weekly_list = []
    ref_init_dates_list = []
else:
    print(f"Found {len(paths_init)} directories for reference model 6-hourly data")
    
    ds_inf_ref_weekly_list = []
    ref_init_dates_list = []
    
    import re
    
    for idx, path_init in enumerate(paths_init, 1):
        print(f"[{idx}/{len(paths_init)}] Processing {os.path.basename(path_init)}...")
        
        # Get all files for this initialization date
        all_files = sorted(glob.glob(f"{path_init}/*.nc"))
        
        if not all_files:
            print(f"  Warning: No .nc files found, skipping...")
            continue
        
        # Group files by member number (member-X-HHMM.nc format)
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
        
        # Process each member
        member_weekly_datasets = []
        
        for member_num in sorted(member_files.keys()):
            # Load all 4 initialization times (00:00, 06:00, 12:00, 18:00)
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
                
                # DIAGNOSTIC: Check that forecast times start at the expected hour
                if member_num == 0:
                    first_time = pd.to_datetime(ds.time.values[0])
                    expected_hour = int(hour[:2])
                    if first_time.hour != expected_hour:
                        print(f"  WARNING: {hour} init has unexpected first hour: {first_time.hour} (expected {expected_hour})")
                    else:
                        print(f"  ✓ {hour} init starts at {first_time.strftime('%Y-%m-%d %H:%M')} (correct)")
            
            # Extract the base date from the 12:00 initialization (normalized to 12:00)
            if member_num == 0:
                base_time = pd.to_datetime(forecasts_by_hour['1200'].time.values[0])
                init_date = pd.Timestamp(year=base_time.year, month=base_time.month, 
                                        day=base_time.day, hour=12).to_datetime64()
                print(f"  Base init_date set to: {pd.to_datetime(init_date).strftime('%Y-%m-%d %H:%M')}")
            
            # Create continuous 6-hourly time series by interleaving the 4 forecasts
            # Each forecast contains one value per day at its initialization hour
            # Example: 0000.nc has day0 00:00, day1 00:00, day2 00:00, ... (57 timesteps = 57 days)
            # Interleave starting at 12:00 to match finetuned model training
            continuous_timesteps = []
            
            # DIAGNOSTIC: Check that all forecasts have the same length
            forecast_lengths = {hour: len(forecasts_by_hour[hour].time) for hour in required_hours}
            if len(set(forecast_lengths.values())) > 1:
                print(f"  ERROR: Member {member_num} has inconsistent forecast lengths: {forecast_lengths}")
                print(f"  Skipping this member...")
                continue
            
            # Get the number of days in the forecast (each forecast has one timestep per day)
            n_days = forecast_lengths['0000']
            
            # DIAGNOSTIC: Verify data structure (for all members now to catch issues)
            print(f"  Member {member_num}: Each forecast has {n_days} timesteps")
            if member_num == 0:
                print(f"  Verifying hour extraction for first 3 days...")
                for day in range(min(3, n_days)):
                    for hour in ['0000', '0600', '1200', '1800']:
                        sample_time = pd.to_datetime(forecasts_by_hour[hour].time.values[day])
                        expected_hour = int(hour[:2])
                        if sample_time.hour != expected_hour:
                            print(f"    WARNING: {hour} init, day {day} has hour {sample_time.hour} (expected {expected_hour})")
                        elif day == 0:  # Only print day 0 for brevity
                            print(f"    ✓ {hour} init, day {day}: {sample_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Build continuous series starting at 12:00 (matching finetuned model initialization)
            # Day 0: Start with 12:00 and 18:00
            continuous_timesteps.append(forecasts_by_hour['1200'].isel(time=0).drop_vars('time'))
            continuous_timesteps.append(forecasts_by_hour['1800'].isel(time=0).drop_vars('time'))
            
            # Days 1 onwards: Add all 4 hours (00, 06, 12, 18) by extracting timestep at day index
            for day in range(1, n_days):
                continuous_timesteps.append(forecasts_by_hour['0000'].isel(time=day).drop_vars('time'))
                continuous_timesteps.append(forecasts_by_hour['0600'].isel(time=day).drop_vars('time'))
                continuous_timesteps.append(forecasts_by_hour['1200'].isel(time=day).drop_vars('time'))
                continuous_timesteps.append(forecasts_by_hour['1800'].isel(time=day).drop_vars('time'))
            
            # Concatenate all timesteps to create continuous series (without time coordinate)
            ds_member_continuous = xr.concat(continuous_timesteps, dim='time')
            
            # Assign proper 6-hourly time coordinates
            # Start from the 12:00 time of day 0
            start_time = pd.to_datetime(forecasts_by_hour['1200'].time.values[0])
            time_coords = pd.date_range(start=start_time, periods=len(ds_member_continuous.time), freq='6h')
            ds_member_continuous = ds_member_continuous.assign_coords(time=time_coords)
            
            # DIAGNOSTIC: Verify the continuous series starts at 12:00 and has correct 6-hourly structure
            print(f"  Member {member_num}: Total timesteps in continuous series: {len(ds_member_continuous.time)}")
            if member_num == 0:
                print(f"  Verifying continuous series structure...")
                # Check first 10 timesteps
                for i in range(min(10, len(ds_member_continuous.time))):
                    time_val = pd.to_datetime(ds_member_continuous.time.values[i])
                    # Expected pattern: 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, ...
                    if i == 0:
                        expected_hour = 12
                    elif i == 1:
                        expected_hour = 18
                    else:
                        expected_hour = ((i - 2) % 4) * 6  # 0, 6, 12, 18, 0, 6, ...
                    
                    if time_val.hour != expected_hour:
                        print(f"    WARNING: Timestep {i} has hour {time_val.hour} (expected {expected_hour})")
                    elif i < 6:  # Only print first 6 for brevity
                        print(f"    ✓ Timestep {i}: {time_val.strftime('%Y-%m-%d %H:%M')}")
            
            # Compute weekly means from the continuous 6-hourly series
            n_weeks = 8
            n_timesteps_per_week = 28  # 7 days * 4 timesteps per day
            week_means_member = []
            
            for week_num in range(1, n_weeks + 1):
                
                # Week boundaries with overlap (matching observations structure)
                start_idx = (week_num - 1) * n_timesteps_per_week
                end_idx = start_idx + n_timesteps_per_week + 1  # +1 to include boundary
                
                # Make sure we don't exceed available data
                if end_idx > len(ds_member_continuous.time):
                    break
                
                # DIAGNOSTIC: Show time range for first week (member 0 only)
                if member_num == 0 and week_num == 1:
                    start_time = pd.to_datetime(ds_member_continuous.time.values[start_idx])
                    end_time = pd.to_datetime(ds_member_continuous.time.values[end_idx - 1])
                    n_timesteps = end_idx - start_idx
                    n_hours = (end_time - start_time).total_seconds() / 3600
                    n_days = n_hours / 24
                    print(f"  Week 1 uses timesteps [{start_idx}:{end_idx}] = {n_timesteps} timesteps")
                    print(f"    From: {start_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f"    To:   {end_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f"    Duration: {n_hours:.0f} hours ({n_days:.1f} days)")
                
                week_slice = ds_member_continuous.isel(time=slice(start_idx, end_idx))
                
                # Compute mean for each variable
                week_mean_dict = {}
                for var in var_of_interest:
                    if var == "tp":
                        # Reference model tp values are 24h accumulations from overlapping windows
                        # For each day, average the overlapping 24h accumulations, then sum to get weekly total
                        # Structure: day 0 (2 timesteps) + days 1-6 (4 each) + day 7 (3 timesteps) = 29 total
                        daily_means = []
                        
                        # Day 0: 2 timesteps (12:00, 18:00)
                        if len(week_slice.time) >= 2:
                            day0_mean = week_slice[var].isel(time=slice(0, 2)).mean(dim='time')
                            daily_means.append(day0_mean)
                            if member_num == 0 and week_num == 1:
                                print(f"    tp Day 0: mean of 2 timesteps = {float(day0_mean.mean().compute().values):.4f} mm")
                        
                        # Days 1-6: 4 timesteps each
                        for day in range(6):
                            start_idx_day = 2 + day * 4
                            end_idx_day = start_idx_day + 4
                            if end_idx_day <= len(week_slice.time):
                                day_mean = week_slice[var].isel(time=slice(start_idx_day, end_idx_day)).mean(dim='time')
                                daily_means.append(day_mean)
                                if member_num == 0 and week_num == 1 and day < 2:  # Show first 2 days only
                                    print(f"    tp Day {day+1}: mean of 4 timesteps = {float(day_mean.mean().compute().values):.4f} mm")
                        
                        # Day 7: remaining timesteps (should be 3)
                        remaining_start = 2 + 6 * 4  # = 26
                        if remaining_start < len(week_slice.time):
                            day7_mean = week_slice[var].isel(time=slice(remaining_start, None)).mean(dim='time')
                            daily_means.append(day7_mean)
                            if member_num == 0 and week_num == 1:
                                n_remaining = len(week_slice.time) - remaining_start
                                print(f"    tp Day 7: mean of {n_remaining} timesteps = {float(day7_mean.mean().compute().values):.4f} mm")
                        
                        # DIAGNOSTIC: Show summary (member 0, week 1 only)
                        if member_num == 0 and week_num == 1:
                            print(f"    tp: Computed {len(daily_means)} daily means")
                        
                        # Sum daily means to get weekly accumulation
                        week_mean_dict[var] = xr.concat(daily_means, dim='temp_day').sum(dim='temp_day')
                        
                        # DIAGNOSTIC: Show final weekly sum (member 0, week 1 only)
                        if member_num == 0 and week_num == 1:
                            weekly_sum = float(week_mean_dict[var].mean().compute().values)
                            print(f"    tp: Weekly accumulation (sum of daily means) = {weekly_sum:.4f} mm")
                    else:
                        # For other variables: average all timesteps in the week
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
        
        # Concatenate all members for this init date
        init_weekly = xr.concat(
            member_weekly_datasets,
            dim=xr.DataArray(range(len(member_weekly_datasets)), dims='member', name='member')
        )
        
        ds_inf_ref_weekly_list.append(init_weekly)
        ref_init_dates_list.append(init_date)
        
        # Free memory
        del member_weekly_datasets, init_weekly
        gc.collect()
        
        print(f"  Completed: {pd.to_datetime(init_date).strftime('%Y-%m-%d')}")
    
    print(f"Processed {len(ref_init_dates_list)} init dates for reference model")


##################################################
#load model finetuned on daily means
print("Loading daily finetuned model...")
#Same process for the daily finetuned model
paths_init = sorted(glob.glob(
    "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/DailyMeans/"
    "aifs-subs-pretrain-*"
))

# Load all reforecasts with an additional init_date dimension
reforecast_datasets = []
init_dates = []

for path_init in paths_init:
    # Get all member paths for this initialization date
    paths_member = sorted(glob.glob(f"{path_init}/*.nc"))
    
    # Load all members for this init date
    ds_init = xr.open_mfdataset(
        paths_member,
        combine="nested",
        concat_dim="member",
        parallel=False,  
        chunks={}        
    )
    
    # Extract initialization date from the dataset
    init_date = ds_init.time.values[0].astype('datetime64[ns]')
    init_dates.append(init_date)
    
    
    # Store the absolute time as a non-dimension coordinate
    ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
    
    # Replace time dimension with relative lead_time (in days from init)
    lead_times = (ds_init.time.values - init_date) / np.timedelta64(1, 'D')
    ds_init = ds_init.assign_coords(time=lead_times).rename({'time': 'lead_time'})
    
    # NOTE: With rolling average window=(-4, 0, 'freq'), the first valid forecast is at day 1 (24h)
    # If the model outputs include day 0, it should be removed as it's not a valid rolling average
    # Check if day 0 exists and remove it
    if 0 in ds_init.lead_time.values:
        print(f"  Removing day 0 from daily finetuned model (not valid with rolling average window)")
        ds_init = ds_init.sel(lead_time=slice(1, None))
    
    reforecast_datasets.append(ds_init)

# Concatenate all reforecasts along a new 'init_date' dimension
ds_inf_daily_finetuned = xr.concat(
    reforecast_datasets, 
    dim=xr.DataArray(init_dates, dims='init_date', name='init_date')
)

print(f"Loaded {len(init_dates)} reforecasts")
print(f"Dataset dimensions: {dict(ds_inf_daily_finetuned.dims)}")

#select only variables of interest for the forecast
ds_inf_daily_finetuned = ds_inf_daily_finetuned[var_of_interest]

######################################################################################
#load finetuned models on weekly means
print("Loading weekly finetuned models...")
#Same process with finetune models for weekly means
finetune_models_list = ["Weekly_Means_14k_lr_0.625e-5", "Weekly_Means_14k_lr_0.625e-7", "Weekly_Means_18k_lr_0.625e-6"]
finetune_models = {}
for finetune_model in finetune_models_list:
    paths_init_finetune = sorted(glob.glob(
        f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/{finetune_model}/"
        "aifs-subs-pretrain-weeklymeans-*"
    ))
    
    if not paths_init_finetune:
        print(f"Warning: No directories found for {finetune_model}, skipping model...")
        continue
    
    print(f"Found {len(paths_init_finetune)} directories for {finetune_model}")

    # Load all reforecasts with an additional init_date dimension
    reforecast_datasets_finetuning = []
    init_dates_finetuning = []

    for path_init in paths_init_finetune:
        # Get all member paths for this initialization date
        paths_member = sorted(glob.glob(f"{path_init}/*.nc"))
        
        if not paths_member:
            print(f"Warning: No .nc files found in {path_init}, skipping...")
            continue
        
        print(f"Loading {len(paths_member)} members from {os.path.basename(path_init)}")
        
        # Load all members for this init date
        ds_init = xr.open_mfdataset(
            paths_member,
            combine="nested",
            concat_dim="member",
            parallel=False,  
            chunks={}        
        )
        
        # Extract initialization date from the dataset
        init_date = ds_init.time.values[0].astype('datetime64[ns]')
        init_dates_finetuning.append(init_date)
        
        
        # Store the absolute time as a non-dimension coordinate
        ds_init = ds_init.assign_coords(forecast_time=('time', ds_init.time.values))
        
        # Replace time dimension with relative lead_time (in weeks from init)
        lead_times = (ds_init.time.values - init_date) / np.timedelta64(7, 'D')
        ds_init = ds_init.assign_coords(time=lead_times).rename({'time': 'week_lead_time'})
        
        # Remove the first timestep (day 0 init) to compare only forecasts
        # Finetuned model's first forecast (day 7) represents avg[day 0-7]
        ds_init = ds_init.isel(week_lead_time=slice(1, None))    
        
        reforecast_datasets_finetuning.append(ds_init)
    
    if not reforecast_datasets_finetuning:
        print(f"Warning: No valid data loaded for {finetune_model}, skipping model...")
        continue

    # Concatenate all reforecasts along a new 'init_date' dimension
    ds_inf_weekly_finetuning = xr.concat(
        reforecast_datasets_finetuning, 
        dim=xr.DataArray(init_dates_finetuning, dims='init_date', name='init_date')
    )

    ds_inf_weekly_finetuning = ds_inf_weekly_finetuning[var_of_interest]
    
    # Transpose to match reference model dimension order: (week_lead_time, init_date, member, values)
    ds_inf_weekly_finetuning = ds_inf_weekly_finetuning.transpose('week_lead_time', 'init_date', 'member', 'values')
    
    finetune_models[finetune_model] = ds_inf_weekly_finetuning


##################################################################################################################################
#Create observations dataset with weekly means matching finetuned model structure
print("Creating weekly observations from 6-hourly data...")
# Finetuned model's week_lead_time=1 represents avg[day 0-7], week_lead_time=2 is avg[day 7-14], etc.
# We need to compute the average of ALL 6-hourly values in each week window

ds_obs_weekly_list = []

# Use the same init dates as the finetuned models (assuming all models have the same init dates)
# Get init dates from one of the finetuned models
sample_model = list(finetune_models.values())[0]
weekly_init_dates = sample_model.init_date.values

for init_date in weekly_init_dates:
    # Find the observation time range for this initialization
    t0 = np.where(times == init_date)[0][0]
    
    # We need 8 weeks of data (56 days + 1 for the endpoint = 57 days)
    # Each week is 7 days = 168 hours = 28 timesteps (at 6-hourly resolution)
    n_timesteps_per_week = 28  # 7 days * 4 timesteps per day
    n_weeks = 8
    total_timesteps = n_timesteps_per_week * n_weeks + 1  # +1 to include the right boundary
    
    t1 = t0 + total_timesteps
    
    # Extract the time slice
    ds_init = obs.isel(time=slice(t0, t1))
    ds_init = ds_init.squeeze("ensemble").assign_coords(variable=var_of_interest)
    
    # Compute weekly means for each week window
    week_means = []
    
    for week_num in range(1, n_weeks + 1):
        # Week 1: timesteps [0:29] (includes boundaries, 7 days = 28 intervals + 1)
        # Week 2: timesteps [28:57] (starts at end of week 1, includes next 7 days)
        start_idx = (week_num - 1) * n_timesteps_per_week
        end_idx = start_idx + n_timesteps_per_week + 1  # +1 to include right boundary
        
        week_slice = ds_init.isel(time=slice(start_idx, end_idx))
        
        # Process each variable separately
        var_week_means = []
        for var in var_of_interest:
            var_data = week_slice.sel(variable=var)['data']
            
            if var == "tp":
                # For precipitation: sum all 6-hourly values in the week
                var_week_mean = var_data.sum(dim='time')
            else:
                # For other variables: average all 6-hourly values in the week
                var_week_mean = var_data.mean(dim='time')
            
            var_week_means.append(var_week_mean)
        
        # Combine variables for this week
        week_data = xr.concat(var_week_means, dim=xr.DataArray(var_of_interest, dims='variable', name='variable'))
        week_means.append(week_data)
    
    # Stack all weeks for this init date
    obs_init = xr.concat(week_means, dim=xr.DataArray(range(1, n_weeks + 1), dims='week_lead_time', name='week_lead_time'))
    
    # Convert to Dataset with separate variables
    obs_init_ds = xr.Dataset({var: obs_init.sel(variable=var).drop_vars('variable') for var in var_of_interest})
    
    ds_obs_weekly_list.append(obs_init_ds)

# Concatenate all init dates
ds_obs_weekly = xr.concat(ds_obs_weekly_list, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))

print(f"Created weekly observations for {len(weekly_init_dates)} init dates")
print(f"Observation dataset dimensions: {dict(ds_obs_weekly.dims)}")


##################################################################################################################
# Create second ground truth: rolled observations matching daily finetuned model structure
print("Creating rolled observations matching daily finetuned model structure...")
print("Step 1: Apply rolling average to 6-hourly data to create daily rolled observations")
# The daily finetuned model was trained with rolling average window=(-4, 0, 'freq')
# This means each daily output at 12:00 is the average of 5 consecutive 6-hourly timesteps:
# - For day N at 12:00: average from (N-1)@12:00 through N@12:00 = [12:00(N-1), 18:00(N-1), 00:00(N), 06:00(N), 12:00(N)]
# - For day N+1 at 12:00: average from N@12:00 through (N+1)@12:00 = [12:00(N), 18:00(N), 00:00(N+1), 06:00(N+1), 12:00(N+1)]
# Note: 12:00(N) appears in both windows, creating the overlap we need to match
# IMPORTANT: The first output is at day 1 (not day 0) because we need 4 previous timesteps (24 hours)

ds_obs_weekly_from_daily_list = []

for init_date in weekly_init_dates:
    # Find the observation time range for this initialization
    t0 = np.where(times == init_date)[0][0]
    
    # We need 56 days of output (day 1 through day 56, no day 0) at 6-hourly resolution
    # Total 6-hourly timesteps needed: 56 days * 4 + 4 lookback = 228 timesteps
    n_days_output = 56  # Days 1-56 (no day 0 because rolling average needs lookback)
    n_timesteps_6hourly = (n_days_output + 1) * 4  # +1 day for the lookback window = 228 timesteps
    
    t1 = t0 + n_timesteps_6hourly
    
    # Extract the time slice
    ds_6hourly = obs.isel(time=slice(t0, t1))
    ds_6hourly = ds_6hourly.squeeze("ensemble").assign_coords(variable=var_of_interest)
    
    # Apply rolling average to create daily observations (one value per day at 12:00)
    # Window: (-4, 0, 'freq') means looking back 4 steps and forward 0 steps
    # i_start = 4, i_end = 1
    # First output is at input index 4 (day 1 at 12:00), which uses inputs [0:5]
    daily_rolled = []
    
    for day in range(1, n_days_output + 1):  # Start at day 1, not day 0
        # Day N at 12:00 is at 6-hourly index (N * 4)
        idx_12h = day * 4  # Index of 12:00 for this day
        
        # Rolling window: average 5 consecutive 6-hourly values ending at idx_12h
        # For day 1 (idx 4): indices [0:5] = day0@12h, day0@18h, day1@00h, day1@06h, day1@12h
        # For day 2 (idx 8): indices [4:9] = day1@12h, day1@18h, day2@00h, day2@06h, day2@12h
        idx_start = idx_12h - 4  # Look back 4 steps (24 hours)
        idx_end = idx_12h + 1    # Include current step
        # Rolling window: average 5 consecutive 6-hourly values ending at idx_12h
        # For day 1 (idx 4): indices [0:5] = day0@12h, day0@18h, day1@00h, day1@06h, day1@12h
        # For day 2 (idx 8): indices [4:9] = day1@12h, day1@18h, day2@00h, day2@06h, day2@12h
        idx_start = idx_12h - 4  # Look back 4 steps (24 hours)
        idx_end = idx_12h + 1    # Include current step
        
        # Extract the window for each variable
        var_daily_means = []
        for var in var_of_interest:
            var_data = ds_6hourly.sel(variable=var)['data'].isel(time=slice(idx_start, idx_end))
            
            if var == "tp":
                # For precipitation: sum the 5 6-hourly values (each is a 6h accumulation)
                # tp is accumulated and not affected by rolling average
                var_daily_rolled = var_data.sum(dim='time')
            else:
                # For other variables: average the 5 6-hourly values
                # These are affected by the rolling average transformation
                var_daily_rolled = var_data.mean(dim='time')
            
            var_daily_means.append(var_daily_rolled)
        
        # Combine variables for this day
        day_data = xr.concat(var_daily_means, dim=xr.DataArray(var_of_interest, dims='variable', name='variable'))
        daily_rolled.append(day_data)
    
    # Stack all days to create daily rolled observations (starting at day 1)
    # lead_time coordinates: 1, 2, 3, ..., 56
    daily_rolled_ds = xr.concat(
        daily_rolled, 
        dim=xr.DataArray(range(1, n_days_output + 1), dims='lead_time', name='lead_time')
    )
    
    # Convert to Dataset with separate variables
    daily_rolled_ds = xr.Dataset({var: daily_rolled_ds.sel(variable=var).drop_vars('variable') for var in var_of_interest})
    
    # Step 2: Compute weekly means from daily rolled observations
    print(f"Step 2: Computing weekly means from daily rolled observations for {pd.to_datetime(init_date).strftime('%Y-%m-%d')}...")
    
    # Week 1: days [1:8] (8 days total, including both boundaries for overlap)
    # Week 2: days [8:15] (8 days total, starts where week 1 ends)
    # This matches the overlapping structure of the daily finetuned model
    n_days_per_week = 7
    n_weeks = 8
    week_means_rolled = []
    
    for week_num in range(1, n_weeks + 1):
        # Week boundaries with overlap (note: we start at day 1, not day 0)
        start_day = 1 + (week_num - 1) * n_days_per_week  # Week 1: day 1, Week 2: day 8, etc.
        end_day = start_day + n_days_per_week             # Week 1: day 8, Week 2: day 15, etc.
        
        # Select days for this week using coordinate-based selection
        # slice(1, 8) selects lead_time values 1,2,3,4,5,6,7,8 (8 days)
        week_slice = daily_rolled_ds.sel(lead_time=slice(start_day, end_day))
        
        # Compute mean for each variable
        week_mean_dict = {}
        for var in var_of_interest:
            if var == "tp":
                # For precipitation: sum all daily values in the week
                week_mean_dict[var] = week_slice[var].sum(dim='lead_time')
            else:
                # For other variables: average all daily values in the week
                week_mean_dict[var] = week_slice[var].mean(dim='lead_time')
        
        week_mean_ds = xr.Dataset(week_mean_dict)
        week_means_rolled.append(week_mean_ds)
    
    # Stack all weeks for this init date
    obs_init_rolled = xr.concat(
        week_means_rolled, 
        dim=xr.DataArray(range(1, len(week_means_rolled) + 1), dims='week_lead_time', name='week_lead_time')
    )
    
    ds_obs_weekly_from_daily_list.append(obs_init_rolled)

# Concatenate all init dates
ds_obs_weekly_from_daily = xr.concat(
    ds_obs_weekly_from_daily_list, 
    dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date')
)

print(f"Created weekly observations from daily rolling average for {len(weekly_init_dates)} init dates")
print(f"Weekly from daily observation dataset dimensions: {dict(ds_obs_weekly_from_daily.dims)}")
print("This dataset matches the structure of the daily finetuned model (with rolling average overlap)")


##################################################################################################################
# Concatenate reference model weekly means
print("Finalizing reference model weekly dataset...")

if ds_inf_ref_weekly_list:
    # Concatenate all init dates
    ds_inf_weekly = xr.concat(
        ds_inf_ref_weekly_list, 
        dim=xr.DataArray(ref_init_dates_list, dims='init_date', name='init_date')
    )
    
    print(f"Reference model weekly dataset dimensions: {dict(ds_inf_weekly.dims)}")
else:
    print("Warning: No reference model weekly data created")
    ds_inf_weekly = None

##################################################################################################################
# Create weekly means for daily finetuned model
print("Creating weekly means for daily finetuned model...")
# The daily finetuned model has lead_time in days starting at day 1 (due to rolling average window=(-4, 0, 'freq'))
# Each daily value represents the rolling average of 5 6-hourly timesteps
# We need to compute weekly means with overlapping windows to match the rolled observations

ds_inf_daily_weekly_list = []

for init_date in weekly_init_dates:
    # Check if this init date is in the daily finetuned model
    if init_date not in ds_inf_daily_finetuned.init_date.values:
        print(f"Warning: Init date {init_date} not found in daily finetuned model, skipping...")
        continue
    
    # Get the daily forecast for this init date
    daily_forecast = ds_inf_daily_finetuned.sel(init_date=init_date)
    
    # Compute weekly means for each week window
    # Week 1: days [1:8] (8 days total, starting at day 1)
    # Week 2: days [8:15] (8 days total, overlap at day 8)
    # This matches the structure of the rolled observations
    n_days_per_week = 7
    n_weeks = 8
    
    week_means = []
    
    for week_num in range(1, n_weeks + 1):
        # Start at day 1, not day 0
        start_day = 1 + (week_num - 1) * n_days_per_week  # Week 1: day 1, Week 2: day 8, etc.
        end_day = start_day + n_days_per_week             # Week 1: day 8, Week 2: day 15, etc.
        
        # Select days for this week using coordinate-based selection
        week_slice = daily_forecast.sel(lead_time=slice(start_day, end_day))
        
        # Compute mean for each variable
        # For tp (precipitation), we sum instead of averaging
        week_mean_dict = {}
        for var in var_of_interest:
            if var == "tp":
                # For precipitation: sum all daily values in the week
                week_mean_dict[var] = week_slice[var].sum(dim='lead_time')
            else:
                # For other variables: average all daily values in the week
                week_mean_dict[var] = week_slice[var].mean(dim='lead_time')
        
        week_mean_ds = xr.Dataset(week_mean_dict)
        week_means.append(week_mean_ds)
    
    if week_means:
        # Stack all weeks for this init date
        daily_weekly = xr.concat(week_means, dim=xr.DataArray(range(1, len(week_means) + 1), dims='week_lead_time', name='week_lead_time'))
        ds_inf_daily_weekly_list.append(daily_weekly)

if ds_inf_daily_weekly_list:
    # Concatenate all init dates
    ds_inf_daily_weekly = xr.concat(
        ds_inf_daily_weekly_list, 
        dim=xr.DataArray([init_date for init_date in weekly_init_dates if init_date in ds_inf_daily_finetuned.init_date.values], 
                       dims='init_date', name='init_date')
    )
    
    print(f"Created weekly means from daily finetuned model for {len(ds_inf_daily_weekly_list)} init dates")
    print(f"Daily finetuned model weekly dataset dimensions: {dict(ds_inf_daily_weekly.dims)}")
else:
    print(f"Warning: No valid daily finetuned model weekly data created")
    ds_inf_daily_weekly = None




# Climatology computation for reference model and finetuned models
print("Computing climatology for all models...")
ds_clim_multiple_inf_init = []
ds_clim_multiple_inf_daily_init = []
ds_clim_multiple_obs_init = []
ds_clim_multiple_obs_from_daily_init = []
ds_clim_multiple_inf_finetuning_14k_e5_init = []
ds_clim_multiple_inf_finetuning_14k_e7_init = []
ds_clim_multiple_inf_finetuning_18k_e6_init = []

#compute climatology per member using weekly_init_dates
for init_date in weekly_init_dates:
    year = pd.to_datetime(init_date).year
    month = pd.to_datetime(init_date).month
    day = pd.to_datetime(init_date).day
    
    # Create mask for observations and models
    mask = (ds_obs_weekly.init_date.dt.month == month) & (ds_obs_weekly.init_date.dt.day == day) & (ds_obs_weekly.init_date.dt.year != year)
    mask_from_daily = (ds_obs_weekly_from_daily.init_date.dt.month == month) & (ds_obs_weekly_from_daily.init_date.dt.day == day) & (ds_obs_weekly_from_daily.init_date.dt.year != year)
    
    # Reference model climatology (if available)
    if ds_inf_weekly is not None and init_date in ds_inf_weekly.init_date.values:
        ref_mask = (ds_inf_weekly.init_date.dt.month == month) & (ds_inf_weekly.init_date.dt.day == day) & (ds_inf_weekly.init_date.dt.year != year)
        climatology_weekly_inf = ds_inf_weekly.isel(init_date=ref_mask).mean(dim="init_date")
        ds_clim_multiple_inf_init.append(climatology_weekly_inf)
    
    # Daily finetuned model climatology (if available)
    if ds_inf_daily_weekly is not None and init_date in ds_inf_daily_weekly.init_date.values:
        daily_mask = (ds_inf_daily_weekly.init_date.dt.month == month) & (ds_inf_daily_weekly.init_date.dt.day == day) & (ds_inf_daily_weekly.init_date.dt.year != year)
        climatology_weekly_inf_daily = ds_inf_daily_weekly.isel(init_date=daily_mask).mean(dim="init_date")
        ds_clim_multiple_inf_daily_init.append(climatology_weekly_inf_daily)
    
    #Same for finetuning model 14k_lr_0.625e-5
    if "Weekly_Means_14k_lr_0.625e-5" in finetune_models:
        climatology_weekly_inf_finetuning_14k_e5 = finetune_models["Weekly_Means_14k_lr_0.625e-5"].isel(init_date=mask).mean(dim="init_date")
        ds_clim_multiple_inf_finetuning_14k_e5_init.append(climatology_weekly_inf_finetuning_14k_e5)
    
    #Same for finetuning model 14k_lr_0.625e-7
    if "Weekly_Means_14k_lr_0.625e-7" in finetune_models:
        climatology_weekly_inf_finetuning_14k_e7 = finetune_models["Weekly_Means_14k_lr_0.625e-7"].isel(init_date=mask).mean(dim="init_date")
        ds_clim_multiple_inf_finetuning_14k_e7_init.append(climatology_weekly_inf_finetuning_14k_e7)
    
    #Same for finetuning model 18k_lr_0.625e-6
    if "Weekly_Means_18k_lr_0.625e-6" in finetune_models:
        climatology_weekly_inf_finetuning_18k_e6 = finetune_models["Weekly_Means_18k_lr_0.625e-6"].isel(init_date=mask).mean(dim="init_date")
        ds_clim_multiple_inf_finetuning_18k_e6_init.append(climatology_weekly_inf_finetuning_18k_e6)

    
    #build climatology for observations
    climatology_weekly_obs = ds_obs_weekly.isel(init_date=mask).mean(dim="init_date")
    ds_clim_multiple_obs_init.append(climatology_weekly_obs)
    
    #build climatology for observations matching structure of daily finetuned model
    climatology_weekly_obs_from_daily = ds_obs_weekly_from_daily.isel(init_date=mask_from_daily).mean(dim="init_date")
    ds_clim_multiple_obs_from_daily_init.append(climatology_weekly_obs_from_daily)

# Concatenate climatologies for reference model (if available)
if ds_inf_weekly is not None and ds_clim_multiple_inf_init:
    # Get the init dates that are actually in the reference model
    ref_init_dates = [init_date for init_date in weekly_init_dates if init_date in ds_inf_weekly.init_date.values]
    ds_climatology_weekly_inf = xr.concat(ds_clim_multiple_inf_init, dim=xr.DataArray(ref_init_dates, dims='init_date', name='init_date'))
else:
    ds_climatology_weekly_inf = None

# Concatenate climatologies for daily finetuned model (if available)
if ds_inf_daily_weekly is not None and ds_clim_multiple_inf_daily_init:
    # Get the init dates that are actually in the daily model
    daily_init_dates = [init_date for init_date in weekly_init_dates if init_date in ds_inf_daily_weekly.init_date.values]
    ds_climatology_weekly_inf_daily = xr.concat(ds_clim_multiple_inf_daily_init, dim=xr.DataArray(daily_init_dates, dims='init_date', name='init_date'))
else:
    ds_climatology_weekly_inf_daily = None

# Only concat if models were loaded
if "Weekly_Means_14k_lr_0.625e-5" in finetune_models:
    ds_climatology_weekly_inf_finetuning_14k_e5 = xr.concat(ds_clim_multiple_inf_finetuning_14k_e5_init, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))
if "Weekly_Means_14k_lr_0.625e-7" in finetune_models:
    ds_climatology_weekly_inf_finetuning_14k_e7 = xr.concat(ds_clim_multiple_inf_finetuning_14k_e7_init, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))
if "Weekly_Means_18k_lr_0.625e-6" in finetune_models:
    ds_climatology_weekly_inf_finetuning_18k_e6 = xr.concat(ds_clim_multiple_inf_finetuning_18k_e6_init, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))

ds_climatology_weekly_obs = xr.concat(ds_clim_multiple_obs_init, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))
ds_climatology_weekly_obs_from_daily = xr.concat(ds_clim_multiple_obs_from_daily_init, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))

print(f"Computed climatology for {len(weekly_init_dates)} init dates")
print(f"  Reference model: {len(ds_clim_multiple_inf_init)} climatologies")
print(f"  Daily model: {len(ds_clim_multiple_inf_daily_init)} climatologies")
print(f"  Observations: {len(ds_clim_multiple_obs_init)} climatologies")
print(f"  Observations from daily: {len(ds_clim_multiple_obs_from_daily_init)} climatologies")

#observed anomalies
print("Computing anomalies for all models...")
observed_anomalies = ds_obs_weekly - ds_climatology_weekly_obs
observed_anomalies_from_daily = ds_obs_weekly_from_daily - ds_climatology_weekly_obs_from_daily

# Reference model anomalies (if available)
if ds_inf_weekly is not None and ds_climatology_weekly_inf is not None:
    print(f"Computing reference model anomalies...")
    print(f"  ds_inf_weekly shape: {ds_inf_weekly.dims}")
    print(f"  ds_climatology_weekly_inf shape: {ds_climatology_weekly_inf.dims}")
    predicted_anomalies = ds_inf_weekly - ds_climatology_weekly_inf
    print(f"  Reference model anomalies computed successfully")
else:
    print(f"Warning: Cannot compute reference model anomalies")
    print(f"  ds_inf_weekly is None: {ds_inf_weekly is None}")
    print(f"  ds_climatology_weekly_inf is None: {ds_climatology_weekly_inf is None}")
    predicted_anomalies = None

# Daily finetuned model anomalies (if available)
if ds_inf_daily_weekly is not None and ds_climatology_weekly_inf_daily is not None:
    predicted_anomalies_daily = ds_inf_daily_weekly - ds_climatology_weekly_inf_daily
else:
    predicted_anomalies_daily = None

#predicted anomalies for finetuning models (only if they were loaded)
if "Weekly_Means_14k_lr_0.625e-5" in finetune_models:
    predicted_anomalies_finetuning_14k_e5 = finetune_models["Weekly_Means_14k_lr_0.625e-5"] - ds_climatology_weekly_inf_finetuning_14k_e5
else:
    predicted_anomalies_finetuning_14k_e5 = None

if "Weekly_Means_14k_lr_0.625e-7" in finetune_models:
    predicted_anomalies_finetuning_14k_e7 = finetune_models["Weekly_Means_14k_lr_0.625e-7"] - ds_climatology_weekly_inf_finetuning_14k_e7
else:
    predicted_anomalies_finetuning_14k_e7 = None

if "Weekly_Means_18k_lr_0.625e-6" in finetune_models:
    predicted_anomalies_finetuning_18k_e6 = finetune_models["Weekly_Means_18k_lr_0.625e-6"] - ds_climatology_weekly_inf_finetuning_18k_e6
else:
    predicted_anomalies_finetuning_18k_e6 = None

    
#Compute thresholds for forecast using a finetuned model
print("Computing thresholds...")
# (reference model is excluded as it requires 6-hourly data)
if finetune_models:
    sample_model = list(finetune_models.values())[0]
    thresholds_per_init = []
    
    for init_date in sample_model.init_date.values:
        month = pd.to_datetime(init_date).month
        day = pd.to_datetime(init_date).day
        mask = (sample_model.init_date.dt.month == month) & (sample_model.init_date.dt.day == day)
        #thresholds for forecast
        threshold_95 = sample_model.isel(init_date=mask).quantile(0.95, dim=['init_date', 'member'])
        thresholds_per_init.append(threshold_95)
    
    ds_thresholds = xr.concat(thresholds_per_init, dim=xr.DataArray(weekly_init_dates, dims='init_date', name='init_date'))
else:
    print("Warning: No finetuned models loaded, cannot compute thresholds")
    ds_thresholds = None

############################################################################################################################################################
# Export preprocessed data
print("=" * 80)
print("EXPORTING PREPROCESSED DATA")
print("=" * 80)

output_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Add lat/lon coordinates
ds_obs_weekly = ds_obs_weekly.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)

ds_obs_weekly_from_daily = ds_obs_weekly_from_daily.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)

observed_anomalies = observed_anomalies.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)

observed_anomalies_from_daily = observed_anomalies_from_daily.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)

ds_climatology_weekly_obs = ds_climatology_weekly_obs.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)

ds_climatology_weekly_obs_from_daily = ds_climatology_weekly_obs_from_daily.assign_coords(
    latitude=("values",lat_lon_coords["latitude"].values),
    longitude=("values", lat_lon_coords["longitude"].values),)

# Export observations
obs_path = os.path.join(output_dir, "observations_weekly.nc")
if os.path.exists(obs_path):
    os.remove(obs_path)
print(f"Saving observations to {obs_path}")
ds_obs_weekly.to_netcdf(obs_path, mode='w')

obs_anom_path = os.path.join(output_dir, "observations_anomalies_weekly.nc")
if os.path.exists(obs_anom_path):
    os.remove(obs_anom_path)
print(f"Saving observation anomalies to {obs_anom_path}")
observed_anomalies.to_netcdf(obs_anom_path, mode='w')

obs_clim_path = os.path.join(output_dir, "observations_climatology_weekly.nc")
if os.path.exists(obs_clim_path):
    os.remove(obs_clim_path)
print(f"Saving observation climatology to {obs_clim_path}")
ds_climatology_weekly_obs.to_netcdf(obs_clim_path, mode='w')

# Export observations from daily (for comparison with daily finetuned model)
print(f"Exporting observations from daily (matching daily finetuned model structure)...")
obs_from_daily_path = os.path.join(output_dir, "observations_weekly_from_daily.nc")
if os.path.exists(obs_from_daily_path):
    os.remove(obs_from_daily_path)
print(f"Saving observations from daily to {obs_from_daily_path}")
ds_obs_weekly_from_daily.to_netcdf(obs_from_daily_path, mode='w')

obs_from_daily_anom_path = os.path.join(output_dir, "observations_anomalies_weekly_from_daily.nc")
if os.path.exists(obs_from_daily_anom_path):
    os.remove(obs_from_daily_anom_path)
print(f"Saving observation anomalies from daily to {obs_from_daily_anom_path}")
observed_anomalies_from_daily.to_netcdf(obs_from_daily_anom_path, mode='w')

obs_clim_from_daily_path = os.path.join(output_dir, "observations_climatology_weekly_from_daily.nc")
if os.path.exists(obs_clim_from_daily_path):
    os.remove(obs_clim_from_daily_path)
print(f"Saving rolled observation climatology to {obs_clim_from_daily_path}")
ds_climatology_weekly_obs_from_daily.to_netcdf(obs_clim_from_daily_path, mode='w')

# Export reference model (if available)
if ds_inf_weekly is not None:
    print(f"Exporting reference model data...")
    ds_inf_weekly = ds_inf_weekly.assign_coords(
        latitude=("values",lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values),)
    ref_path = os.path.join(output_dir, "reference_model_weekly.nc")
    if os.path.exists(ref_path):
        os.remove(ref_path)
    print(f"Saving reference model to {ref_path}")
    ds_inf_weekly.to_netcdf(ref_path, mode='w')
    
    if predicted_anomalies is not None:
        print(f"Exporting reference model anomalies...")
        predicted_anomalies = predicted_anomalies.assign_coords(
            latitude=("values",lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values),)
        ref_anom_path = os.path.join(output_dir, "reference_model_anomalies_weekly.nc")
        if os.path.exists(ref_anom_path):
            os.remove(ref_anom_path)
        print(f"Saving reference model anomalies to {ref_anom_path}")
        predicted_anomalies.to_netcdf(ref_anom_path, mode='w')
    else:
        print(f"Warning: Reference model anomalies not computed (likely insufficient init dates for climatology)")
    
    if ds_climatology_weekly_inf is not None:
        print(f"Exporting reference model climatology...")
        ds_climatology_weekly_inf = ds_climatology_weekly_inf.assign_coords(
            latitude=("values",lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values),)
        ref_clim_path = os.path.join(output_dir, "reference_model_climatology_weekly.nc")
        if os.path.exists(ref_clim_path):
            os.remove(ref_clim_path)
        print(f"Saving reference model climatology to {ref_clim_path}")
        ds_climatology_weekly_inf.to_netcdf(ref_clim_path, mode='w')
    else:
        print(f"Warning: Reference model climatology not computed (likely insufficient init dates)")

# Export daily finetuned model (if available)
if ds_inf_daily_weekly is not None:
    ds_inf_daily_weekly = ds_inf_daily_weekly.assign_coords(
        latitude=("values",lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values),)
    daily_path = os.path.join(output_dir, "daily_finetuned_model_weekly.nc")
    if os.path.exists(daily_path):
        os.remove(daily_path)
    print(f"Saving daily finetuned model to {daily_path}")
    ds_inf_daily_weekly.to_netcdf(daily_path, mode='w')
    
    if predicted_anomalies_daily is not None:
        predicted_anomalies_daily = predicted_anomalies_daily.assign_coords(
            latitude=("values",lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values),)
        daily_anom_path = os.path.join(output_dir, "daily_finetuned_model_anomalies_weekly.nc")
        if os.path.exists(daily_anom_path):
            os.remove(daily_anom_path)
        print(f"Saving daily finetuned model anomalies to {daily_anom_path}")
        predicted_anomalies_daily.to_netcdf(daily_anom_path, mode='w')
    
    if ds_climatology_weekly_inf_daily is not None:
        ds_climatology_weekly_inf_daily = ds_climatology_weekly_inf_daily.assign_coords(
            latitude=("values",lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values),)
        daily_clim_path = os.path.join(output_dir, "daily_finetuned_model_climatology_weekly.nc")
        if os.path.exists(daily_clim_path):
            os.remove(daily_clim_path)
        print(f"Saving daily finetuned model climatology to {daily_clim_path}")
        ds_climatology_weekly_inf_daily.to_netcdf(daily_clim_path, mode='w')

# Export weekly finetuned models
for model_name in ["Weekly_Means_14k_lr_0.625e-5", "Weekly_Means_14k_lr_0.625e-7", "Weekly_Means_18k_lr_0.625e-6"]:
    if model_name in finetune_models:
        model_data = finetune_models[model_name].assign_coords(
            latitude=("values",lat_lon_coords["latitude"].values),
            longitude=("values", lat_lon_coords["longitude"].values),)
        model_path = os.path.join(output_dir, f"{model_name}_weekly.nc")
        if os.path.exists(model_path):
            os.remove(model_path)
        print(f"Saving {model_name} to {model_path}")
        model_data.to_netcdf(model_path, mode='w')
        
        # Save anomalies
        if model_name == "Weekly_Means_14k_lr_0.625e-5" and predicted_anomalies_finetuning_14k_e5 is not None:
            anom_data = predicted_anomalies_finetuning_14k_e5
        elif model_name == "Weekly_Means_14k_lr_0.625e-7" and predicted_anomalies_finetuning_14k_e7 is not None:
            anom_data = predicted_anomalies_finetuning_14k_e7
        elif model_name == "Weekly_Means_18k_lr_0.625e-6" and predicted_anomalies_finetuning_18k_e6 is not None:
            anom_data = predicted_anomalies_finetuning_18k_e6
        else:
            anom_data = None
        
        if anom_data is not None:
            anom_data = anom_data.assign_coords(
                latitude=("values",lat_lon_coords["latitude"].values),
                longitude=("values", lat_lon_coords["longitude"].values),)
            anom_path = os.path.join(output_dir, f"{model_name}_anomalies_weekly.nc")
            if os.path.exists(anom_path):
                os.remove(anom_path)
            print(f"Saving {model_name} anomalies to {anom_path}")
            anom_data.to_netcdf(anom_path, mode='w')
        
        # Save climatology
        if model_name == "Weekly_Means_14k_lr_0.625e-5" and ds_climatology_weekly_inf_finetuning_14k_e5 is not None:
            clim_data = ds_climatology_weekly_inf_finetuning_14k_e5
        elif model_name == "Weekly_Means_14k_lr_0.625e-7" and ds_climatology_weekly_inf_finetuning_14k_e7 is not None:
            clim_data = ds_climatology_weekly_inf_finetuning_14k_e7
        elif model_name == "Weekly_Means_18k_lr_0.625e-6" and ds_climatology_weekly_inf_finetuning_18k_e6 is not None:
            clim_data = ds_climatology_weekly_inf_finetuning_18k_e6
        else:
            clim_data = None
        
        if clim_data is not None:
            clim_data = clim_data.assign_coords(
                latitude=("values",lat_lon_coords["latitude"].values),
                longitude=("values", lat_lon_coords["longitude"].values),)
            clim_path = os.path.join(output_dir, f"{model_name}_climatology_weekly.nc")
            if os.path.exists(clim_path):
                os.remove(clim_path)
            print(f"Saving {model_name} climatology to {clim_path}")
            clim_data.to_netcdf(clim_path, mode='w')

# Export thresholds
if ds_thresholds is not None:
    ds_thresholds = ds_thresholds.assign_coords(
        latitude=("values",lat_lon_coords["latitude"].values),
        longitude=("values", lat_lon_coords["longitude"].values),)
    thresh_path = os.path.join(output_dir, "thresholds_95.nc")
    if os.path.exists(thresh_path):
        os.remove(thresh_path)
    print(f"Saving thresholds to {thresh_path}")
    ds_thresholds.to_netcdf(thresh_path, mode='w')

# Export lat_weights for use in metrics computation
lat_weights_path = os.path.join(output_dir, "lat_weights.nc")
if os.path.exists(lat_weights_path):
    os.remove(lat_weights_path)
print(f"Saving latitude weights to {lat_weights_path}")
lat_weights_ds = xr.Dataset({'lat_weights': lat_weights})
lat_weights_ds.to_netcdf(lat_weights_path, mode='w')

print("=" * 80)
print("PREPROCESSING COMPLETE")
print(f"All preprocessed data saved to: {output_dir}")
print("=" * 80)
