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
from xskillscore import roc, brier_score, reliability
warnings.filterwarnings('ignore')


# Add paths
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')

  
################################################################################################################################  
#Load datasets

#load dataset with observations
print("Loading observations...", flush=True)
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-6h-v1-for-single-v2.zarr"
# Use consolidated=False to avoid the slow fallback warning
obs = xr.open_zarr(dataset_path, consolidated=False)
times = obs.dates.values
obs = obs.assign_coords(time=times).rename({"cell": "values"})

#select only a subset of variable
var_names = obs.attrs["variables"]
var_of_interest = ['2t']
var_indices = [var_names.index(var) for var in var_of_interest]
obs = obs.isel(variable=var_indices)["data"].squeeze("ensemble").assign_coords(variable=var_of_interest)

# Load preprocessed data
preprocessed_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init"

# Load latitude weights
print("Loading latitude weights...")
lat_weights_ds = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed/lat_weights.nc")
lat_weights = lat_weights_ds['lat_weights']

# Load observations
print("Loading observations...")
ds_obs = xr.open_dataset(os.path.join(preprocessed_dir, "observations_weekly_27_0.nc"))
ds_obs_anomalies = xr.open_zarr(os.path.join("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/observations_anomalies_weekly_27_0_ROBUST.zarr"))
ds_obs_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "observations_climatology_weekly_27_0.nc"))

# Get lat/lon coordinates from observations
lat_lon_coords = {
    'latitude': ds_obs['latitude'],
    'longitude': ds_obs['longitude']
}

weekly_init_dates = ds_obs.init_date.values

# Load reference model (if available)
print("Loading reference model...")
ref_path = os.path.join(preprocessed_dir, "reference_model_weekly_27_0.nc")
ds_ref = xr.open_dataset(ref_path)
ds_ref_anomalies = xr.open_zarr(os.path.join("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/reference_model_anomalies_weekly_27_0_ROBUST.zarr"))
ds_ref_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "reference_model_climatology_weekly_27_0.nc"))

#Load weekly finetuned model
print("Loading weekly finetuned model...")
ds_weekly_finetuned = xr.open_dataset(os.path.join(preprocessed_dir, "weekly_finetuned_model_27_0.nc"))
ds_weekly_finetuned_anomalies = xr.open_zarr(os.path.join( "/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/weekly_finetuned_model_anomalies_27_0_ROBUST.zarr"))
ds_weekly_finetuned_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "weekly_finetuned_model_climatology_27_0.nc"))

#Load daily finetuned model
print("Loading daily finetuned model...")
ds_daily_finetuned = xr.open_dataset(os.path.join(preprocessed_dir, "daily_finetuned_model_weekly_3_0.nc"))
ds_daily_finetuned_anomalies = xr.open_zarr(os.path.join("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_preprocessing/daily_finetuned_model_weekly_anomalies_3_0_ROBUST.zarr"))
ds_daily_finetuned_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "daily_finetuned_model_weekly_climatology_3_0.nc"))


SURFACE_VARS = ['2t', '2d', 'msl', 'sp', 'skt', 'tp', '10u', '10v']
UPPER_VARS = ['t_500', 't_700', 't_850', 't_925', 'q_500', 'q_700', 'q_850', 'q_925']
ALL_VARS = SURFACE_VARS + UPPER_VARS

# Heat wave definitions (standard deviation thresholds for temperature anomalies)
# Based on: weekly mean temperature anomaly > N standard deviations
HW_THRESHOLDS = {
    'moderate': 1.0,   # > 1 std dev
    'severe': 1.25,    # > 1.25 std dev
    'extreme': 1.5     # > 1.5 std dev
}

OUTPUT_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS-75-init/heat_wave_analysis_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

lat = ds_obs.latitude.values
lon = ds_obs.longitude.values

lon_adjusted = np.where(lon <= 180, lon, lon - 360)
region_lat_mask = (lat >= 20) & (lat <= 80)
region_lon_mask = (lon_adjusted >= -60) & (lon_adjusted <= 90)
region_mask = region_lat_mask & region_lon_mask
europe_lat_mask = (lat >= 35) & (lat <= 70)
europe_lon_mask = (lon_adjusted >= -10) & (lon_adjusted <= 40)
europe_mask = europe_lat_mask & europe_lon_mask
europe_mask_da = xr.DataArray(europe_mask, dims=["values"])
print(f"    Total grid points: {len(lat) * len(lon)}")
print(f"    Europe grid points: {europe_mask.sum().item()}")

################################################################################################################################  
#Compute standard deviation of temperature anomalies for heat wave thresholds
print("Computing standard deviation of temperature anomalies for heat wave definition...")
obs_t2 = ds_obs_anomalies['2t']
obs_stacked = obs_t2.stack(time_point=['init_date', 'week_lead_time'])
# Compute valid date for each (init_date, week_lead_time) pair
valid_dates = []
for init_date_val, week_lead_val in obs_stacked.time_point.values:
    valid_date = pd.Timestamp(init_date_val) + pd.Timedelta(days=int(week_lead_val) * 7)
    valid_dates.append(valid_date)
    
# Assign valid_date as coordinate
obs_stacked = obs_stacked.assign_coords(valid_date=('time_point', valid_dates))
# Group by valid_date and take first occurrence (removes duplicates)
obs_unique = obs_stacked.groupby('valid_date').first()

print(f"  Summary:")
print(f"  Original obs shape: {len(obs_t2.init_date)} init_dates × {len(obs_t2.week_lead_time)} lead_times = {len(obs_t2.init_date) * len(obs_t2.week_lead_time)} points")
print(f"  Unique valid dates: {len(obs_unique.valid_date)}")
print(f"  Reduction: {len(obs_t2.init_date) * len(obs_t2.week_lead_time) - len(obs_unique.valid_date)} points removed")

# compute LOYO std for each init_date
obs_std_list = []
for init_date in ds_obs_anomalies.init_date.values:
    year = pd.Timestamp(init_date).year
    
    # Mask for valid dates not in target year
    mask = obs_unique.valid_date.dt.year != year
    
    # Observations: std dev per grid point over unique valid dates, excluding this year
    obs_std_loyo = obs_unique.sel(valid_date=mask).std(dim='valid_date')
    
    obs_std_list.append(obs_std_loyo)

# Concatenate to create std dev for each init_date
obs_std = xr.concat(obs_std_list, dim=ds_obs_anomalies.init_date)

print(f"  Observed temperature anomaly std dev (global mean): {float(obs_std.mean().compute()):.2f} K")
print(f"  Observed temperature anomaly std dev (range): {float(obs_std.min().compute()):.2f} to {float(obs_std.max().compute()):.2f} K")
print(f"  Note: Each init date uses leave-one-year-out std dev to avoid circular threshold inflation")
print(f"  Note: Computed from unique valid dates (avoids counting same observation multiple times)")

thresholds = {}

for severity, std_threshold in HW_THRESHOLDS.items():
    print(f"  {severity.capitalize()}: > {std_threshold} std dev...")
    thresholds[severity] = obs_std * std_threshold

################################################################################################################################  
# NEW THRESHOLD COMPUTATION APPROACH
print("\n" + "="*80)
print("COMPUTING NEW THRESHOLDS FROM NON-OVERLAPPING CALENDAR WEEKS")
print("="*80)

# 1. Extract observations from May 2007 - September 2011
print("\n1. Extracting observations from May 2007 to September 2011...")
obs_subset = obs.sel(time=slice('2007-05-01', '2011-09-30'))



# 2. Aggregate to weekly means using ISO calendar weeks
print("\n2. Aggregating to non-overlapping ISO calendar weeks (May-September only)...")

# Convert to pandas for easier manipulation
times_pd = pd.to_datetime(obs_subset.time.values)

# Get ISO calendar info
iso_cal = times_pd.isocalendar()
iso_years = iso_cal.year.values
iso_weeks = iso_cal.week.values
iso_months = times_pd.month.values

# Filter to May-September only
may_sept_mask = (iso_months >= 5) & (iso_months <= 9)

# Create week identifiers for filtered data
week_ids_all = [f"{year}_W{week:02d}" if may_sept else None 
                for year, week, may_sept in zip(iso_years, iso_weeks, may_sept_mask)]
week_ids_filtered = [wid for wid in week_ids_all if wid is not None]
unique_week_ids = sorted(set(week_ids_filtered))

print(f"   Total unique ISO weeks (May-Sep, 2007-2011): {len(unique_week_ids)}")
print(f"   Sample week IDs: {unique_week_ids[:5]}...")

# Compute weekly means
weekly_means_list = []
week_id_coords = []

for week_id in unique_week_ids:
    # Get all timesteps for this week
    week_mask = np.array(week_ids_all) == week_id
    
    # Select and average over time
    week_data = obs_subset.isel(time=week_mask).mean(dim='time')
    weekly_means_list.append(week_data)
    week_id_coords.append(week_id)

# Concatenate into a single DataArray
obs_weekly_new = xr.concat(weekly_means_list, dim='week_id')
obs_weekly_new = obs_weekly_new.assign_coords(week_id=week_id_coords)

print(f"   Weekly means shape: {obs_weekly_new.shape}")
print(f"   Week ID range: {obs_weekly_new.week_id.values[0]} to {obs_weekly_new.week_id.values[-1]}")

# 3. Compute climatology with leave-one-year-out and ±1 week buffer
print("\n3. Computing leave-one-year-out climatology with ±1 week buffer...")

# Parse week_ids into year and week components
week_info = []
for wid in obs_weekly_new.week_id.values:
    year_str, week_str = wid.split('_W')
    week_info.append({'week_id': wid, 'year': int(year_str), 'week': int(week_str)})

week_info_df = pd.DataFrame(week_info)

# Compute climatology for each week
clim_list = []
clim_week_ids = []

for idx, row in week_info_df.iterrows():
    target_year = row['year']
    target_week = row['week']
    target_week_id = row['week_id']
    
    # Select weeks from other years within ±1 week window
    other_years_mask = week_info_df['year'] != target_year
    week_range_mask = ((week_info_df['week'] >= target_week - 1) & 
                      (week_info_df['week'] <= target_week + 1))
    clim_mask = other_years_mask & week_range_mask
    
    clim_week_ids_for_target = week_info_df[clim_mask]['week_id'].values
    
    if len(clim_week_ids_for_target) > 0:
        # Compute climatology as mean over these weeks
        clim_data = obs_weekly_new.sel(week_id=clim_week_ids_for_target).mean(dim='week_id')
        clim_list.append(clim_data)
        clim_week_ids.append(target_week_id)
    else:
        # If no climatology available (edge cases), use the observation itself
        print(f"   Warning: No climatology data for {target_week_id}, using observation")
        clim_list.append(obs_weekly_new.sel(week_id=target_week_id))
        clim_week_ids.append(target_week_id)

# Concatenate climatology
obs_climatology_new = xr.concat(clim_list, dim='week_id')
obs_climatology_new = obs_climatology_new.assign_coords(week_id=clim_week_ids)

print(f"   Climatology shape: {obs_climatology_new.shape}")

# Show sample sizes for a few weeks
sample_sizes = []
for wid in week_info_df['week_id'].values[:3]:
    target_row = week_info_df[week_info_df['week_id'] == wid].iloc[0]
    target_year = target_row['year']
    target_week = target_row['week']
    n_samples = len(week_info_df[(week_info_df['year'] != target_year) & 
                                  (week_info_df['week'] >= target_week - 1) & 
                                  (week_info_df['week'] <= target_week + 1)])
    sample_sizes.append(n_samples)

print(f"   Sample climatology sizes (first 3 weeks): {sample_sizes}")

# 4. Compute anomalies
print("\n4. Computing anomalies...")
obs_anomalies_new = obs_weekly_new - obs_climatology_new

print(f"   Anomalies shape: {obs_anomalies_new.shape}")
print(f"   Anomalies mean: {float(obs_anomalies_new.mean().compute()):.4f} K (should be ~0)")
print(f"   Anomalies std: {float(obs_anomalies_new.std().compute()):.4f} K")

# 5. Compute standard deviation for thresholds using LOYO
print("\n5. Computing standard deviation for heat wave thresholds (LOYO)...")

# Compute LOYO std for each year in the dataset
obs_std_new_list = []
years_in_data = sorted(set(week_info_df['year'].values))

for target_year in years_in_data:
    # Mask for weeks not in target year
    mask = week_info_df['year'] != target_year
    week_ids_excl_year = week_info_df[mask]['week_id'].values
    
    # Compute std dev over these weeks (excluding target year)
    std_loyo = obs_anomalies_new.sel(week_id=week_ids_excl_year).std(dim='week_id')
    obs_std_new_list.append(std_loyo)

# Average across LOYO iterations to get a single std dev field
obs_std_new = xr.concat(obs_std_new_list, dim='loyo_year').mean(dim='loyo_year')

print(f"   New std dev (global mean): {float(obs_std_new.mean().compute()):.2f} K")
print(f"   New std dev (range): {float(obs_std_new.min().compute()):.2f} to {float(obs_std_new.max().compute()):.2f} K")
print(f"")
print(f"   Comparison with old method:")
print(f"     Old std dev (global mean): {float(obs_std.mean().compute()):.2f} K")
print(f"     Difference: {float((obs_std_new.mean() - obs_std.mean()).compute()):.2f} K")

# Compute new thresholds (single field applied to all dates)
thresholds_new = {}
for severity, std_threshold in HW_THRESHOLDS.items():
    thresholds_new[severity] = obs_std_new * std_threshold
    print(f"   {severity.capitalize()}: > {std_threshold} std dev")

print("\n" + "="*80)
print("NEW THRESHOLD COMPUTATION COMPLETE")
print("Note: thresholds_new provides spatially-varying but temporally-constant thresholds")
print("Note: Original thresholds vary by init_date; new thresholds are fixed for all dates")
print("="*80)
    


###################################################################################################################################
#DETECT HEAT WAVES (NEW THRESHOLDS)

print("\n" + "="*80)
print("DETECTING HEAT WAVES WITH NEW THRESHOLDS (spatially-varying, temporally-constant)")
print("="*80)

hw_obs_new = {}
hw_ref_new = {}
hw_weekly_finetuned_new = {}
hw_daily_finetuned_new = {}

hw_models_new = [hw_ref_new, hw_weekly_finetuned_new, hw_daily_finetuned_new]

anomalies_list = [ds_ref_anomalies, ds_weekly_finetuned_anomalies, ds_daily_finetuned_anomalies]
model_names = ['Reference', 'Weekly Finetuned', 'Daily Finetuned']

for model, anomalies, name in zip(hw_models_new, anomalies_list, model_names):
    for severity in HW_THRESHOLDS.keys():
        print(f"  {severity.capitalize()} heat waves...")

        # Binary detection using NEW threshold (no init_date dimension)
        # Squeeze variable dimension immediately to avoid slowdowns in subsequent operations
        hw_obs_new[severity] = (ds_obs_anomalies['2t'] > thresholds_new[severity]).astype(int).squeeze('variable')
        hw_members = (anomalies['2t'] > thresholds_new[severity]).astype(int)
        model[severity] = hw_members.mean(dim='member')


# ============================================================================
# PROBABILISTIC METRICS FOR HEAT WAVE PREDICTION (NEW THRESHOLDS)
# ============================================================================

print("\n" + "="*80)
print("COMPUTING PROBABILISTIC METRICS WITH NEW THRESHOLDS")
print("="*80)

ref_metrics_by_severity_new = {}
ref_metrics_spatial_new = {}
ref_roc_by_severity_new = {}

weekly_finetuned_metrics_by_severity_new = {}
weekly_finetuned_metrics_spatial_new = {}
weekly_finetuned_roc_by_severity_new = {}

daily_finetuned_metrics_by_severity_new = {} 
daily_finetuned_metrics_spatial_new = {}
daily_finetuned_roc_by_severity_new = {}

list_of_metrics_by_severity_new = [ref_metrics_by_severity_new, weekly_finetuned_metrics_by_severity_new, daily_finetuned_metrics_by_severity_new]
list_of_metrics_spatial_new = [ref_metrics_spatial_new, weekly_finetuned_metrics_spatial_new, daily_finetuned_metrics_spatial_new]
list_of_roc_by_severity_new = [ref_roc_by_severity_new, weekly_finetuned_roc_by_severity_new, daily_finetuned_roc_by_severity_new]

for hw, metrics_by_severity, metrics_spatial in zip(hw_models_new, list_of_metrics_by_severity_new, list_of_metrics_spatial_new):
    for severity in HW_THRESHOLDS.keys():
        print(f"\n{severity.upper()} HEAT WAVES:")
        print("-" * 80)

        # Extract data for Europe only (variable already squeezed during detection)
        obs_europe = hw_obs_new[severity].where(europe_mask_da, drop=False)
        prob_europe = hw[severity].where(europe_mask_da, drop=False)

        # Compute climatological frequency ONCE (same for all lead times since we average over both dims)
        clim_freq_spatial = obs_europe.mean(dim=['init_date', 'week_lead_time'])
        clim_freq_spatial_masked = clim_freq_spatial.where(europe_mask_da, drop=True)
        n_invalid = (~np.isfinite(clim_freq_spatial_masked)).sum()
        print(f"  Climatology invalid values (Europe only): {n_invalid}")
        bs_clim_spatial = clim_freq_spatial * (1 - clim_freq_spatial)

        # Pre-compute flattened data ONCE for all lead times (more efficient than computing 8 times)
        # Drop non-Europe points and stack dimensions
        obs_europe_valid = obs_europe.where(europe_mask_da, drop=True)
        prob_europe_valid = prob_europe.where(europe_mask_da, drop=True)
        obs_flat_all = obs_europe_valid.stack(sample=['init_date', 'values']).compute()
        prob_flat_all = prob_europe_valid.stack(sample=['init_date', 'values']).compute()
        
        metrics_lead = []
        spatial_data = []  # Store spatial fields for each lead time

        for week_lead in range(1, 9):
            print(f"  Lead time: {week_lead} weeks")

            # Select this lead time
            obs_lead = obs_europe.sel(week_lead_time=week_lead)
            prob_lead = prob_europe.sel(week_lead_time=week_lead)

            # === PER-GRID-POINT COMPUTATION (for spatial maps) ===
            # Compute BS per grid point (over init_date dimension)
            bs_spatial = ((prob_lead - obs_lead) ** 2).mean(dim='init_date')

            # Compute BSS per grid point
            bss_spatial = 1 - (bs_spatial / bs_clim_spatial) 
            
            #check for invalid values in bss_spatial in europe_masked area
            bss_spatial_masked = bss_spatial.where(europe_mask_da, drop = True)
            n_invalid_bss = (~np.isfinite(bss_spatial_masked)).sum()
            print(f"  Number of invalid values in Brier Skill Score (Europe only): {n_invalid_bss}")

            # Store spatial data
            spatial_data.append({
                'week_lead': week_lead,
                'bs': bs_spatial,
                'bs_clim': bs_clim_spatial,
                'bss': bss_spatial,
                'clim_freq': clim_freq_spatial
            })

            # === AGGREGATED METRICS (for reliability diagrams and summary stats) ===
            # Use pre-computed flattened data (already filtered to Europe, already computed)
            obs_flat = obs_flat_all.sel(week_lead_time=week_lead)
            prob_flat = prob_flat_all.sel(week_lead_time=week_lead)
            
            # Remove any remaining NaN values (safety check for missing data)
            # Arrays are now 1D after selecting week_lead_time
            obs_flat_np = obs_flat.values.ravel()  # Ensure 1D
            prob_flat_np = prob_flat.values.ravel()  # Ensure 1D
            valid_mask = ~np.isnan(obs_flat_np) & ~np.isnan(prob_flat_np)
            obs_valid = obs_flat_np[valid_mask]
            prob_valid = prob_flat_np[valid_mask]      

            # Compute Brier Score (aggregated)
            # Note: obs_valid and prob_valid are numpy arrays, so we don't specify dim
            bs = float(brier_score(obs_valid, prob_valid).mean())
            
            # Compute climatological frequency per grid point (for Brier Skill Score)
            # Following supervisor's recommendation: compute BS_clim at each grid point, then average
            clim_freq_per_point = obs_lead.mean(dim='init_date')  # Climatology at each grid point
            bs_clim_per_point = clim_freq_per_point * (1 - clim_freq_per_point)  # BS_clim at each point
            bs_clim = float(bs_clim_per_point.where(europe_mask_da, drop=True).mean())  # Average over Europe
            bss = 1 - (bs / bs_clim) if bs_clim > 0 else np.nan
            
            # Also compute spatially-averaged climatological frequency for reference
            clim_freq = float(obs_valid.mean())  

            # Compute reliability diagram data
            # Bin probabilities into intervals
            prob_bins = np.linspace(0, 1, 11)  # 10 bins: [0-0.1], [0.1-0.2], ..., [0.9-1.0]
            bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2

            obs_freq_per_bin = []
            forecast_prob_per_bin = []
            counts_per_bin = []

            for i in range(len(prob_bins) - 1):
                bin_mask = (prob_valid >= prob_bins[i]) & (prob_valid < prob_bins[i+1])
                if i == len(prob_bins) - 2:  # Last bin includes upper bound
                    bin_mask = (prob_valid >= prob_bins[i]) & (prob_valid <= prob_bins[i+1])

                count = int(bin_mask.sum())

                if count > 0:
                    obs_freq = float(obs_valid[bin_mask].mean())
                    fcst_prob = float(prob_valid[bin_mask].mean())
                    obs_freq_per_bin.append(obs_freq)
                    forecast_prob_per_bin.append(fcst_prob)
                    counts_per_bin.append(count)
                else:
                    obs_freq_per_bin.append(np.nan)
                    forecast_prob_per_bin.append(np.nan)
                    counts_per_bin.append(0)

            # Store metrics (include obs and prob for economic value computation)
            metrics_lead.append({
                'week_lead': week_lead,
                'brier_score': bs,
                'brier_skill_score': bss,
                'climatology_freq': clim_freq,
                'n_samples': int(valid_mask.sum()),
                'obs_valid': obs_valid,  # Store for economic value
                'prob_valid': prob_valid,  # Store for economic value
                'reliability': {
                    'bin_centers': bin_centers,
                    'observed_freq': np.array(obs_freq_per_bin),
                    'forecast_prob': np.array(forecast_prob_per_bin),
                    'counts': np.array(counts_per_bin)
                }
            })

            print(f"    Brier Score: {bs:.4f}")
            print(f"    Brier Skill Score: {bss:.4f}")
            print(f"    Climatology frequency: {clim_freq:.3f}")
            print(f"    Sample size: {int(valid_mask.sum())}")

        metrics_by_severity[severity] = metrics_lead
        metrics_spatial[severity] = spatial_data

#Roc computation (NEW THRESHOLDS)
print("\n" + "="*80)
print("COMPUTING ROC CURVES WITH NEW THRESHOLDS")
print("="*80)
for hw, roc_by_severity in zip(hw_models_new, list_of_roc_by_severity_new):
    for severity in HW_THRESHOLDS.keys():
        print(f"\nComputing ROC for {severity} heat waves...")
        obs_europe = hw_obs_new[severity].where(europe_mask_da, drop=True)
        prob_europe = hw[severity].where(europe_mask_da, drop=True)

        roc_by_severity[severity] = roc(obs_europe, prob_europe, dim = ["values","init_date"], bin_edges='continuous', return_results = "all_as_metric_dim")
             
             
##########################################################################################################################################
#EXPORT
# Save spatial data
 
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)



model_identifiers = ['reference', 'weekly_finetuned', 'daily_finetuned']


# Export spatial metrics for NEW THRESHOLDS
print("\nExporting spatial metrics (NEW THRESHOLDS)...")

for metrics_spatial, model_id in zip(list_of_metrics_spatial_new, model_identifiers):
    spatial_datasets = []

    for severity in HW_THRESHOLDS.keys():
        severity_data = metrics_spatial[severity]

        # Stack lead times for this severity
        bs_list = [d['bs'] for d in severity_data]
        bs_clim_list = [d['bs_clim'] for d in severity_data]
        bss_list = [d['bss'] for d in severity_data]
        clim_freq_list = [d['clim_freq'] for d in severity_data]
        week_leads = [d['week_lead'] for d in severity_data]

        # Create dataset for this severity
        ds_severity = xr.Dataset({
            'brier_score': xr.concat(bs_list, dim='week_lead_time'),
            'brier_score_climatology': xr.concat(bs_clim_list, dim='week_lead_time'),
            'brier_skill_score': xr.concat(bss_list, dim='week_lead_time'),
            'climatology_frequency': xr.concat(clim_freq_list, dim='week_lead_time')
        })
        ds_severity = ds_severity.assign_coords(week_lead_time=week_leads)

        # Add latitude and longitude as coordinates for the 'values' dimension
        ds_severity = ds_severity.assign_coords({
            'latitude': ('values', lat),
            'longitude': ('values', lon)
        })

        spatial_datasets.append(ds_severity)

    # Combine across severities
    ds_spatial = xr.concat(spatial_datasets, dim='severity')
    ds_spatial = ds_spatial.assign_coords(severity=list(HW_THRESHOLDS.keys()))

    # Save with _new suffix
    filename = f"heat_wave_metrics_spatial_{model_id}_new.nc"
    ds_spatial.to_netcdf(os.path.join("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_metrics", filename))
    print(f"✓ Saved spatial metrics (new thresholds) for {model_id}: shape {dict(ds_spatial.dims)}")



# Export aggregated metrics for NEW THRESHOLDS
print("\nExporting aggregated metrics (NEW THRESHOLDS)...")

for metrics_by_severity, model_id in zip(list_of_metrics_by_severity_new, model_identifiers):
    print(f"  Exporting aggregated metrics for {model_id}...")
    aggregated_datasets = []

    for severity in HW_THRESHOLDS.keys():
        severity_data = metrics_by_severity[severity]

        week_leads = [d['week_lead'] for d in severity_data]

        ds_severity = xr.Dataset({
            'brier_score': (['week_lead_time'], [d['brier_score'] for d in severity_data]),
            'brier_skill_score': (['week_lead_time'], [d['brier_skill_score'] for d in severity_data]),
            'climatology_frequency': (['week_lead_time'], [d['climatology_freq'] for d in severity_data]),
            'n_samples': (['week_lead_time'], [d['n_samples'] for d in severity_data])
        })
        ds_severity = ds_severity.assign_coords(week_lead_time=week_leads)
        aggregated_datasets.append(ds_severity)

    # Combine across severities
    ds_aggregated = xr.concat(aggregated_datasets, dim='severity')
    ds_aggregated = ds_aggregated.assign_coords(severity=list(HW_THRESHOLDS.keys()))

    # Save with _new suffix
    filename = f"heat_wave_metrics_aggregated_{model_id}_new.nc"
    ds_aggregated.to_netcdf(os.path.join("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_metrics", filename))
    print(f"✓ Saved aggregated metrics (new thresholds) for {model_id}: shape {dict(ds_aggregated.dims)}")

# Save reliability diagram data separately (doesn't fit well in regular xarray structure)
import json

print("\nExporting reliability diagrams (OLD THRESHOLDS)...")

for metrics_by_severity, model_id in zip(list_of_metrics_by_severity, model_identifiers):
    reliability_data = {}
    for severity in HW_THRESHOLDS.keys():
        reliability_data[severity] = {}
        for lead_metrics in metrics_by_severity[severity]:
            week_lead = lead_metrics['week_lead']
            reliability_data[severity][f'week_{week_lead}'] = {
                'bin_centers': lead_metrics['reliability']['bin_centers'].tolist(),
                'observed_freq': lead_metrics['reliability']['observed_freq'].tolist(),
                'forecast_prob': lead_metrics['reliability']['forecast_prob'].tolist(),
                'counts': lead_metrics['reliability']['counts'].tolist()
            }

    filename = f"heat_wave_reliability_diagrams_{model_id}.json"
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        json.dump(reliability_data, f, indent=2)
    print(f"✓ Saved reliability diagram data for {model_id}")

# Export reliability diagrams for NEW THRESHOLDS
print("\nExporting reliability diagrams (NEW THRESHOLDS)...")

for metrics_by_severity, model_id in zip(list_of_metrics_by_severity_new, model_identifiers):
    reliability_data = {}
    for severity in HW_THRESHOLDS.keys():
        reliability_data[severity] = {}
        for lead_metrics in metrics_by_severity[severity]:
            week_lead = lead_metrics['week_lead']
            reliability_data[severity][f'week_{week_lead}'] = {
                'bin_centers': lead_metrics['reliability']['bin_centers'].tolist(),
                'observed_freq': lead_metrics['reliability']['observed_freq'].tolist(),
                'forecast_prob': lead_metrics['reliability']['forecast_prob'].tolist(),
                'counts': lead_metrics['reliability']['counts'].tolist()
            }

    filename = f"heat_wave_reliability_diagrams_{model_id}_new.json"
    with open(os.path.join("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_metrics", filename), 'w') as f:
        json.dump(reliability_data, f, indent=2)
    print(f"✓ Saved reliability diagram data (new thresholds) for {model_id}")


### export ROC data (OLD THRESHOLDS)
print("\nExporting ROC data (OLD THRESHOLDS)...")

for roc_by_severity, model_id in zip(list_of_roc_by_severity, model_identifiers):

    severity_order = list(HW_THRESHOLDS.keys())

    # Concatenate all severities into one DataArray
    roc_da = xr.concat(
        [roc_by_severity[severity] for severity in severity_order],
        dim="severity"
    )

    # Add severity coordinate
    roc_da = roc_da.assign_coords(
        severity=("severity", severity_order)
    )

    # Convert to Dataset
    ds_roc = roc_da.to_dataset(name="roc")

    filename = f"heat_wave_roc_{model_id}.nc"
    ds_roc.to_netcdf(os.path.join(OUTPUT_DIR, filename))

    print(f"✓ Saved ROC data for {model_id}: shape {dict(ds_roc.dims)}")

### export ROC data (NEW THRESHOLDS)
print("\nExporting ROC data (NEW THRESHOLDS)...")

for roc_by_severity, model_id in zip(list_of_roc_by_severity_new, model_identifiers):

    severity_order = list(HW_THRESHOLDS.keys())

    # Concatenate all severities into one DataArray
    roc_da = xr.concat(
        [roc_by_severity[severity] for severity in severity_order],
        dim="severity"
    )

    # Add severity coordinate
    roc_da = roc_da.assign_coords(
        severity=("severity", severity_order)
    )

    # Convert to Dataset
    ds_roc = roc_da.to_dataset(name="roc")

    filename = f"heat_wave_roc_{model_id}_new.nc"
    ds_roc.to_netcdf(os.path.join("/ec/res4/scratch/nld4584/Anemoi_S2S_experiment/output_metrics", filename))

    print(f"✓ Saved ROC data (new thresholds) for {model_id}: shape {dict(ds_roc.dims)}")

print("\n" + "="*80)
print("ALL EXPORTS COMPLETE")
print("Files saved to:", OUTPUT_DIR)
print("="*80)