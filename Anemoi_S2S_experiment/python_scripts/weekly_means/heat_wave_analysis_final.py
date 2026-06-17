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

# Import rpy2 for R interface (using modern API)
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# For numpy-R conversion (using context manager to avoid deprecation warning)
try:
    # Try modern API first (rpy2 >= 3.5)
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import numpy2ri
    USE_CONTEXT_CONVERTER = True
except ImportError:
    # Fallback for older rpy2
    from rpy2.robjects import numpy2ri
    USE_CONTEXT_CONVERTER = False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        numpy2ri.activate()

# Add paths
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')

# Setup R library path and load verification package
r_home = os.path.expanduser("~/R/library")
ro.r(f'.libPaths(c("{r_home}", .libPaths()))')
try:
    verification = importr('verification')
    print(f"✓ R verification package loaded from {r_home}")
except Exception as e:
    print(f"Warning: Could not load R verification package: {e}")
    verification = None
  
################################################################################################################################  
#Load datasets
# Load preprocessed data
preprocessed_dir = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed_75_init"

# Load latitude weights
print("Loading latitude weights...")
lat_weights_ds = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/preprocessed/lat_weights.nc")
lat_weights = lat_weights_ds['lat_weights']

# Load observations
print("Loading observations...")
ds_obs = xr.open_dataset(os.path.join(preprocessed_dir, "observations_weekly_27_0.nc"))
ds_obs_anomalies = xr.open_zarr(os.path.join(preprocessed_dir, "observations_anomalies_weekly_27_0.zarr"))
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
ds_ref_anomalies = xr.open_zarr(os.path.join(preprocessed_dir, "reference_model_anomalies_weekly_27_0.zarr"))
ds_ref_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "reference_model_climatology_weekly_27_0.nc"))

#Load weekly finetuned model
print("Loading weekly finetuned model...")
ds_weekly_finetuned = xr.open_dataset(os.path.join(preprocessed_dir, "weekly_finetuned_model_27_0.nc"))
ds_weekly_finetuned_anomalies = xr.open_zarr(os.path.join(preprocessed_dir, "weekly_finetuned_model_anomalies_27_0.zarr"))
ds_weekly_finetuned_climatology = xr.open_dataset(os.path.join(preprocessed_dir, "weekly_finetuned_model_climatology_27_0.nc"))

#Load daily finetuned model
print("Loading daily finetuned model...")
ds_daily_finetuned = xr.open_dataset(os.path.join(preprocessed_dir, "daily_finetuned_model_weekly_3_0.nc"))
ds_daily_finetuned_anomalies = xr.open_zarr(os.path.join(preprocessed_dir, "daily_finetuned_model_weekly_anomalies_3_0.zarr"))
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

print(f"  Observed temperature anomaly std dev (global mean): {float(obs_std.mean()):.2f} K")
print(f"  Observed temperature anomaly std dev (range): {float(obs_std.min()):.2f} to {float(obs_std.max()):.2f} K")
print(f"  Note: Each init date uses leave-one-year-out std dev to avoid circular threshold inflation")
print(f"  Note: Computed from unique valid dates (avoids counting same observation multiple times)")

thresholds = {}

for severity, std_threshold in HW_THRESHOLDS.items():
    print(f"  {severity.capitalize()}: > {std_threshold} std dev...")
    thresholds[severity] = obs_std * std_threshold
    
###################################################################################################################################
#DETECT HEAT WAVES

hw_obs = {}
hw_ref = {}
hw_weekly_finetuned = {}
hw_daily_finetuned = {}

hw_models = [hw_ref, hw_weekly_finetuned, hw_daily_finetuned]
anomalies_list = [ds_ref_anomalies, ds_weekly_finetuned_anomalies, ds_daily_finetuned_anomalies]
model_names = ['Reference', 'Weekly Finetuned', 'Daily Finetuned']

for model, anomalies, name in zip(hw_models, anomalies_list, model_names):
    for severity in HW_THRESHOLDS.keys():
        print(f"  {severity.capitalize()} heat waves...")

        # Binary detection using OBSERVATIONAL threshold for both obs and fcst
        # This ensures we're detecting the same "real-world" heat waves in both
        hw_obs[severity] = (ds_obs_anomalies['2t'] > thresholds[severity]).astype(int)
        hw_members = (anomalies['2t'] > thresholds[severity]).astype(int)
        model[severity] = hw_members.mean(dim='member')

# ============================================================================
# PROBABILISTIC METRICS FOR HEAT WAVE PREDICTION
# ============================================================================

europe_mask_da = xr.DataArray(europe_mask, dims=["values"])

ref_metrics_by_severity = {}
ref_metrics_spatial = {}
ref_roc_by_severity = {}

weekly_finetuned_metrics_by_severity = {}
weekly_finetuned_metrics_spatial = {}
weekly_finetuned_roc_by_severity = {}

daily_finetuned_metrics_by_severity = {} 
daily_finetuned_metrics_spatial = {}
daily_finetuned_roc_by_severity = {}

list_of_metrics_by_severity = [ref_metrics_by_severity, weekly_finetuned_metrics_by_severity, daily_finetuned_metrics_by_severity]
list_of_metrics_spatial = [ref_metrics_spatial, weekly_finetuned_metrics_spatial, daily_finetuned_metrics_spatial]
list_of_roc_by_severity = [ref_roc_by_severity, weekly_finetuned_roc_by_severity, daily_finetuned_roc_by_severity]

for hw, metrics_by_severity, metrics_spatial in zip(hw_models, list_of_metrics_by_severity, list_of_metrics_spatial):
    for severity in HW_THRESHOLDS.keys():
        print(f"\n{severity.upper()} HEAT WAVES:")
        print("-" * 80)

        # Extract data for Europe only
        obs_europe = hw_obs[severity].where(europe_mask_da, drop=False)
        prob_europe = hw[severity].where(europe_mask_da, drop=False)

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

            # Compute climatological frequency per grid point
            clim_freq_spatial = obs_europe.mean(dim=['init_date', 'week_lead_time'])  # Climatological frequency of heat wave occurrence
            #check for invalid values in clim_freq_spatial in europe_masked area
            clim_freq_spatial_masked = clim_freq_spatial.where(europe_mask_da, drop = True)
            n_invalid = (~np.isfinite(clim_freq_spatial_masked)).sum()
            print(f"  Number of invalid values in climatological frequency (Europe only): {n_invalid}")
            # Compute BS_clim per grid point: p(1-p)
            bs_clim_spatial = clim_freq_spatial * (1 - clim_freq_spatial)

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
            # Flatten to 1D for aggregated verification (stack init_date and values)
            #obs_lead_europe = obs_lead.where(europe_mask_da, drop=True)
            #prob_lead_europe = prob_lead.where(europe_mask_da, drop=True)
            obs_flat = obs_lead.stack(sample=['init_date', 'values'])
            prob_flat = prob_lead.stack(sample=['init_date', 'values'])

            # Remove NaN values (masked areas)
            # Compute arrays first to avoid dask boolean indexing issue
            obs_flat_computed = obs_flat.compute()
            prob_flat_computed = prob_flat.compute()
            valid_mask = ~np.isnan(obs_flat_computed) & ~np.isnan(prob_flat_computed)
            obs_valid = obs_flat_computed[valid_mask]
            prob_valid = prob_flat_computed[valid_mask]        

            # Compute Brier Score (aggregated)
            bs = float(brier_score(obs_valid, prob_valid, dim='sample'))

            # Compute climatological frequency (aggregated, for Brier Skill Score)
            clim_freq = float(obs_valid.mean())
            bs_clim = clim_freq * (1 - clim_freq)  # Brier score of climatology
            bss = 1 - (bs / bs_clim) if bs_clim > 0 else np.nan  

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
    

#Roc computation
for hw, roc_by_severity in zip(hw_models, list_of_roc_by_severity):
    for severity in HW_THRESHOLDS.keys():
        print(f"\nComputing ROC for {severity} heat waves...")
        obs_europe = hw_obs[severity].where(europe_mask_da, drop=True)
        prob_europe = hw[severity].where(europe_mask_da, drop=True)

        roc_by_severity[severity] = roc(obs_europe, prob_europe, dim = ["values","init_date"], bin_edges='continuous', return_results = "all_as_metric_dim")
             
             
##########################################################################################################################################
#EXPORT
# Save spatial data
 
# Convert metrics_spatial to xarray Dataset with proper lat/lon dimensions
print("\nExporting spatial metrics...")

model_identifiers = ['reference', 'weekly_finetuned', 'daily_finetuned']

for metrics_spatial, model_id in zip(list_of_metrics_spatial, model_identifiers):
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

    # Save
    filename = f"heat_wave_metrics_spatial_{model_id}.nc"
    ds_spatial.to_netcdf(os.path.join(OUTPUT_DIR, filename))
    print(f"✓ Saved spatial metrics for {model_id}: shape {dict(ds_spatial.dims)}")


# Convert metrics_by_severity to xarray Dataset (scalar metrics only)

for metrics_by_severity, model_id in zip(list_of_metrics_by_severity, model_identifiers):
    print(f"\nExporting aggregated metrics for {model_id}...")
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

    # Save
    filename = f"heat_wave_metrics_aggregated_{model_id}.nc"
    ds_aggregated.to_netcdf(os.path.join(OUTPUT_DIR, filename))
    print(f"✓ Saved aggregated metrics for {model_id}: shape {dict(ds_aggregated.dims)}")

# Save reliability diagram data separately (doesn't fit well in regular xarray structure)
import json

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


### export ROC data

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