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
WEEKLY_MEANS_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/weekly_means_physics"
OBS_WEEKLY_FILE = os.path.join(WEEKLY_MEANS_DIR, "Observations_weekly_AIFS_physics.nc")
FCST_WEEKLY_FILE = os.path.join(WEEKLY_MEANS_DIR, "Forecasts_weekly_AIFS_physics.nc")
OUTPUT_DIR = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/heat_wave_analysis_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variables for comprehensive physics analysis
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

print("="*80)
print("ENHANCED HEAT WAVE ANALYSIS WITH COMPREHENSIVE PHYSICS")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nVariables for physics analysis:")
print(f"  Surface: {SURFACE_VARS}")
print(f"  Upper levels: {UPPER_VARS}")
print()

# ============================================================================
# DATA LOADING (from pre-computed weekly means)
# ============================================================================

def load_weekly_data():
    """
    Load pre-computed weekly mean data for observations and forecasts
    
    These files should be created by running preprocess_heatwave_physics_data.py first
    """
    print("Loading pre-computed weekly mean data...")
    
    # Check files exist
    if not os.path.exists(OBS_WEEKLY_FILE):
        raise FileNotFoundError(
            f"Observation weekly file not found: {OBS_WEEKLY_FILE}\n"
            f"Please run preprocess_heatwave_physics_data.py first!"
        )
    if not os.path.exists(FCST_WEEKLY_FILE):
        raise FileNotFoundError(
            f"Forecast weekly file not found: {FCST_WEEKLY_FILE}\n"
            f"Please run preprocess_heatwave_physics_data.py first!"
        )
    
    # Load observations
    print(f"  Loading observations from: {os.path.basename(OBS_WEEKLY_FILE)}")
    obs_weekly = xr.open_dataset(OBS_WEEKLY_FILE)
    print(f"    Shape: {dict(obs_weekly.dims)}")
    print(f"    Variables: {list(obs_weekly.data_vars)}")
    
    # Load forecasts
    print(f"  Loading forecasts from: {os.path.basename(FCST_WEEKLY_FILE)}")
    fcst_weekly = xr.open_dataset(FCST_WEEKLY_FILE)
    print(f"    Shape: {dict(fcst_weekly.dims)}")
    print(f"    Variables: {list(fcst_weekly.data_vars)}")
    
    # Get lat/lon from original 4-variable weekly means file (has coordinates)
    print(f"  Loading lat/lon coordinates from original weekly means file...")
    obs_orig = xr.open_dataset("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Observations_weekly_AIFS.nc")
    lat = obs_orig.latitude.values
    lon = obs_orig.longitude.values
    lon_adjusted = np.where(lon <= 180, lon, lon - 360)
    region_lat_mask = (lat >= 20) & (lat <= 80)
    region_lon_mask = (lon_adjusted >= -60) & (lon_adjusted <= 90)
    region_mask = region_lat_mask & region_lon_mask
    print(f"    Grid points: {len(lat)}")
    
    # Create latitude weights for area-weighted averaging
    lat_weights = np.cos(np.radians(lat))
    lat_weights = xr.DataArray(lat_weights, dims=["values"])
    
    print(f"  ✓ Data loaded successfully")
    print(f"  Initialization dates: {len(obs_weekly.init_date)}")
    print(f"  Week lead times: {len(obs_weekly.week_lead_time)}")
    
    return obs_weekly, fcst_weekly, lat, lon, lat_weights, region_mask


def compute_climatology_and_anomalies(obs_weekly, fcst_weekly):
    """
    Compute climatology and anomalies for all variables
    """
    print("\nComputing climatology and anomalies...")
    
    # Convert init_date to datetime
    obs_weekly = obs_weekly.assign_coords(init_date=pd.to_datetime(obs_weekly.init_date.values))
    fcst_weekly = fcst_weekly.assign_coords(init_date=pd.to_datetime(fcst_weekly.init_date.values))
    
    # Compute leave-one-year-out climatology
    obs_clim_list = []
    fcst_clim_list = []
    
    for init_date in obs_weekly.init_date.values:
        year = pd.Timestamp(init_date).year
        month = pd.Timestamp(init_date).month
        day = pd.Timestamp(init_date).day
        
        # Mask for same calendar date, different years
        mask = ((obs_weekly.init_date.dt.month == month) &
                (obs_weekly.init_date.dt.day == day) &
                (obs_weekly.init_date.dt.year != year))
        
        # Climatology
        # Observations: average over init_date only
        obs_clim = obs_weekly.sel(init_date=mask).mean(dim='init_date')
        # Forecasts: average over init_date only, keeping member dimension
        # This gives each member its own climatology (removes member-specific biases)
        fcst_clim = fcst_weekly.sel(init_date=mask).mean(dim='init_date')
        
        obs_clim_list.append(obs_clim)
        fcst_clim_list.append(fcst_clim)
    
    # Concatenate
    obs_climatology = xr.concat(obs_clim_list, dim=obs_weekly.init_date)
    fcst_climatology = xr.concat(fcst_clim_list, dim=obs_weekly.init_date)
    
    # Compute anomalies
    obs_anomalies = obs_weekly - obs_climatology
    fcst_anomalies = fcst_weekly - fcst_climatology
    
    print(f"  Temperature anomaly range (obs): {float(obs_anomalies['2t'].min()):.2f} to {float(obs_anomalies['2t'].max()):.2f} K")
    
    return obs_climatology, fcst_climatology, obs_anomalies, fcst_anomalies


# ============================================================================
# HEAT WAVE DETECTION
# ============================================================================

def detect_heat_waves(obs_anomalies, fcst_anomalies):
    """
    Detect heat waves based on 2m temperature anomaly thresholds
    Uses standard deviation thresholds: anomaly > N * std_dev
    
    Uses leave-one-year-out (LOYO) approach for computing std dev thresholds
    to avoid circular issue where extreme years inflate their own thresholds.
    
    For forecasts: uses ensemble-mean anomalies and std dev
    (we want to know if THE MODEL predicts heat waves, not individual members)
    """
    print("\n" + "="*80)
    print("HEAT WAVE DETECTION")
    print("="*80)
    
    # Compute standard deviation using leave-one-year-out (LOYO) approach
    # This prevents extreme years from inflating their own thresholds
    print("\nComputing standard deviation thresholds (leave-one-year-out)...")
    
    # For observations: first get unique valid dates to avoid counting duplicates
    # (multiple init_date/week_lead_time pairs can point to same valid date)
    print("  Reshaping observations to unique valid dates...")
    obs_t2 = obs_anomalies['2t']
    
    # Stack dimensions to create single time dimension
    obs_stacked = obs_t2.stack(time_point=['init_date', 'week_lead_time'])
    
    # Compute valid date for each (init_date, week_lead_time) pair
    valid_dates = []
    for init_date_val, week_lead_val in obs_stacked.time_point.values:
        valid_date = pd.Timestamp(init_date_val) + pd.Timedelta(days=int(week_lead_val) * 7)
        valid_dates.append(valid_date)
    
    # Assign valid_date as coordinate
    obs_stacked = obs_stacked.assign_coords(valid_date=('time_point', valid_dates))
    
    # DIAGNOSTIC: Check if "duplicate" valid dates actually have different values
    print(f"\n  Checking for actual duplicates vs. overlapping weekly windows...")
    valid_dates_array = np.array(valid_dates)
    unique_dates, inverse_indices, counts = np.unique(valid_dates_array, return_inverse=True, return_counts=True)
    duplicated_dates = unique_dates[counts > 1]
    
    if len(duplicated_dates) > 0:
        print(f"  Found {len(duplicated_dates)} valid dates appearing multiple times")
        # Check first few duplicates to see if values are actually the same
        n_check = min(3, len(duplicated_dates))
        for i in range(n_check):
            dup_date = duplicated_dates[i]
            # Get all time_points with this valid date
            dup_indices = np.where(valid_dates_array == dup_date)[0]
            # Get corresponding (init_date, week_lead) pairs
            init_lead_pairs = [obs_stacked.time_point.values[idx] for idx in dup_indices]
            # Get values for first grid point
            values_at_first_gridpoint = [float(obs_stacked.isel(values=0)[idx]) for idx in dup_indices]
            
            print(f"\n    Example {i+1}: Valid date {pd.Timestamp(dup_date).date()}")
            for j, (init, lead) in enumerate(init_lead_pairs):
                init_date_str = pd.Timestamp(init).date()
                print(f"      Init {init_date_str} + {lead} weeks → value at grid[0]: {values_at_first_gridpoint[j]:.4f} K")
            
            # Check if all values are identical
            if np.allclose(values_at_first_gridpoint, values_at_first_gridpoint[0]):
                print(f"      → Values are IDENTICAL (true duplicate)")
            else:
                print(f"      → Values are DIFFERENT (overlapping weekly windows)")
    else:
        print(f"  No duplicate valid dates found - all {len(unique_dates)} dates are unique!")
    
    # Group by valid_date and take first occurrence (removes duplicates)
    obs_unique = obs_stacked.groupby('valid_date').first()
    
    print(f"  Summary:")
    print(f"  Original obs shape: {len(obs_t2.init_date)} init_dates × {len(obs_t2.week_lead_time)} lead_times = {len(obs_t2.init_date) * len(obs_t2.week_lead_time)} points")
    print(f"  Unique valid dates: {len(obs_unique.valid_date)}")
    print(f"  Reduction: {len(obs_t2.init_date) * len(obs_t2.week_lead_time) - len(obs_unique.valid_date)} points removed")
    
    # Show example of overlapping weekly windows with different values
    print(f"\n  Example of overlapping weekly windows creating independent samples:")
    # Take first two init dates in May
    may_inits = [d for d in obs_anomalies.init_date.values if pd.Timestamp(d).month == 5][:2]
    if len(may_inits) >= 2:
        init1 = pd.Timestamp(may_inits[0])
        init2 = pd.Timestamp(may_inits[1])
        
        # Week 2 from first init
        valid1 = init1 + pd.Timedelta(days=14)
        week_start1 = init1 + pd.Timedelta(days=7)
        week_end1 = valid1
        
        # Week 1 from second init  
        valid2 = init2 + pd.Timedelta(days=7)
        week_start2 = init2
        week_end2 = valid2
        
        print(f"    Case 1: Init {init1.date()} + 2 weeks → Valid date {valid1.date()}")
        print(f"            Weekly mean period: {week_start1.date()} to {week_end1.date()}")
        print(f"    Case 2: Init {init2.date()} + 1 week  → Valid date {valid2.date()}")
        print(f"            Weekly mean period: {week_start2.date()} to {week_end2.date()}")
        
        # Check if these overlap
        overlap_days = max(0, (min(week_end1, week_end2) - max(week_start1, week_start2)).days)
        print(f"            → Calendar overlap: {overlap_days} days")
        print(f"            → But these are DIFFERENT 7-day averages (independent samples)")
    
    # Now compute LOYO std for each init_date
    obs_std_list = []
    
    # Diagnostic: Show sample sizes for first init_date (e.g., 2010)
    first_year = pd.Timestamp(obs_anomalies.init_date.values[0]).year
    mask_first = obs_unique.valid_date.dt.year != first_year
    n_samples_loyo = int(mask_first.sum())
    n_excluded = len(obs_unique.valid_date) - n_samples_loyo
    
    print(f"\n  LOYO sample sizes (example for year {first_year}):")
    print(f"    Total unique valid dates: {len(obs_unique.valid_date)}")
    print(f"    Excluded (year {first_year}): {n_excluded}")
    print(f"    Used for std computation: {n_samples_loyo} independent weekly observations per grid point")
    
    for init_date in obs_anomalies.init_date.values:
        year = pd.Timestamp(init_date).year
        
        # Mask for valid dates not in target year
        mask = obs_unique.valid_date.dt.year != year
        
        # Observations: std dev per grid point over unique valid dates, excluding this year
        obs_std_loyo = obs_unique.sel(valid_date=mask).std(dim='valid_date')
        
        obs_std_list.append(obs_std_loyo)
    
    # Concatenate to create std dev for each init_date
    obs_std = xr.concat(obs_std_list, dim=obs_anomalies.init_date)
    
    print(f"  Observed temperature anomaly std dev (global mean): {float(obs_std.mean()):.2f} K")
    print(f"  Observed temperature anomaly std dev (range): {float(obs_std.min()):.2f} to {float(obs_std.max()):.2f} K")
    print(f"  Note: Each init date uses leave-one-year-out std dev to avoid circular threshold inflation")
    print(f"  Note: Computed from unique valid dates (avoids counting same observation multiple times)")
    print(f"  Note: Using observational thresholds for both obs and fcst (to detect REAL heat waves)")
    
    thresholds = {}
    
    for severity, std_threshold in HW_THRESHOLDS.items():
        print(f"  {severity.capitalize()}: > {std_threshold} std dev...")
        thresholds[severity] = obs_std * std_threshold
    
    # Detect heat waves
    print("\nDetecting heat waves...")
    
    # Compute forecast ensemble-mean anomalies (needed for detection)
    fcst_anom_ens_mean = fcst_anomalies['2t'].mean(dim='member')
    
    hw_obs = {}
    hw_fcst = {}
    hw_fcst_prob = {}
    hw_stats_by_leadtime = {}  # Store statistics per lead time
    
    
    for severity in HW_THRESHOLDS.keys():
        print(f"  {severity.capitalize()} heat waves...")
        
        # Binary detection using OBSERVATIONAL threshold for both obs and fcst
        # This ensures we're detecting the same "real-world" heat waves in both
        hw_obs[severity] = (obs_anomalies['2t'] > thresholds[severity]).astype(int)
        # For forecasts: detect based on ensemble-mean anomaly using obs threshold
        hw_fcst_ens = (fcst_anom_ens_mean > thresholds[severity]).astype(int)
        
        # Also keep per-member detection for probability calculation
        hw_fcst_members = (fcst_anomalies['2t'] > thresholds[severity]).astype(int)
        hw_fcst_prob[severity] = hw_fcst_members.mean(dim='member')
        
        # Use ensemble-mean detection as primary forecast
        hw_fcst[severity] = hw_fcst_ens
        
        # Global statistics
        obs_freq = float(hw_obs[severity].mean() * 100)
        fcst_freq = float(hw_fcst[severity].mean() * 100)
        print(f"    Global - Observed frequency: {obs_freq:.2f}%")
        print(f"    Global - Forecast frequency: {fcst_freq:.2f}%")
        
        # Statistics by lead time
        print(f"\n    Statistics by lead time:")
        print(f"    {'Week':<6} {'Obs %':<10} {'Fcst %':<10} {'Bias %':<10} {'Hit Rate':<10}")
        print(f"    {'-'*50}")
        
        hw_stats_by_leadtime[severity] = []
        
        for week_lead in range(1, 9):  # Lead times 1-8
            obs_lead = hw_obs[severity].sel(week_lead_time=week_lead)
            fcst_lead = hw_fcst[severity].sel(week_lead_time=week_lead)
            
            obs_freq_lead = float(obs_lead.mean() * 100)
            fcst_freq_lead = float(fcst_lead.mean() * 100)
            bias = fcst_freq_lead - obs_freq_lead
            
            # Hit rate: fraction of observed heat waves that were forecast
            hits = float((obs_lead & fcst_lead).sum())
            total_obs = float(obs_lead.sum())
            hit_rate = (hits / total_obs * 100) if total_obs > 0 else 0
            
            print(f"    {week_lead:<6} {obs_freq_lead:>6.2f}%   {fcst_freq_lead:>6.2f}%   "
                  f"{bias:>+6.2f}%   {hit_rate:>6.1f}%")
            
            hw_stats_by_leadtime[severity].append({
                'week_lead': week_lead,
                'obs_freq': obs_freq_lead,
                'fcst_freq': fcst_freq_lead,
                'bias': bias,
                'hit_rate': hit_rate
            })
        
        print()  # Extra newline between severities
    
    return hw_obs, hw_fcst, hw_fcst_prob, thresholds, thresholds, hw_stats_by_leadtime


# ============================================================================
# 2010 RUSSIAN HEAT WAVE IDENTIFICATION
# ============================================================================

def identify_2010_russian_heatwave_region(obs_anomalies, hw_obs, lat, lon):
    """
    Identify the spatial region affected by the 2010 Russian/Eastern European heat wave
    
    Returns:
        spatial_mask: Boolean array indicating affected grid points
        year_2010_mask: Boolean array for 2010 summer init dates
    """
    print("\n" + "="*80)
    print("IDENTIFYING 2010 RUSSIAN/EASTERN EUROPEAN HEAT WAVE REGION")
    print("="*80)
    
    # Define approximate region for Eastern Europe and Western Russia
    # Based on documented extent of 2010 heat wave
    # Latitude: 45°N to 65°N, Longitude: 30°E to 70°E
    region_lat_mask = (lat >= 45) & (lat <= 65)
    region_lon_mask = (lon >= 30) & (lon <= 70)
    region_mask = region_lat_mask & region_lon_mask
    
    print(f"  Geographic region: 45-65°N, 30-70°E")
    print(f"  Grid points in region: {region_mask.sum()}")
    
    # Define heat wave VALID date period (when heat wave actually occurred)
    # The 2010 Russian heat wave occurred during June-August 2010
    heat_wave_start = pd.Timestamp('2010-06-01')
    heat_wave_end = pd.Timestamp('2010-08-31')
    
    print(f"  Heat wave valid date period: {heat_wave_start.date()} to {heat_wave_end.date()}")
    
    # Create mask for all (init_date, week_lead_time) pairs where valid date falls in heat wave period
    # Valid date = init_date + week_lead_time * 7 days
    year_2010_mask = xr.DataArray(
        np.zeros((len(obs_anomalies.init_date), len(obs_anomalies.week_lead_time)), dtype=bool),
        dims=['init_date', 'week_lead_time'],
        coords={'init_date': obs_anomalies.init_date, 'week_lead_time': obs_anomalies.week_lead_time}
    )
    
    for i, init_date in enumerate(obs_anomalies.init_date.values):
        for j, week_lead in enumerate(obs_anomalies.week_lead_time.values):
            valid_date = pd.Timestamp(init_date) + pd.Timedelta(days=int(week_lead) * 7)
            if heat_wave_start <= valid_date <= heat_wave_end:
                year_2010_mask[i, j] = True
    
    n_valid_pairs = int(year_2010_mask.sum().values)
    print(f"  (init_date, week_lead_time) pairs with valid dates in heat wave period: {n_valid_pairs}")
    
    # Identify grid points that experienced heat waves during 2010
    # Use moderate heat waves (with local thresholds, this captures significant anomalies)
    hw_2010_moderate = hw_obs['moderate'].where(year_2010_mask, 0)
    
    # Count how many times each grid point experienced heat waves
    # This identifies the core affected region
    hw_frequency = hw_2010_moderate.sum(dim=['init_date', 'week_lead_time'])
    
    # Define affected region: grid points with at least 2 heat wave occurrences
    # This ensures we capture the persistent heat wave, not isolated events
    hw_threshold = 2
    hw_affected_mask = hw_frequency >= hw_threshold
    
    # Combine geographic region with heat wave occurrence
    spatial_mask = region_mask & hw_affected_mask.values
    
    print(f"  Grid points with ≥{hw_threshold} HW occurrences in 2010: {hw_affected_mask.sum().values}")
    print(f"  Final spatial mask (region ∩ HW): {spatial_mask.sum()} grid points")
    print(f"  This defines the 2010 Russian heat wave affected region")
    print(f"\nNote: year_2010_mask is 2D (init_date, week_lead_time) based on valid dates")
    print(f"      Each lead time may have different initialization dates")
    
    # For analysis, use the broader region_mask (not just grid points with many HWs)
    # This avoids the issue where specific lead times might have no HWs at the exact core points
    return region_mask, spatial_mask, year_2010_mask


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main analysis pipeline: 2010 Russian Heat Wave Case Study
    
    This version focuses on the 2010 Russian/Eastern European heat wave,
    analyzing forecast skill degradation with lead time.
    
    NOTE: Before running this script, you must first run:
          preprocess_heatwave_physics_data.py
    to create the weekly mean files with all physics variables.
    """
    print("\n" + "="*80)
    print("2010 RUSSIAN HEAT WAVE CASE STUDY")
    print("="*80)
    print("\nThis analysis focuses on a single extreme event:")
    print("  - Event: 2010 Russian/Eastern European heat wave")
    print("  - Region: 45-65°N, 30-70°E")
    print("  - Period: June-August 2010")
    print("  - Analysis: Forecast skill by lead time (weeks 1-8)")
    print()
    
    # Load pre-computed weekly mean data
    obs_weekly, fcst_weekly, lat, lon, lat_weights = load_weekly_data()
    
    # Compute climatology and anomalies
    obs_clim, fcst_clim, obs_anom, fcst_anom = compute_climatology_and_anomalies(
        obs_weekly, fcst_weekly)
    
    # Detect heat waves (global detection for all severities)
    hw_obs, hw_fcst, hw_fcst_prob, thresh_obs, thresh_fcst, hw_stats_by_leadtime = detect_heat_waves(
        obs_anom, fcst_anom)
    
    # Identify the 2010 Russian heat wave region
    region_mask, spatial_mask, year_2010_mask = identify_2010_russian_heatwave_region(
        obs_anom, hw_obs, lat, lon)
    
    # Analyze physics patterns by lead time
    results_by_leadtime = analyze_2010_heatwave_by_leadtime(
        obs_weekly, fcst_weekly, obs_anom, fcst_anom, hw_obs, hw_fcst,
        region_mask, spatial_mask, year_2010_mask, lat, lon, lat_weights)
    
    # Visualization
    plot_2010_leadtime_skill_evolution(results_by_leadtime)
    plot_2010_spatial_patterns_by_leadtime(obs_anom, fcst_anom, hw_obs, 
                                            region_mask, year_2010_mask, lat, lon)
    plot_2010_physics_patterns_by_leadtime(obs_weekly, fcst_weekly, obs_anom, fcst_anom,
                                            hw_obs, hw_fcst, region_mask, year_2010_mask, lat, lon)
    create_2010_summary_report(results_by_leadtime, spatial_mask, hw_stats_by_leadtime)
    
    print("\n" + "="*80)
    print("2010 RUSSIAN HEAT WAVE ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*80)
    print("\nOutput files:")
    print("  - 2010_russian_heatwave_leadtime_skill.png (skill evolution)")
    print("  - 2010_russian_heatwave_spatial_by_leadtime.png (temperature spatial patterns)")
    print("  - 2010_russian_heatwave_msl_spatial.png (blocking pattern, all lead times)")
    print("  - 2010_russian_heatwave_2d_spatial.png (moisture deficit, all lead times)")
    print("  - 2010_russian_heatwave_tp_spatial.png (precipitation, all lead times)")
    print("  - 2010_russian_heatwave_analysis.txt (summary report)")
    print()


if __name__ == "__main__":
    main()
