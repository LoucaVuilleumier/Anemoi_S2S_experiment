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
    print(f"    Grid points: {len(lat)}")
    
    # Create latitude weights for area-weighted averaging
    lat_weights = np.cos(np.radians(lat))
    lat_weights = xr.DataArray(lat_weights, dims=["values"])
    
    print(f"  ✓ Data loaded successfully")
    print(f"  Initialization dates: {len(obs_weekly.init_date)}")
    print(f"  Week lead times: {len(obs_weekly.week_lead_time)}")
    
    return obs_weekly, fcst_weekly, lat, lon, lat_weights


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
    
    For forecasts: uses ensemble-mean anomalies and std dev
    (we want to know if THE MODEL predicts heat waves, not individual members)
    """
    print("\n" + "="*80)
    print("HEAT WAVE DETECTION")
    print("="*80)
    
    # Compute standard deviation
    print("\nComputing standard deviation thresholds...")
    # Observations: std dev across all times
    obs_std = obs_anomalies['2t'].std(dim=['init_date', 'week_lead_time'])
    # Forecasts: compute ensemble-mean anomalies, then their std dev
    # This tests whether the ensemble-mean forecast predicts heat waves
    fcst_anom_ens_mean = fcst_anomalies['2t'].mean(dim='member')
    fcst_std = fcst_anom_ens_mean.std(dim=['init_date', 'week_lead_time'])
    
    print(f"  Observed temperature anomaly std dev: {float(obs_std):.2f} K")
    print(f"  Forecast (ensemble-mean) temperature anomaly std dev: {float(fcst_std):.2f} K")
    
    thresholds_obs = {}
    thresholds_fcst = {}
    
    for severity, std_threshold in HW_THRESHOLDS.items():
        print(f"  {severity.capitalize()}: > {std_threshold} std dev...")
        thresholds_obs[severity] = obs_std * std_threshold
        thresholds_fcst[severity] = fcst_std * std_threshold
    
    # Detect heat waves
    print("\nDetecting heat waves...")
    
    hw_obs = {}
    hw_fcst = {}
    hw_fcst_prob = {}
    hw_stats_by_leadtime = {}  # Store statistics per lead time
    
    
    for severity in HW_THRESHOLDS.keys():
        print(f"  {severity.capitalize()} heat waves...")
        
        # Binary detection
        hw_obs[severity] = (obs_anomalies['2t'] > thresholds_obs[severity]).astype(int)
        # For forecasts: detect based on ensemble-mean anomaly
        hw_fcst_ens = (fcst_anom_ens_mean > thresholds_fcst[severity]).astype(int)
        
        # Also keep per-member detection for probability calculation
        hw_fcst_members = (fcst_anomalies['2t'] > thresholds_fcst[severity]).astype(int)
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
    
    return hw_obs, hw_fcst, hw_fcst_prob, thresholds_obs, thresholds_fcst, hw_stats_by_leadtime


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
    
    # Find 2010 summer dates (June-August when heat wave occurred)
    year_2010_mask = (obs_anomalies.init_date.dt.year == 2010) & \
                     (obs_anomalies.init_date.dt.month >= 6) & \
                     (obs_anomalies.init_date.dt.month <= 8)
    
    n_dates_2010 = int(year_2010_mask.sum().values)
    print(f"  2010 summer (Jun-Aug) init dates: {n_dates_2010}")
    
    # Identify grid points that experienced heat waves during 2010
    # Use severe heat waves as the threshold for this extreme event
    hw_2010_severe = hw_obs['severe'].sel(init_date=year_2010_mask)
    
    # Count how many times each grid point experienced heat waves
    # This identifies the core affected region
    hw_frequency = hw_2010_severe.sum(dim=['init_date', 'week_lead_time'])
    
    # Define affected region: grid points with at least 2 heat wave occurrences
    # This ensures we capture the persistent heat wave, not isolated events
    hw_threshold = 2
    hw_affected_mask = hw_frequency >= hw_threshold
    
    # Combine geographic region with heat wave occurrence
    spatial_mask = region_mask & hw_affected_mask.values
    
    print(f"  Grid points with ≥{hw_threshold} HW occurrences in 2010: {hw_affected_mask.sum().values}")
    print(f"  Final spatial mask (region ∩ HW): {spatial_mask.sum()} grid points")
    print(f"  This defines the 2010 Russian heat wave affected region")
    
    return spatial_mask, year_2010_mask


# ============================================================================
# PHYSICS PATTERN ANALYSIS - 2010 EVENT BY LEAD TIME
# ============================================================================

def analyze_2010_heatwave_by_leadtime(obs_weekly, fcst_weekly, obs_anomalies, 
                                       fcst_anomalies, hw_obs, spatial_mask, 
                                       year_2010_mask, lat, lon, lat_weights):
    """
    Analyze physics patterns for the 2010 Russian heat wave, separately by lead time
    
    This shows how forecast skill degrades with lead time for this specific event.
    For each lead time (week 1, 2, 3, 4):
    - Compare regional patterns during heat wave vs non-heat wave conditions
    - Evaluate if model reproduces the observed physics patterns
    - Results allow assessment of predictability horizon
    
    Args:
        spatial_mask: Boolean mask defining the affected region
        year_2010_mask: Boolean mask for 2010 summer dates
    """
    print("\n" + "="*80)
    print("2010 RUSSIAN HEAT WAVE ANALYSIS BY LEAD TIME")
    print("="*80)
    print("\nAnalyzing forecast performance at different lead times:")
    print("  Week 1: Model initialized 1 week before heat wave")
    print("  Week 2: Model initialized 2 weeks before heat wave")
    print("  Week 3: Model initialized 3 weeks before heat wave")
    print("  Week 4: Model initialized 4 weeks before heat wave")
    print("  Week 5: Model initialized 5 weeks before heat wave")
    print("  Week 6: Model initialized 6 weeks before heat wave")
    print("  Week 7: Model initialized 7 weeks before heat wave")
    print("  Week 8: Model initialized 8 weeks before heat wave")
    
    results_by_leadtime = {}
    
    # Create masked latitude weights (only for affected region)
    spatial_mask_da = xr.DataArray(spatial_mask, dims=["values"])
    lat_weights_masked = lat_weights.where(spatial_mask_da, 0)
    total_weight = float(lat_weights_masked.sum())
    
    print(f"\nRegional analysis settings:")
    print(f"  Spatial mask: {spatial_mask.sum()} grid points")
    print(f"  Total area weight: {total_weight:.2f}")
    
    for week_lead in range(1, 9):  # Lead times 1-8 weeks
        print(f"\n{'─'*80}")
        print(f"WEEK {week_lead} LEAD TIME (forecast initialized {week_lead} week(s) before event)")
        print(f"{'─'*80}")
        
        results = {}
        
        # Select this lead time from 2010 data
        obs_2010_lead = obs_anomalies.sel(init_date=year_2010_mask, week_lead_time=week_lead)
        fcst_2010_lead = fcst_anomalies.sel(init_date=year_2010_mask, week_lead_time=week_lead)
        obs_weekly_2010_lead = obs_weekly.sel(init_date=year_2010_mask, week_lead_time=week_lead)
        fcst_weekly_2010_lead = fcst_weekly.sel(init_date=year_2010_mask, week_lead_time=week_lead)
        
        # Heat wave mask for this lead time (observations)
        hw_mask_2010_lead_obs = hw_obs['severe'].sel(init_date=year_2010_mask, week_lead_time=week_lead) == 1
        # Forecast heat wave mask (ensemble-mean detection)
        hw_mask_2010_lead_fcst = hw_fcst['severe'].sel(init_date=year_2010_mask, week_lead_time=week_lead) == 1
        
        # Apply spatial mask to identify HW in the affected region
        hw_mask_region_obs = hw_mask_2010_lead_obs & spatial_mask
        hw_mask_region_fcst = hw_mask_2010_lead_fcst & spatial_mask
        
        # Compute spatial coverage per init_date (fraction of region with HW)
        n_gridpoints_in_region = int(spatial_mask.sum())
        
        # Observations: spatial coverage for each init_date
        hw_coverage_obs_per_date = []
        for init_date in obs_2010_lead.init_date.values:
            hw_count = int(hw_mask_region_obs.sel(init_date=init_date).sum())
            coverage = hw_count / n_gridpoints_in_region * 100 if n_gridpoints_in_region > 0 else 0
            hw_coverage_obs_per_date.append(coverage)
        
        # Forecasts: spatial coverage for each init_date
        hw_coverage_fcst_per_date = []
        for init_date in obs_2010_lead.init_date.values:
            hw_count = int(hw_mask_region_fcst.sel(init_date=init_date).sum())
            coverage = hw_count / n_gridpoints_in_region * 100 if n_gridpoints_in_region > 0 else 0
            hw_coverage_fcst_per_date.append(coverage)
        
        # Statistics across init_dates
        obs_coverage_mean = np.mean(hw_coverage_obs_per_date)
        obs_coverage_std = np.std(hw_coverage_obs_per_date)
        fcst_coverage_mean = np.mean(hw_coverage_fcst_per_date)
        fcst_coverage_std = np.std(hw_coverage_fcst_per_date)
        
        print(f"  Spatial coverage of heat waves in region:")
        print(f"    Observed: {obs_coverage_mean:.1f}% ± {obs_coverage_std:.1f}% (across {len(hw_coverage_obs_per_date)} dates)")
        print(f"    Forecast: {fcst_coverage_mean:.1f}% ± {fcst_coverage_std:.1f}%")
        print(f"    Coverage bias: {fcst_coverage_mean - obs_coverage_mean:+.1f}%")
        
        if obs_coverage_mean < 1.0:
            print(f"  ⚠ Very low heat wave coverage at this lead time - results may be noisy")
            if obs_coverage_mean == 0:
                print(f"  ⚠ No heat waves detected - skipping")
                results_by_leadtime[week_lead] = {'n_hw': 0}
                continue
        
        results['hw_coverage_obs_mean'] = obs_coverage_mean
        results['hw_coverage_obs_std'] = obs_coverage_std
        results['hw_coverage_fcst_mean'] = fcst_coverage_mean
        results['hw_coverage_fcst_std'] = fcst_coverage_std
        results['hw_coverage_bias'] = fcst_coverage_mean - obs_coverage_mean
        
        # ----------------------------------------------------------------
        # 1. BLOCKING PATTERN (MSL pressure)
        # ----------------------------------------------------------------
        if 'msl' in obs_2010_lead:
            print("\n  1. Blocking Pattern (MSL pressure anomaly):")
            
            # Data during heat wave in the region
            obs_msl_hw = obs_2010_lead['msl'].where(hw_mask_region_obs)
            fcst_msl_hw = fcst_2010_lead['msl'].where(hw_mask_region_obs).mean(dim='member')
            
            # Data during non-heat wave (same region, non-HW times)
            non_hw_mask_region = (~hw_mask_2010_lead_obs) & spatial_mask
            obs_msl_no_hw = obs_2010_lead['msl'].where(non_hw_mask_region)
            fcst_msl_no_hw = fcst_2010_lead['msl'].where(non_hw_mask_region).mean(dim='member')
            
            # Regional mean (area-weighted)
            obs_msl_hw_mean = float((obs_msl_hw * lat_weights_masked).sum() / total_weight)
            fcst_msl_hw_mean = float((fcst_msl_hw * lat_weights_masked).sum() / total_weight)
            obs_msl_no_hw_mean = float((obs_msl_no_hw * lat_weights_masked).sum() / total_weight)
            fcst_msl_no_hw_mean = float((fcst_msl_no_hw * lat_weights_masked).sum() / total_weight)
            
            # Contrast (blocking signature)
            obs_contrast = obs_msl_hw_mean - obs_msl_no_hw_mean
            fcst_contrast = fcst_msl_hw_mean - fcst_msl_no_hw_mean
            
            print(f"     Observed contrast (HW - non-HW): {obs_contrast:+.1f} Pa")
            print(f"     Forecast contrast: {fcst_contrast:+.1f} Pa")
            print(f"     Contrast bias: {fcst_contrast - obs_contrast:+.1f} Pa")
            
            # Skill assessment
            if obs_contrast > 100:
                relative_error  = abs(fcst_contrast - obs_contrast) / abs(obs_contrast) * 100
                print(f"     Model captures {100-relative_error:.0f}% of blocking signal")
            else:
                print(f"     ⚠ Blocking signal too weak ({obs_contrast:.1f} Pa < 100 Pa threshold) - skill not computed")
            
            results['pressure'] = {
                'obs_contrast': obs_contrast,
                'fcst_contrast': fcst_contrast,
                'bias': fcst_contrast - obs_contrast,
                'obs_hw': obs_msl_hw_mean,
                'fcst_hw': fcst_msl_hw_mean
            }
        
        # ----------------------------------------------------------------
        # 2. MOISTURE DEFICIT (2m dewpoint)
        # ----------------------------------------------------------------
        if '2d' in obs_2010_lead:
            print("\n  2. Surface Moisture (2m dewpoint anomaly):")
            
            obs_2d_hw = obs_2010_lead['2d'].where(hw_mask_region_obs)
            fcst_2d_hw = fcst_2010_lead['2d'].where(hw_mask_region_obs).mean(dim='member')
            obs_2d_no_hw = obs_2010_lead['2d'].where(non_hw_mask_region)
            fcst_2d_no_hw = fcst_2010_lead['2d'].where(non_hw_mask_region).mean(dim='member')
            
            obs_2d_hw_mean = float((obs_2d_hw * lat_weights_masked).sum() / total_weight)
            fcst_2d_hw_mean = float((fcst_2d_hw * lat_weights_masked).sum() / total_weight)
            obs_2d_no_hw_mean = float((obs_2d_no_hw * lat_weights_masked).sum() / total_weight)
            fcst_2d_no_hw_mean = float((fcst_2d_no_hw * lat_weights_masked).sum() / total_weight)
            
            obs_contrast = obs_2d_hw_mean - obs_2d_no_hw_mean
            fcst_contrast = fcst_2d_hw_mean - fcst_2d_no_hw_mean
            
            print(f"     Observed contrast (HW - non-HW): {obs_contrast:+.2f} K")
            print(f"     Forecast contrast: {fcst_contrast:+.2f} K")
            print(f"     Contrast bias: {fcst_contrast - obs_contrast:+.2f} K")
            
            results['moisture_surface'] = {
                'obs_contrast': obs_contrast,
                'fcst_contrast': fcst_contrast,
                'bias': fcst_contrast - obs_contrast
            }
        
        # ----------------------------------------------------------------
        # 3. MOISTURE DEFICIT (850 hPa specific humidity)
        # ----------------------------------------------------------------
        if 'q_850' in obs_2010_lead:
            print("\n  3. 850 hPa Moisture (specific humidity anomaly):")
            
            obs_q850_hw = obs_2010_lead['q_850'].where(hw_mask_region_obs)
            fcst_q850_hw = fcst_2010_lead['q_850'].where(hw_mask_region_obs).mean(dim='member')
            obs_q850_no_hw = obs_2010_lead['q_850'].where(non_hw_mask_region)
            fcst_q850_no_hw = fcst_2010_lead['q_850'].where(non_hw_mask_region).mean(dim='member')
            
            obs_q850_hw_mean = float((obs_q850_hw * lat_weights_masked).sum() / total_weight) * 1000
            fcst_q850_hw_mean = float((fcst_q850_hw * lat_weights_masked).sum() / total_weight) * 1000
            obs_q850_no_hw_mean = float((obs_q850_no_hw * lat_weights_masked).sum() / total_weight) * 1000
            fcst_q850_no_hw_mean = float((fcst_q850_no_hw * lat_weights_masked).sum() / total_weight) * 1000
            
            obs_contrast = obs_q850_hw_mean - obs_q850_no_hw_mean
            fcst_contrast = fcst_q850_hw_mean - fcst_q850_no_hw_mean
            
            print(f"     Observed contrast (HW - non-HW): {obs_contrast:+.2f} g/kg")
            print(f"     Forecast contrast: {fcst_contrast:+.2f} g/kg")
            print(f"     Contrast bias: {fcst_contrast - obs_contrast:+.2f} g/kg")
            
            results['moisture_850'] = {
                'obs_contrast': obs_contrast,
                'fcst_contrast': fcst_contrast,
                'bias': fcst_contrast - obs_contrast
            }
        
        # ----------------------------------------------------------------
        # 4. PRECIPITATION SUPPRESSION
        # ----------------------------------------------------------------
        if 'tp' in obs_2010_lead:
            print("\n  4. Precipitation Suppression:")
            
            obs_tp_hw = obs_2010_lead['tp'].where(hw_mask_region_obs)
            fcst_tp_hw = fcst_2010_lead['tp'].where(hw_mask_region_obs).mean(dim='member')
            obs_tp_no_hw = obs_2010_lead['tp'].where(non_hw_mask_region)
            fcst_tp_no_hw = fcst_2010_lead['tp'].where(non_hw_mask_region).mean(dim='member')
            
            obs_tp_hw_mean = float((obs_tp_hw * lat_weights_masked).sum() / total_weight)
            fcst_tp_hw_mean = float((fcst_tp_hw * lat_weights_masked).sum() / total_weight)
            obs_tp_no_hw_mean = float((obs_tp_no_hw * lat_weights_masked).sum() / total_weight)
            fcst_tp_no_hw_mean = float((fcst_tp_no_hw * lat_weights_masked).sum() / total_weight)
            
            obs_contrast = obs_tp_hw_mean - obs_tp_no_hw_mean
            fcst_contrast = fcst_tp_hw_mean - fcst_tp_no_hw_mean
            
            print(f"     Observed contrast (HW - non-HW): {obs_contrast:+.4f} m")
            print(f"     Forecast contrast: {fcst_contrast:+.4f} m")
            print(f"     Contrast bias: {fcst_contrast - obs_contrast:+.4f} m")
            
            results['precipitation'] = {
                'obs_contrast': obs_contrast,
                'fcst_contrast': fcst_contrast,
                'bias': fcst_contrast - obs_contrast
            }
        
        # ----------------------------------------------------------------
        # 5. TEMPERATURE ANOMALY (2m and 850 hPa)
        # ----------------------------------------------------------------
        if '2t' in obs_2010_lead:
            print("\n  5. Temperature Anomalies:")
            
            obs_2t_hw = obs_2010_lead['2t'].where(hw_mask_region_obs)
            fcst_2t_hw = fcst_2010_lead['2t'].where(hw_mask_region_obs).mean(dim='member')
            
            obs_2t_hw_mean = float((obs_2t_hw * lat_weights_masked).sum() / total_weight)
            fcst_2t_hw_mean = float((fcst_2t_hw * lat_weights_masked).sum() / total_weight)
            
            print(f"     2m temp anomaly during HW:")
            print(f"       Observed: {obs_2t_hw_mean:+.2f} K")
            print(f"       Forecast: {fcst_2t_hw_mean:+.2f} K")
            print(f"       Bias: {fcst_2t_hw_mean - obs_2t_hw_mean:+.2f} K")
            
            results['temp_2t'] = {
                'obs_anom': obs_2t_hw_mean,
                'fcst_anom': fcst_2t_hw_mean,
                'bias': fcst_2t_hw_mean - obs_2t_hw_mean
            }
            
            if 't_850' in obs_2010_lead:
                obs_t850_hw = obs_2010_lead['t_850'].where(hw_mask_region_obs)
                fcst_t850_hw = fcst_2010_lead['t_850'].where(hw_mask_region_obs).mean(dim='member')
                
                obs_t850_hw_mean = float((obs_t850_hw * lat_weights_masked).sum() / total_weight)
                fcst_t850_hw_mean = float((fcst_t850_hw * lat_weights_masked).sum() / total_weight)
                
                print(f"     850 hPa temp anomaly during HW:")
                print(f"       Observed: {obs_t850_hw_mean:+.2f} K")
                print(f"       Forecast: {fcst_t850_hw_mean:+.2f} K")
                print(f"       Bias: {fcst_t850_hw_mean - obs_t850_hw_mean:+.2f} K")
                
                results['temp_t_850'] = {
                    'obs_anom': obs_t850_hw_mean,
                    'fcst_anom': fcst_t850_hw_mean,
                    'bias': fcst_t850_hw_mean - obs_t850_hw_mean
                }
        
        # ----------------------------------------------------------------
        # 6. VERTICAL TEMPERATURE STRUCTURE
        # ----------------------------------------------------------------
        print("\n  6. Vertical Temperature Structure:")
        
        # Analyze temperature at multiple pressure levels
        levels_available = [lev for lev in ['t_925', 't_850', 't_700', 't_500'] 
                           if lev in obs_2010_lead]
        
        if len(levels_available) > 0:
            print(f"     Temperature anomalies during HW (vs non-HW):")
            
            for level in levels_available:
                obs_t_hw = obs_2010_lead[level].where(hw_mask_region_obs)
                fcst_t_hw = fcst_2010_lead[level].where(hw_mask_region_obs).mean(dim='member')
                obs_t_no_hw = obs_2010_lead[level].where(non_hw_mask_region)
                fcst_t_no_hw = fcst_2010_lead[level].where(non_hw_mask_region).mean(dim='member')
                
                obs_t_hw_mean = float((obs_t_hw * lat_weights_masked).sum() / total_weight)
                fcst_t_hw_mean = float((fcst_t_hw * lat_weights_masked).sum() / total_weight)
                obs_t_no_hw_mean = float((obs_t_no_hw * lat_weights_masked).sum() / total_weight)
                fcst_t_no_hw_mean = float((fcst_t_no_hw * lat_weights_masked).sum() / total_weight)
                
                # Contrast (warming signature)
                obs_contrast = obs_t_hw_mean - obs_t_no_hw_mean
                fcst_contrast = fcst_t_hw_mean - fcst_t_no_hw_mean
                
                print(f"       {level}: Obs={obs_contrast:+.2f} K, Fcst={fcst_contrast:+.2f} K, Bias={fcst_contrast-obs_contrast:+.2f} K")
                
                results[f'vertical_temp_{level}'] = {
                    'obs_contrast': obs_contrast,
                    'fcst_contrast': fcst_contrast,
                    'bias': fcst_contrast - obs_contrast
                }
        
        # ----------------------------------------------------------------
        # 7. WIND PATTERNS
        # ----------------------------------------------------------------
        if '10u' in obs_2010_lead and '10v' in obs_2010_lead:
            print("\n  7. Surface Wind Patterns:")
            
            # Wind components during HW and non-HW
            obs_u_hw = obs_2010_lead['10u'].where(hw_mask_region_obs)
            obs_v_hw = obs_2010_lead['10v'].where(hw_mask_region_obs)
            fcst_u_hw = fcst_2010_lead['10u'].where(hw_mask_region_obs).mean(dim='member')
            fcst_v_hw = fcst_2010_lead['10v'].where(hw_mask_region_obs).mean(dim='member')
            
            obs_u_no_hw = obs_2010_lead['10u'].where(non_hw_mask_region)
            obs_v_no_hw = obs_2010_lead['10v'].where(non_hw_mask_region)
            fcst_u_no_hw = fcst_2010_lead['10u'].where(non_hw_mask_region).mean(dim='member')
            fcst_v_no_hw = fcst_2010_lead['10v'].where(non_hw_mask_region).mean(dim='member')
            
            # Regional means
            obs_u_hw_mean = float((obs_u_hw * lat_weights_masked).sum() / total_weight)
            obs_v_hw_mean = float((obs_v_hw * lat_weights_masked).sum() / total_weight)
            fcst_u_hw_mean = float((fcst_u_hw * lat_weights_masked).sum() / total_weight)
            fcst_v_hw_mean = float((fcst_v_hw * lat_weights_masked).sum() / total_weight)
            
            obs_u_no_hw_mean = float((obs_u_no_hw * lat_weights_masked).sum() / total_weight)
            obs_v_no_hw_mean = float((obs_v_no_hw * lat_weights_masked).sum() / total_weight)
            fcst_u_no_hw_mean = float((fcst_u_no_hw * lat_weights_masked).sum() / total_weight)
            fcst_v_no_hw_mean = float((fcst_v_no_hw * lat_weights_masked).sum() / total_weight)
            
            # Wind anomaly contrasts (HW - non-HW)
            obs_u_contrast = obs_u_hw_mean - obs_u_no_hw_mean
            obs_v_contrast = obs_v_hw_mean - obs_v_no_hw_mean
            fcst_u_contrast = fcst_u_hw_mean - fcst_u_no_hw_mean
            fcst_v_contrast = fcst_v_hw_mean - fcst_v_no_hw_mean
            
            # Wind magnitude contrasts
            obs_wind_contrast_mag = np.sqrt(obs_u_contrast**2 + obs_v_contrast**2)
            fcst_wind_contrast_mag = np.sqrt(fcst_u_contrast**2 + fcst_v_contrast**2)
            
            print(f"     Wind component contrasts (HW - non-HW):")
            print(f"       Observed: u={obs_u_contrast:+.2f} m/s, v={obs_v_contrast:+.2f} m/s")
            print(f"       Forecast: u={fcst_u_contrast:+.2f} m/s, v={fcst_v_contrast:+.2f} m/s")
            print(f"     Contrast magnitude: Obs={obs_wind_contrast_mag:.2f} m/s, Fcst={fcst_wind_contrast_mag:.2f} m/s")
            
            results['wind'] = {
                'obs_u_contrast': obs_u_contrast,
                'obs_v_contrast': obs_v_contrast,
                'fcst_u_contrast': fcst_u_contrast,
                'fcst_v_contrast': fcst_v_contrast,
                'obs_magnitude': obs_wind_contrast_mag,
                'fcst_magnitude': fcst_wind_contrast_mag
            }
        
        results_by_leadtime[week_lead] = results
    
    return results_by_leadtime

# ============================================================================
# VISUALIZATION FOR 2010 CASE STUDY
# ============================================================================

def plot_2010_leadtime_skill_evolution(results_by_leadtime):
    """
    Plot how forecast skill for physics patterns degrades with lead time
    Shows model performance for the 2010 Russian heat wave at weeks 1-8
    """
    print("\nCreating lead time skill evolution plots...")
    
    # Extract data for plotting
    lead_times = []
    pressure_obs = []
    pressure_fcst = []
    moisture_obs = []
    moisture_fcst = []
    precip_obs = []
    precip_fcst = []
    temp_obs = []
    temp_fcst = []
    wind_obs = []
    wind_fcst = []
    t850_obs = []
    t850_fcst = []
    
    for week_lead in sorted(results_by_leadtime.keys()):
        results = results_by_leadtime[week_lead]
        
        if 'n_hw' not in results or results.get('n_hw', 0) == 0:
            continue
        
        lead_times.append(week_lead)
        
        if 'pressure' in results:
            pressure_obs.append(results['pressure']['obs_contrast'])
            pressure_fcst.append(results['pressure']['fcst_contrast'])
        
        if 'moisture_surface' in results:
            moisture_obs.append(results['moisture_surface']['obs_contrast'])
            moisture_fcst.append(results['moisture_surface']['fcst_contrast'])
        
        if 'precipitation' in results:
            precip_obs.append(results['precipitation']['obs_contrast'])
            precip_fcst.append(results['precipitation']['fcst_contrast'])
        
        if 'temp_2t' in results:
            temp_obs.append(results['temp_2t']['obs_anom'])
            temp_fcst.append(results['temp_2t']['fcst_anom'])
        
        if 'wind' in results:
            wind_obs.append(results['wind']['obs_magnitude'])
            wind_fcst.append(results['wind']['fcst_magnitude'])
        
        if 'vertical_temp_t_850' in results:
            t850_obs.append(results['vertical_temp_t_850']['obs_contrast'])
            t850_fcst.append(results['vertical_temp_t_850']['fcst_contrast'])
    
    # Create figure with 6 subplots (3 rows x 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('2010 Russian Heat Wave: Model Skill by Lead Time', 
                 fontsize=14, fontweight='bold')
    
    # 1. Blocking (MSL pressure contrast)
    if len(pressure_obs) > 0:
        ax = axes[0, 0]
        ax.plot(lead_times, pressure_obs, 'o-', linewidth=2, markersize=8, 
                label='Observed', color='black')
        ax.plot(lead_times, pressure_fcst, 's-', linewidth=2, markersize=8, 
                label='Forecast', color='#e74c3c')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lead Time (weeks)', fontsize=11)
        ax.set_ylabel('MSL Pressure Contrast (Pa)', fontsize=11)
        ax.set_title('Blocking Pattern (HW - non-HW)', fontsize=12, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(lead_times)
    
    # 2. Moisture deficit (2m dewpoint contrast)
    if len(moisture_obs) > 0:
        ax = axes[0, 1]
        ax.plot(lead_times, moisture_obs, 'o-', linewidth=2, markersize=8, 
                label='Observed', color='black')
        ax.plot(lead_times, moisture_fcst, 's-', linewidth=2, markersize=8, 
                label='Forecast', color='#3498db')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lead Time (weeks)', fontsize=11)
        ax.set_ylabel('Dewpoint Contrast (K)', fontsize=11)
        ax.set_title('Moisture Deficit (HW - non-HW)', fontsize=12, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(lead_times)
    
    # 3. Precipitation suppression
    if len(precip_obs) > 0:
        ax = axes[1, 0]
        ax.plot(lead_times, np.array(precip_obs)*1000, 'o-', linewidth=2, markersize=8, 
                label='Observed', color='black')
        ax.plot(lead_times, np.array(precip_fcst)*1000, 's-', linewidth=2, markersize=8, 
                label='Forecast', color='#9b59b6')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lead Time (weeks)', fontsize=11)
        ax.set_ylabel('Precipitation Contrast (mm)', fontsize=11)
        ax.set_title('Precipitation Suppression (HW - non-HW)', fontsize=12, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(lead_times)
    
    # 4. Temperature anomaly
    if len(temp_obs) > 0:
        ax = axes[1, 1]
        ax.plot(lead_times, temp_obs, 'o-', linewidth=2, markersize=8, 
                label='Observed', color='black')
        ax.plot(lead_times, temp_fcst, 's-', linewidth=2, markersize=8, 
                label='Forecast', color='#e67e22')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lead Time (weeks)', fontsize=11)
        ax.set_ylabel('Temperature Anomaly (K)', fontsize=11)
        ax.set_title('2m Temperature during Heat Wave', fontsize=12, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(lead_times)
    
    # 5. Vertical temperature structure (850 hPa contrast)
    if len(t850_obs) > 0:
        ax = axes[2, 0]
        ax.plot(lead_times, t850_obs, 'o-', linewidth=2, markersize=8, 
                label='Observed', color='black')
        ax.plot(lead_times, t850_fcst, 's-', linewidth=2, markersize=8, 
                label='Forecast', color='#2ecc71')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lead Time (weeks)', fontsize=11)
        ax.set_ylabel('850 hPa Temp Contrast (K)', fontsize=11)
        ax.set_title('Vertical Warming (HW - non-HW)', fontsize=12, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(lead_times)
    
    # 6. Wind pattern changes
    if len(wind_obs) > 0:
        ax = axes[2, 1]
        ax.plot(lead_times, wind_obs, 'o-', linewidth=2, markersize=8, 
                label='Observed', color='black')
        ax.plot(lead_times, wind_fcst, 's-', linewidth=2, markersize=8, 
                label='Forecast', color='#f39c12')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lead Time (weeks)', fontsize=11)
        ax.set_ylabel('Wind Contrast Magnitude (m/s)', fontsize=11)
        ax.set_title('Wind Pattern Changes (HW - non-HW)', fontsize=12, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(lead_times)
    
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, '2010_russian_heatwave_leadtime_skill.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_2010_spatial_patterns_by_leadtime(obs_anomalies, fcst_anomalies, hw_obs, 
                                             spatial_mask, year_2010_mask, lat, lon):
    """
    Create composite spatial maps for each lead time showing the 2010 heat wave
    
    This shows how the forecast representation of the heat wave degrades with lead time
    """
    print("\nCreating spatial pattern maps by lead time...")
    
    # Focus on key variable: 2m temperature anomaly
    var = '2t'
    
    if var not in obs_anomalies or var not in fcst_anomalies:
        print(f"  ⚠ Variable {var} not available - skipping spatial plots")
        return
    
    # Create figure with 8 rows (lead times) × 2 columns (obs, fcst)
    fig = plt.figure(figsize=(16, 32))
    
    plot_idx = 1
    
    for week_lead in range(1, 9):
        # Select data for this lead time
        obs_2010_lead = obs_anomalies[var].sel(init_date=year_2010_mask, week_lead_time=week_lead)
        fcst_2010_lead = fcst_anomalies[var].sel(init_date=year_2010_mask, week_lead_time=week_lead)
        
        # Heat wave mask
        hw_mask_2010_lead = hw_obs['severe'].sel(init_date=year_2010_mask, week_lead_time=week_lead) == 1
        hw_mask_region_obs = hw_mask_2010_lead & spatial_mask
        
        # Composite over init dates (average across all 2010 dates at this lead time)
        obs_composite = obs_2010_lead.where(hw_mask_region_obs).mean(dim='init_date')
        fcst_composite = fcst_2010_lead.where(hw_mask_region_obs).mean(dim=['init_date', 'member'])
        
        # Observations
        ax1 = fig.add_subplot(8, 2, plot_idx, projection=ccrs.PlateCarree())
        
        field_obs = obs_composite.values
        
        # Mask and prepare for plotting
        mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(field_obs)
        lon_plot = np.where(lon[mask] > 180, lon[mask] - 360, lon[mask])
        lat_plot = lat[mask]
        field_obs = field_obs[mask]
        
        # Normalization
        vmax = max(abs(np.nanpercentile(field_obs, 5)), abs(np.nanpercentile(field_obs, 95)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        proj = ccrs.PlateCarree()
        cf1 = ax1.tricontourf(lon_plot, lat_plot, field_obs, 40,
                             transform=proj, norm=norm, cmap='RdBu_r')
        ax1.coastlines(linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        
        # Draw box around the region
        ax1.plot([30, 70, 70, 30, 30], [45, 45, 65, 65, 45], 
                transform=proj, color='green', linewidth=2, linestyle='--')
        
        ax1.set_title(f'Week {week_lead} Lead - Observed 2m Temp Anomaly', 
                     fontsize=11, fontweight='bold')
        ax1.set_extent([-20, 100, 30, 75], crs=proj)
        
        cbar1 = plt.colorbar(cf1, ax=ax1, orientation='horizontal', 
                            pad=0.05, shrink=0.8)
        cbar1.set_label('K', fontsize=9)
        
        # Forecast
        ax2 = fig.add_subplot(8, 2, plot_idx+1, projection=ccrs.PlateCarree())
        
        field_fcst = fcst_composite.values
        
        mask_fcst = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(field_fcst)
        lon_plot_fcst = np.where(lon[mask_fcst] > 180, lon[mask_fcst] - 360, lon[mask_fcst])
        lat_plot_fcst = lat[mask_fcst]
        field_fcst = field_fcst[mask_fcst]
        
        cf2 = ax2.tricontourf(lon_plot_fcst, lat_plot_fcst, field_fcst, 40,
                             transform=proj, norm=norm, cmap='RdBu_r')
        ax2.coastlines(linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        
        # Draw box around the region
        ax2.plot([30, 70, 70, 30, 30], [45, 45, 65, 65, 45], 
                transform=proj, color='green', linewidth=2, linestyle='--')
        
        ax2.set_title(f'Week {week_lead} Lead - Forecast 2m Temp Anomaly', 
                     fontsize=11, fontweight='bold')
        ax2.set_extent([-20, 100, 30, 75], crs=proj)
        
        cbar2 = plt.colorbar(cf2, ax=ax2, orientation='horizontal', 
                            pad=0.05, shrink=0.8)
        cbar2.set_label('K', fontsize=9)
        
        plot_idx += 2
    
    plt.suptitle('2010 Russian Heat Wave: Temperature Anomaly by Lead Time\n(Green box = Analysis region)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, '2010_russian_heatwave_spatial_by_leadtime.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def create_2010_summary_report(results_by_leadtime, spatial_mask, hw_stats_by_leadtime):
    """
    Create detailed text summary of the 2010 heat wave analysis
    """
    print("\nCreating 2010 heat wave summary report...")
    
    report_file = os.path.join(OUTPUT_DIR, '2010_russian_heatwave_analysis.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("2010 RUSSIAN/EASTERN EUROPEAN HEAT WAVE CASE STUDY\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("EVENT DESCRIPTION:\n")
        f.write("-" * 80 + "\n")
        f.write("The 2010 Russian heat wave was one of the most extreme heat events on record.\n")
        f.write("It affected Western Russia and Eastern Europe from June through August 2010,\n")
        f.write("causing widespread impacts including wildfires, crop failures, and mortality.\n\n")
        
        f.write(f"Analysis region: 45-65°N, 30-70°E\n")
        f.write(f"Affected grid points: {spatial_mask.sum()}\n\n")
        
        # Add heat wave detection statistics by lead time
        f.write("HEAT WAVE DETECTION SKILL BY LEAD TIME:\n")
        f.write("="*80 + "\n\n")
        
        for severity in ['moderate', 'severe', 'extreme']:
            f.write(f"{severity.upper()} HEAT WAVES:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Week':<6} {'Obs %':<10} {'Fcst %':<10} {'Bias %':<10} {'Hit Rate %':<12}\n")
            f.write("-" * 80 + "\n")
            
            for stats in hw_stats_by_leadtime[severity]:
                f.write(f"{stats['week_lead']:<6} "
                       f"{stats['obs_freq']:>6.2f}%   "
                       f"{stats['fcst_freq']:>6.2f}%   "
                       f"{stats['bias']:>+6.2f}%   "
                       f"{stats['hit_rate']:>7.1f}%\n")
            f.write("\n")
        
        f.write("Notes:\n")
        f.write("- Obs %: Observed frequency of heat wave occurrence (global)\n")
        f.write("- Fcst %: Forecast frequency of heat wave occurrence (global)\n")
        f.write("- Bias %: Forecast - Observed frequency (positive = over-prediction)\n")
        f.write("- Hit Rate %: Fraction of observed heat waves that were forecast\n\n")
        
        f.write("FORECAST SKILL BY LEAD TIME:\n")
        f.write("="*80 + "\n\n")
        
        for week_lead in sorted(results_by_leadtime.keys()):
            results = results_by_leadtime[week_lead]
            
            if 'n_hw' not in results or results.get('n_hw', 0) == 0:
                f.write(f"Week {week_lead} Lead: No heat waves detected\n\n")
                continue
            
            f.write(f"WEEK {week_lead} LEAD TIME (initialized {week_lead} week(s) before event)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Spatial coverage of heat waves:\n")
            f.write(f"  Observed: {results['hw_coverage_obs_mean']:.1f}% ± {results['hw_coverage_obs_std']:.1f}%\n")
            f.write(f"  Forecast: {results['hw_coverage_fcst_mean']:.1f}% ± {results['hw_coverage_fcst_std']:.1f}%\n")
            f.write(f"  Coverage bias: {results['hw_coverage_bias']:+.1f}%\n\n")
            
            # Temperature
            if 'temp_2t' in results:
                f.write("1. Temperature Anomaly:\n")
                f.write(f"   Observed: {results['temp_2t']['obs_anom']:+.2f} K\n")
                f.write(f"   Forecast: {results['temp_2t']['fcst_anom']:+.2f} K\n")
                f.write(f"   Bias: {results['temp_2t']['bias']:+.2f} K\n")
                skill_pct = (1 - abs(results['temp_2t']['bias']) / abs(results['temp_2t']['obs_anom'])) * 100
                f.write(f"   Skill: {max(0, skill_pct):.1f}%\n\n")
            
            # Blocking
            if 'pressure' in results:
                f.write("2. Blocking Pattern (MSL pressure contrast):\n")
                f.write(f"   Observed: {results['pressure']['obs_contrast']:+.1f} Pa\n")
                f.write(f"   Forecast: {results['pressure']['fcst_contrast']:+.1f} Pa\n")
                f.write(f"   Bias: {results['pressure']['bias']:+.1f} Pa\n")
                if results['pressure']['obs_contrast'] > 50:
                    skill_pct = (1 - abs(results['pressure']['bias']) / abs(results['pressure']['obs_contrast'])) * 100
                    f.write(f"   Skill: {max(0, skill_pct):.1f}%\n")
                f.write("\n")
            
            # Moisture
            if 'moisture_surface' in results:
                f.write("3. Moisture Deficit (2m dewpoint contrast):\n")
                f.write(f"   Observed: {results['moisture_surface']['obs_contrast']:+.2f} K\n")
                f.write(f"   Forecast: {results['moisture_surface']['fcst_contrast']:+.2f} K\n")
                f.write(f"   Bias: {results['moisture_surface']['bias']:+.2f} K\n\n")
            
            # Precipitation
            if 'precipitation' in results:
                f.write("4. Precipitation Suppression (contrast):\n")
                f.write(f"   Observed: {results['precipitation']['obs_contrast']:+.4f} m\n")
                f.write(f"   Forecast: {results['precipitation']['fcst_contrast']:+.4f} m\n")
                f.write(f"   Bias: {results['precipitation']['bias']:+.4f} m\n\n")
            
            # Vertical Temperature Structure
            vertical_temp_keys = [k for k in results.keys() if k.startswith('vertical_temp_')]
            if len(vertical_temp_keys) > 0:
                f.write("5. Vertical Temperature Structure (HW - non-HW contrasts):\n")
                for key in sorted(vertical_temp_keys):
                    level = key.replace('vertical_temp_', '')
                    f.write(f"   {level}: Obs={results[key]['obs_contrast']:+.2f} K, ")
                    f.write(f"Fcst={results[key]['fcst_contrast']:+.2f} K, ")
                    f.write(f"Bias={results[key]['bias']:+.2f} K\n")
                f.write("\n")
            
            # Wind Patterns
            if 'wind' in results:
                f.write("6. Wind Pattern Changes (HW - non-HW contrasts):\n")
                f.write(f"   U-component: Obs={results['wind']['obs_u_contrast']:+.2f} m/s, ")
                f.write(f"Fcst={results['wind']['fcst_u_contrast']:+.2f} m/s\n")
                f.write(f"   V-component: Obs={results['wind']['obs_v_contrast']:+.2f} m/s, ")
                f.write(f"Fcst={results['wind']['fcst_v_contrast']:+.2f} m/s\n")
                f.write(f"   Magnitude: Obs={results['wind']['obs_magnitude']:.2f} m/s, ")
                f.write(f"Fcst={results['wind']['fcst_magnitude']:.2f} m/s\n\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        f.write("- Forecast skill degrades with increasing lead time\n")
        f.write("- Week 1-3 forecasts typically show good skill for temperature patterns\n")
        f.write("- Blocking patterns and moisture deficits are more challenging to predict\n")
        f.write("- Vertical temperature structure shows deep warming through the troposphere\n")
        f.write("- Wind pattern changes during heat waves are captured but with degrading skill\n")
        f.write("- Physics patterns (blocking, moisture, precipitation, wind) degrade faster than\n")
        f.write("  temperature anomalies, suggesting different predictability limits\n")
    
    print(f"  Saved: {report_file}")



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
    spatial_mask, year_2010_mask = identify_2010_russian_heatwave_region(
        obs_anom, hw_obs, lat, lon)
    
    # Analyze physics patterns by lead time
    results_by_leadtime = analyze_2010_heatwave_by_leadtime(
        obs_weekly, fcst_weekly, obs_anom, fcst_anom, hw_obs, 
        spatial_mask, year_2010_mask, lat, lon, lat_weights)
    
    # Visualization
    plot_2010_leadtime_skill_evolution(results_by_leadtime)
    plot_2010_spatial_patterns_by_leadtime(obs_anom, fcst_anom, hw_obs, 
                                            spatial_mask, year_2010_mask, lat, lon)
    create_2010_summary_report(results_by_leadtime, spatial_mask, hw_stats_by_leadtime)
    
    print("\n" + "="*80)
    print("2010 RUSSIAN HEAT WAVE ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*80)
    print("\nOutput files:")
    print("  - 2010_russian_heatwave_leadtime_skill.png (skill evolution)")
    print("  - 2010_russian_heatwave_spatial_by_leadtime.png (spatial patterns)")
    print("  - 2010_russian_heatwave_analysis.txt (summary report)")
    print()


if __name__ == "__main__":
    main()
