# Heat Wave Physics Analysis - Workflow

This directory contains scripts for comprehensive physics-based heat wave analysis.

## Files

1. **preprocess_heatwave_physics_data.py** - Preprocessing script
   - Loads raw 6-hourly observation data (26TB zarr file)
   - Loads raw daily forecast data (NetCDF files)
   - Computes weekly means for 16 variables (surface + upper air)
   - Saves to compressed NetCDF files
   - **Run this FIRST**

2. **heat_wave_analysis_enhanced.py** - Enhanced analysis script
   - Loads pre-computed weekly mean files
   - Performs comprehensive physics diagnostics
   - Creates spatial composite maps
   - **Run this SECOND (after preprocessing)**

3. **heat_wave_analysis.py** - Original analysis script (working)
   - Uses pre-computed 4-variable weekly files
   - Basic heat wave detection and skill analysis
   - This is the one that generated the current results

## Workflow

### Step 1: Run Preprocessing

```bash
cd /ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means
source /ec/res4/hpcperm/nld4584/venv_metrics_2026_04_02/bin/activate
python preprocess_heatwave_physics_data.py
```

**Expected output:**
- `/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/weekly_means_physics/Observations_weekly_AIFS_physics.nc`
- `/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/weekly_means_physics/Forecasts_weekly_AIFS_physics.nc`

**Note:** This step processes one initialization date at a time to avoid memory issues.

### Step 2: Run Enhanced Analysis

```bash
python heat_wave_analysis_enhanced.py
```

**Expected output:**
- Composite maps showing atmospheric patterns during heat waves:
  * Temperature anomalies
  * Pressure anomalies (blocking)
  * Moisture deficits (dewpoint, specific humidity)
  * Precipitation suppression
  * Upper-air temperature structure
- Physics summary reports for each severity level

## Variables Analyzed

**Surface variables:**
- 2t: 2m temperature
- 2d: 2m dewpoint (moisture)
- msl: Mean sea level pressure (blocking)
- sp: Surface pressure
- skt: Skin temperature
- tp: Total precipitation
- 10u, 10v: 10m winds

**Upper-air variables:**
- t_500, t_700, t_850, t_925: Temperature at pressure levels
- q_500, q_700, q_850, q_925: Specific humidity at pressure levels

## Physics Diagnostics

The enhanced script analyzes:

1. **Surface pressure anomalies** - Detect blocking patterns (high pressure dome)
2. **Moisture deficits** - Dewpoint depression and specific humidity anomalies
3. **Vertical temperature structure** - Deep warming through lower troposphere
4. **Precipitation suppression** - Reduced precipitation during heat waves
5. **Wind patterns** - Circulation anomalies

## Output Interpretation

**Composite maps:** Show averaged conditions across ALL heat wave occurrences (all initialization dates and all week lead times). This reveals the typical atmospheric pattern associated with heat waves.

**Lead time information:** The composite averages across weeks 1-8. For lead-time-specific analysis, you would need to modify the plotting function to loop over `week_lead_time` and create separate maps for each week.

## Memory Considerations

- **Preprocessing script:** Processes one init date at a time → manageable memory usage
- **Enhanced script:** Loads pre-computed weekly files → fast and memory-efficient
- **Original approach (failed):** Loading 26TB zarr all at once → 15GB memory kill

## Next Steps

If you want lead-time-specific physics analysis, we can modify the enhanced script to create separate composite maps for:
- Week 1 only
- Week 2 only
- etc.

Or create a time evolution showing how physics patterns change with lead time.
