"""
Diagnostic script to check date alignment between reference and finetuned models
"""
import xarray as xr
import numpy as np
import glob

print("="*80)
print("CHECKING DATE ALIGNMENT BETWEEN REFERENCE AND FINETUNED MODELS")
print("="*80)

# Load one example init date from reference model (daily outputs)
paths_init_ref = sorted(glob.glob(
    "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/AIFS/"
    "aifs-subs-pretrain-*"
))

if len(paths_init_ref) > 0:
    # Load first init date, first member
    path_ref = paths_init_ref[0]
    print(f"\n1. REFERENCE MODEL (daily outputs):")
    print(f"   Path: {path_ref}")
    
    nc_files = sorted(glob.glob(f"{path_ref}/*.nc"))
    if len(nc_files) > 0:
        ds_ref = xr.open_dataset(nc_files[0])
        
        init_date = ds_ref.time.values[0]
        print(f"   Init date: {init_date}")
        print(f"   First 10 forecast times: {ds_ref.time.values[:10]}")
        print(f"   Total timesteps: {len(ds_ref.time)}")
        
        # Calculate days from init
        days_from_init = (ds_ref.time.values - init_date) / np.timedelta64(1, 'D')
        print(f"   Days from init (first 10): {days_from_init[:10]}")
        print(f"   Days from init (last): {days_from_init[-1]}")
        
        # After removing first timestep
        print(f"\n   After removing init timestep:")
        print(f"   First 10 days: {days_from_init[1:11]}")
        
        # First week would be
        print(f"\n   First week (days 1-7) includes forecast times:")
        week1_mask = (days_from_init >= 1) & (days_from_init <= 7)
        print(f"   {ds_ref.time.values[week1_mask]}")
        print(f"   (Days: {days_from_init[week1_mask]})")

# Load one example init date from finetuned model (weekly outputs)
paths_init_ft = sorted(glob.glob(
    "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/Weekly_Means_14k_lr_0.625e-7/"
    "aifs-subs-pretrain-weeklymeans-*"
))

if len(paths_init_ft) > 0:
    # Load first init date, first member
    path_ft = paths_init_ft[0]
    print(f"\n2. FINETUNED MODEL (weekly outputs):")
    print(f"   Path: {path_ft}")
    
    nc_files = sorted(glob.glob(f"{path_ft}/*.nc"))
    if len(nc_files) > 0:
        ds_ft = xr.open_dataset(nc_files[0])
        
        init_date_ft = ds_ft.time.values[0]
        print(f"   Init date: {init_date_ft}")
        print(f"   Forecast times: {ds_ft.time.values}")
        print(f"   Total timesteps: {len(ds_ft.time)}")
        
        # Calculate days from init
        days_from_init_ft = (ds_ft.time.values - init_date_ft) / np.timedelta64(1, 'D')
        print(f"   Days from init: {days_from_init_ft}")
        
        # After removing first timestep
        print(f"\n   After removing init timestep:")
        print(f"   Remaining forecast times: {ds_ft.time.values[1:]}")
        print(f"   Remaining days from init: {days_from_init_ft[1:]}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

print("""
KEY QUESTION: What time period does each weekly forecast represent?

Reference model (after processing):
- Week 1: average of daily forecasts at days [1, 2, 3, 4, 5, 6, 7]
- Week 2: average of daily forecasts at days [8, 9, 10, 11, 12, 13, 14]

Finetuned model:
- First output (after removing init) is at day 7
- Does this represent:
  (A) Average of days [0, 1, 2, 3, 4, 5, 6] - includes init day 0
  (B) Average of days [1, 2, 3, 4, 5, 6, 7] - matches reference week 1
  (C) Average of days [0, 1, 2, 3, 4, 5, 6, 7] - 8-day average
  
If (A): MISALIGNED - finetuned week 1 includes day 0, reference doesn't
If (B): ALIGNED - both represent the same time period
If (C): MISALIGNED - different number of days in average

To determine which, check:
1. The training setup - how were target weekly means computed?
2. The model's autoregressive step size - is it exactly 7 days?
3. Compare actual values - do they match expected climatology/patterns?
""")

print("\nRECOMMENDATION:")
print("  Check a sample variable (e.g., 2t) and compare:")
print("  1. Reference model week 1 value at a specific grid point")
print("  2. Finetuned model week 1 value at the same grid point")
print("  3. Manually compute observations average for days [0-7] and days [1-7]")
print("  4. See which observations match better with finetuned model")
