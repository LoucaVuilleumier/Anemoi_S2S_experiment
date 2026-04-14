from omegaconf import OmegaConf
from anemoi.datasets import open_dataset
import numpy as np
import sys
from datetime import datetime

# Create output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"zarr_corruption_test_{timestamp}.log"

def log_print(message):
    """Print to console and write to file"""
    print(message)
    with open(output_file, 'a') as f:
        f.write(message + '\n')

log_print("=" * 80)
log_print("ZARR CHUNK CORRUPTION TEST")
log_print(f"Started at: {datetime.now()}")
log_print(f"Output file: {output_file}")
log_print("=" * 80)

# Open validation dataset
log_print("\n[1/3] Opening validation dataset...")
ds_val = open_dataset(
    {"dataset": [
      {"join": [
      {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8",
        "start": 2007,
        "end": 2011,
        "rolling_average": [-28, 0, 'frequency']},
      {"dataset": "aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v2-s2s-predictors",
        "start": 2007,
        "end": 2011,
        "rolling_average": [-28, 0, 'frequency']},
      {"dataset": "aifs-ea-an-oper-0001-mars-o96-1950-2025-6h-v1-accumulation168h",
        "start": '2007-01-08',
        "end": 2011}
    ]}]}
)

log_print(f"Dataset length: {len(ds_val)}")
log_print(f"Date range: {ds_val.dates[0]} to {ds_val.dates[-1]}")
log_print(f"Shape: {ds_val.shape}")

# Test 1: Sequential access
log_print("\n[2/3] Testing sequential access...")
failed_indices = []
test_indices = range(0, len(ds_val), 1)

for i, idx in enumerate(test_indices):
    try:
        data = ds_val[idx, ...]
        if i % 100 == 0:
            log_print(f"  ✓ Index {idx} ({ds_val.dates[idx]}): OK")
    except RuntimeError as e:
        if "blosc decompression" in str(e):
            log_print(f"  ✗ Index {idx} ({ds_val.dates[idx]}): BLOSC ERROR!")
            failed_indices.append((idx, ds_val.dates[idx], str(e)))
        else:
            raise
    except Exception as e:
        log_print(f"  ✗ Index {idx} ({ds_val.dates[idx]}): {type(e).__name__}: {e}")
        failed_indices.append((idx, ds_val.dates[idx], str(e)))

# Test 2: Random sampling to catch edge cases
log_print("\n[3/3] Testing random sampling (50 samples)...")
np.random.seed(42)
random_indices = np.random.choice(len(ds_val), size=min(50, len(ds_val)), replace=False)

for i, idx in enumerate(random_indices):
    idx = int(idx)  # Convert numpy.int64 to Python int
    try:
        data = ds_val[idx, ...]
        if i < 5: #only log the first few random samples to avoid clutter
            log_print(f"  ✓ Random index {idx} ({ds_val.dates[idx]}): OK")
    except RuntimeError as e: #but print all errors to catch patterns
        if "blosc decompression" in str(e):
            log_print(f"  ✗ Random index {idx} ({ds_val.dates[idx]}): BLOSC ERROR!")
            if (idx, ds_val.dates[idx], str(e)) not in failed_indices:
                failed_indices.append((idx, ds_val.dates[idx], str(e)))
        else:
            raise
    except Exception as e:
        log_print(f"  ✗ Random index {idx} ({ds_val.dates[idx]}): {type(e).__name__}: {e}")
        if (idx, ds_val.dates[idx], str(e)) not in failed_indices:
            failed_indices.append((idx, ds_val.dates[idx], str(e)))

# Summary
log_print("\n" + "=" * 80)
log_print("RESULTS SUMMARY")
log_print("=" * 80)

if failed_indices:
    log_print(f"\n❌ FOUND {len(failed_indices)} CORRUPTED CHUNKS!\n")
    for idx, date, error in failed_indices:
        log_print(f"  Index {idx} ({date}):")
        log_print(f"    Error: {error[:100]}...")
else:
    log_print("\n✅ All tested chunks are OK!")
    log_print("   The corruption might be:")
    log_print("   - In specific grid shards (test didn't check all spatial points)")
    log_print("   - In specific variables")
    log_print("   - Happening during multi-process access (run with multiple workers)")

log_print("\n" + "=" * 80)
log_print(f"Test completed at: {datetime.now()}")
log_print(f"Results saved to: {output_file}")
log_print("=" * 80)

