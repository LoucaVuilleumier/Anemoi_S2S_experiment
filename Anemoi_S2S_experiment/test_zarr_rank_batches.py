"""
Test for Zarr corruption by simulating which batches each rank would access.
Since rank 3 always fails, test the specific batch indices it would read.
"""

from anemoi.datasets import open_dataset
import numpy as np
from datetime import datetime

# Create output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"zarr_rank_batch_test_{timestamp}.log"

def log_print(message):
    """Print to console and write to file"""
    print(message)
    with open(output_file, 'a') as f:
        f.write(message + '\n')

log_print("=" * 80)
log_print("ZARR RANK-SPECIFIC BATCH CORRUPTION TEST")
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

log_print(f"Dataset shape: {ds_val.shape}")
total_timesteps = ds_val.shape[0]
total_variables = ds_val.shape[1]
total_gridpoints = ds_val.shape[2]

log_print(f"  Timesteps: {total_timesteps}")
log_print(f"  Variables: {total_variables}")
log_print(f"  Grid points: {total_gridpoints}")

# Simulate validation batch distribution across 4 GPUs
log_print("\n[2/3] Simulating batch distribution across ranks...")
num_ranks = 4
total_batches = 128  # From config: limit_batches validation: 128

# Each rank gets ~32 batches (128 / 4)
batches_per_rank = total_batches // num_ranks

log_print(f"Total batches: {total_batches}")
log_print(f"Batches per rank: {batches_per_rank}")
log_print(f"Dataset timesteps available: {total_timesteps}")

# Calculate which timesteps each rank would access
# With batch_size=1 and rollout=8, each batch reads 8 consecutive timesteps
rollout = 8
batch_stride = max(1, (total_timesteps - rollout) // total_batches)

log_print(f"Rollout: {rollout} steps")
log_print(f"Approximate batch stride: {batch_stride}")

failed_batches = []

for rank in range(num_ranks):
    start_batch = rank * batches_per_rank
    end_batch = (rank + 1) * batches_per_rank
    
    log_print(f"\n  Testing RANK {rank} (batches {start_batch}-{end_batch-1})...")
    
    # Test each batch this rank would handle
    for batch_idx in range(start_batch, min(end_batch, total_batches)):
        # Calculate timestep index for this batch
        timestep_idx = batch_idx * batch_stride
        
        # Make sure we don't go out of bounds
        if timestep_idx + rollout > total_timesteps:
            timestep_idx = total_timesteps - rollout
        
        try:
            # Read data as validation dataloader would: rollout consecutive timesteps, all variables, all grid points
            data = ds_val[timestep_idx:timestep_idx + rollout, :, :]
            
            if batch_idx % 8 == 0:  # Log every 8th batch to avoid spam
                log_print(f"    ✓ Batch {batch_idx} (timesteps {timestep_idx}-{timestep_idx+rollout-1}, "
                         f"dates {ds_val.dates[timestep_idx]} to {ds_val.dates[timestep_idx+rollout-1]}): OK")
                
        except RuntimeError as e:
            if "blosc decompression" in str(e):
                log_print(f"    ✗ Batch {batch_idx} (timesteps {timestep_idx}-{timestep_idx+rollout-1}): BLOSC ERROR!")
                log_print(f"       Dates: {ds_val.dates[timestep_idx]} to {ds_val.dates[timestep_idx+rollout-1]}")
                failed_batches.append((rank, batch_idx, timestep_idx, ds_val.dates[timestep_idx]))
            else:
                raise
        except Exception as e:
            log_print(f"    ✗ Batch {batch_idx} (timesteps {timestep_idx}-{timestep_idx+rollout-1}): {type(e).__name__}: {e}")
            failed_batches.append((rank, batch_idx, timestep_idx, ds_val.dates[timestep_idx]))

# Specifically test rank 3's batches in detail
log_print("\n[3/3] Detailed test of RANK 3 batches (where error occurs)...")
rank3_start_batch = 3 * batches_per_rank  # Should be 96
rank3_end_batch = 4 * batches_per_rank    # Should be 128

log_print(f"Testing all {batches_per_rank} batches for rank 3 (batches {rank3_start_batch}-{rank3_end_batch-1})...")

rank3_failures = []
for batch_idx in range(rank3_start_batch, rank3_end_batch):
    timestep_idx = batch_idx * batch_stride
    if timestep_idx + rollout > total_timesteps:
        timestep_idx = total_timesteps - rollout
    
    try:
        data = ds_val[timestep_idx:timestep_idx + rollout, :, :]
        log_print(f"  ✓ Batch {batch_idx}: timesteps {timestep_idx}-{timestep_idx+rollout-1} OK")
    except RuntimeError as e:
        if "blosc decompression" in str(e):
            log_print(f"  ✗ Batch {batch_idx}: timesteps {timestep_idx}-{timestep_idx+rollout-1} BLOSC ERROR!")
            log_print(f"     Date range: {ds_val.dates[timestep_idx]} to {ds_val.dates[timestep_idx+rollout-1]}")
            rank3_failures.append((batch_idx, timestep_idx, ds_val.dates[timestep_idx]))
        else:
            raise

# Summary
log_print("\n" + "=" * 80)
log_print("TEST SUMMARY")
log_print("=" * 80)

if failed_batches:
    log_print(f"\n❌ FOUND {len(failed_batches)} FAILURES:")
    for rank, batch, timestep, date in failed_batches:
        log_print(f"  Rank {rank}, Batch {batch}, Timestep {timestep}, Date {date}")
else:
    log_print("\n✓ NO FAILURES DETECTED")
    log_print("  This suggests the error might be:")
    log_print("  - Related to concurrent access by multiple workers")
    log_print("  - State-dependent (only after many training steps)")
    log_print("  - Network/filesystem issue during long-running jobs")

if rank3_failures:
    log_print(f"\n❌ RANK 3 had {len(rank3_failures)} specific failures")
    log_print("  Corrupted timestep ranges:")
    for batch, timestep, date in rank3_failures:
        log_print(f"    Timesteps {timestep}-{timestep+rollout-1}, starting at {date}")
else:
    log_print("\n✓ All rank 3 batches passed in this test")

log_print(f"\nTest completed at: {datetime.now()}")
log_print(f"Full log saved to: {output_file}")
