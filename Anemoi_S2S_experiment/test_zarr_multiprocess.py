"""
Test Zarr dataset with multiprocessing to simulate actual dataloader behavior.
This is closer to what happens during validation with num_workers=8.
"""

from anemoi.datasets import open_dataset
import numpy as np
from datetime import datetime
from multiprocessing import Pool, Manager
import traceback

# Create output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"zarr_multiproc_test_{timestamp}.log"

def log_print(message, log_file=output_file):
    """Print to console and write to file"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def worker_read_data(args):
    """Worker function to read data from a specific shard"""
    worker_id, start_idx, end_idx, num_samples = args
    
    errors = []
    success_count = 0
    
    try:
        # Each worker opens its own dataset
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
        
        # Read samples from this worker's assignment
        for i in range(num_samples):
            time_idx = (worker_id * num_samples + i) % len(ds_val)
            try:
                # Read a specific grid shard (simulating GPU rank)
                data = ds_val[time_idx, :, start_idx:end_idx]
                success_count += 1
            except RuntimeError as e:
                if "blosc decompression" in str(e):
                    errors.append({
                        'worker': worker_id,
                        'time_idx': time_idx,
                        'date': str(ds_val.dates[time_idx]),
                        'error': 'BLOSC DECOMPRESSION ERROR',
                        'trace': traceback.format_exc()
                    })
                else:
                    raise
            except Exception as e:
                errors.append({
                    'worker': worker_id,
                    'time_idx': time_idx,
                    'date': str(ds_val.dates[time_idx]) if time_idx < len(ds_val) else 'N/A',
                    'error': f'{type(e).__name__}: {str(e)}',
                    'trace': traceback.format_exc()
                })
        
        return {
            'worker_id': worker_id,
            'success_count': success_count,
            'errors': errors
        }
        
    except Exception as e:
        return {
            'worker_id': worker_id,
            'success_count': 0,
            'errors': [{
                'worker': worker_id,
                'error': f'Worker failed to initialize: {type(e).__name__}: {str(e)}',
                'trace': traceback.format_exc()
            }]
        }

def main():
    log_print("=" * 80)
    log_print("ZARR MULTIPROCESS CORRUPTION TEST")
    log_print(f"Started at: {datetime.now()}")
    log_print(f"Output file: {output_file}")
    log_print("=" * 80)
    
    # Configuration
    num_workers = 8  # Same as validation config
    num_samples_per_worker = 20  # Each worker reads 20 samples
    num_gpu_shards = 4
    
    log_print(f"\nConfiguration:")
    log_print(f"  Number of workers: {num_workers}")
    log_print(f"  Samples per worker: {num_samples_per_worker}")
    log_print(f"  Total samples: {num_workers * num_samples_per_worker}")
    log_print(f"  GPU shards: {num_gpu_shards}")
    
    # Get dataset info
    log_print("\nOpening dataset to get dimensions...")
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
    
    total_gridpoints = ds_val.shape[2]
    shard_size = total_gridpoints // num_gpu_shards
    
    log_print(f"Dataset shape: {ds_val.shape}")
    log_print(f"Grid points per shard: {shard_size}")
    
    # Test each GPU shard with multiple workers
    for gpu_rank in range(num_gpu_shards):
        start_idx = gpu_rank * shard_size
        end_idx = (gpu_rank + 1) * shard_size if gpu_rank < num_gpu_shards - 1 else total_gridpoints
        
        log_print(f"\n{'=' * 80}")
        log_print(f"Testing GPU Rank {gpu_rank} (grid points {start_idx}:{end_idx})")
        log_print(f"{'=' * 80}")
        
        # Prepare worker arguments
        worker_args = [
            (worker_id, start_idx, end_idx, num_samples_per_worker)
            for worker_id in range(num_workers)
        ]
        
        log_print(f"Launching {num_workers} workers...")
        
        # Run workers in parallel
        with Pool(processes=num_workers) as pool:
            results = pool.map(worker_read_data, worker_args)
        
        # Collect results
        total_success = sum(r['success_count'] for r in results)
        total_errors = sum(len(r['errors']) for r in results)
        
        log_print(f"\nGPU Rank {gpu_rank} Results:")
        log_print(f"  Successful reads: {total_success}")
        log_print(f"  Failed reads: {total_errors}")
        
        # Report errors
        if total_errors > 0:
            log_print(f"\n  ❌ ERRORS DETECTED:")
            for result in results:
                for error in result['errors']:
                    log_print(f"\n    Worker {error['worker']}:")
                    log_print(f"      Time index: {error.get('time_idx', 'N/A')}")
                    log_print(f"      Date: {error.get('date', 'N/A')}")
                    log_print(f"      Error: {error['error']}")
                    if 'trace' in error:
                        log_print(f"      Traceback:\n{error['trace']}")
        else:
            log_print(f"  ✅ All reads successful!")
    
    # Summary
    log_print("\n" + "=" * 80)
    log_print("OVERALL SUMMARY")
    log_print("=" * 80)
    log_print(f"Test completed at: {datetime.now()}")
    log_print(f"Results saved to: {output_file}")
    log_print("=" * 80)

if __name__ == "__main__":
    main()
