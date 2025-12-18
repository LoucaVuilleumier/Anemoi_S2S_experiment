import xarray as xr
import numpy as np
import re

inference_output_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_benchmark_july2022.nc"

ds_inference = xr.open_dataset(inference_output_path, engine="netcdf4")

# Variable weights from scaler configuration
weights = {
    'default': 1.0,
    'q': 0.6,
    't': 6.0,
    'u': 0.8,
    'v': 0.5,
    'w': 0.001,
    'z': 12.0,
    'sp': 10.0,
    '10u': 0.1,
    '10v': 0.1,
    '2d': 0.5,
    'tp': 0.025,
    'cp': 0.0025,
}

def get_pressure_level(var_name):
    """Extract pressure level from variable name (e.g., 'q500' -> 500)."""
    # Look for digits at the end of the variable name
    match = re.search(r'(\d+)$', var_name)
    if match:
        return int(match.group(1))
    return None

def get_pressure_scaler(var_name, y_intercept=0.2, slope=0.001):
    """Compute pressure level scaler: max(y_intercept, slope * pressure_level)."""
    pressure_level = get_pressure_level(var_name)
    if pressure_level is None:
        # Surface variable
        return 1.0
    else:
        # Pressure level variable
        return max(y_intercept, slope * pressure_level)

def get_variable_weight(var_name, weights_dict):
    """Get general variable weight, using default if not specified."""
    # Try exact match first
    if var_name in weights_dict:
        return weights_dict[var_name]
    
    # Try without level suffix (e.g., 'q500' -> 'q')
    base_name = ''.join([c for c in var_name if not c.isdigit()])
    if base_name in weights_dict:
        return weights_dict[base_name]
    
    # Return default
    return weights_dict['default']

def compute_total_weight(variables, weights_dict, y_intercept=0.2, slope=0.001):
    """Compute total weight sum for all variables including pressure scaler."""
    total = 0.0
    weights_per_var = {}
    
    for var in variables:
        general_weight = get_variable_weight(var, weights_dict)
        pressure_weight = get_pressure_scaler(var, y_intercept, slope)
        combined_weight = general_weight * pressure_weight
        
        weights_per_var[var] = {
            'general': general_weight,
            'pressure': pressure_weight,
            'combined': combined_weight
        }
        total += combined_weight
    
    return total, weights_per_var

# Get list of predicted variables from inference output
predicted_vars = list(ds_inference.data_vars.keys())
print(f"Total number of variables: {len(predicted_vars)}")
print(f"\nVariables: {predicted_vars}\n")

# Compute total weight
W_total, var_weights = compute_total_weight(predicted_vars, weights)

print(f"Total weight sum (W_total): {W_total:.4f}")
print(f"\nIndividual variable weights:")
for var, w_dict in sorted(var_weights.items(), key=lambda x: x[1]['combined'], reverse=True):
    relative_weight = w_dict['combined'] / W_total
    print(f"  {var:10s}: general={w_dict['general']:6.4f}, pressure={w_dict['pressure']:6.4f}, "
          f"combined={w_dict['combined']:8.4f}, relative={relative_weight:.6f}")

# Compute beta for different scenarios
print(f"\n{'='*60}")
print("Beta calculations (assuming physics_weight=1.0):")
print(f"{'='*60}")

w_2t_general = get_variable_weight('2t', weights)
w_2t_pressure = get_pressure_scaler('2t')
w_2t_combined = w_2t_general * w_2t_pressure

print(f"\nWeight of 2t:")
print(f"  General weight: {w_2t_general}")
print(f"  Pressure scaler: {w_2t_pressure}")
print(f"  Combined weight: {w_2t_combined}")
print(f"  Relative weight: {w_2t_combined / W_total:.6f}")

print(f"\nTo match weight of one variable with combined weight=1:")
print(f"  beta = 2 / W_total = 2 / {W_total:.4f} = {2/W_total:.6f}")

print(f"\nTo match weight of 2t specifically:")
print(f"  beta = 2 * w_2t_combined / W_total = 2 * {w_2t_combined} / {W_total:.4f} = {2*w_2t_combined/W_total:.6f}")