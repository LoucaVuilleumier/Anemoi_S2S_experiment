import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import plotting_functions as pf
from utils.physics_functions import compute_r_sur, compute_rh_sur
from utils import metrics_function as mf
from collections import defaultdict
import importlib
importlib.reload(pf)
importlib.reload(mf)

#paths

inference_nofinetuning_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_benchmark_july2022.nc"
inference_default_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_10k_default.nc"
inference_PINNLoss_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_10k_PINNLoss.nc"
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"

#open inference datasets
ds_inference_nofinetuning = xr.open_dataset(inference_nofinetuning_path)
ds_inference_default = xr.open_dataset(inference_default_path)
ds_inference_PINNLoss = xr.open_dataset(inference_PINNLoss_path)

#open dataset and align time with inference datasets
n_steps = len(ds_inference_default.time)
ds_whole_dataset = xr.open_zarr(dataset_path)
times = ds_whole_dataset.dates.values
var_names = ds_whole_dataset.attrs["variables"]
init_time = np.datetime64("2022-07-01T00:00")
t0 = np.where(times == init_time)[0][0]
ds_dataset = ds_whole_dataset.isel(time = slice(t0, t0 + n_steps))

var_list = ['2t', '2d', 'sp'] 
def get_var_dataset(dataset, variable):
    idx_variable = var_names.index(variable)
    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})


ds_dataset = xr.Dataset({var: get_var_dataset(ds_dataset, var) for var in var_list})


list_datasets = [ds_inference_nofinetuning, ds_inference_default, ds_inference_PINNLoss, ds_dataset]

#compute r and rh for each dataset
for dataset in list_datasets:
    #compute r_sur
    t2m = dataset["2t"] - 273.15  # Convert from Kelvin to Celsius
    dp2m = dataset["2d"] - 273.15  # Convert from Kelvin to Celsius
    sp = dataset["sp"] / 100.0  # Convert from Pa to hPa
    r_sur = compute_r_sur(t2m, dp2m, sp)
    dataset["r_sur"] = r_sur

    #compute rh_sur
    rh_sur = compute_rh_sur(t2m, dp2m, clip_for_plot=False)
    dataset["rh_sur"] = rh_sur
    

    
#compute MAE for the different inferences and different variables
var_list_metrics = ["2t", "2d", "sp", "r_sur", "rh_sur"]
mae_results = defaultdict(dict)
for inference in list_datasets[:-1]:  # Exclude the original dataset
    for var in var_list_metrics:
        mae = mf.compute_mae_from_dataset(ds_dataset, inference, variable=var)
        if inference is ds_inference_nofinetuning:
            mae_results['No Finetuning'][var] = mae
        elif inference is ds_inference_default:
            mae_results['Default Finetuning'][var] = mae
        elif inference is ds_inference_PINNLoss:
            mae_results['PINNLoss Finetuning'][var] = mae

# Convert to DataFrame and export
mae_df = pd.DataFrame(mae_results).T
print("\nMAE Results:")
print(mae_df.to_string())

# Save to CSV
mae_df.to_csv('images/mae_results.csv')

# Create sub-datasets with random lat/lon regions for regional MSSS analysis
n_regions = 20  # Number of random regions to sample
region_size = 1000  # Number of grid points per region

# Get all available grid points
n_total_points = ds_dataset.dims['values']


# Randomly select starting indices for each region
np.random.seed(42)  # For reproducibility
region_indices = []
for i in range(n_regions):
    start_idx = np.random.randint(0, max(1, n_total_points - region_size))
    indices = np.arange(start_idx, min(start_idx + region_size, n_total_points))
    region_indices.append(indices)

# Compute MSSS for each region
msss_PINN_regional = {var: [] for var in var_list_metrics}
msss_default_regional = {var: [] for var in var_list_metrics}

mae_PINN_regional = {var: [] for var in var_list_metrics}
mae_default_regional = {var: [] for var in var_list_metrics}
mae_nofinetuning_regional = {var: [] for var in var_list_metrics}

for i, indices in enumerate(region_indices):
    # Select sub-region for all datasets
    ds_dataset_region = ds_dataset.isel(values=indices)
    ds_inference_nofinetuning_region = ds_inference_nofinetuning.isel(values=indices)
    ds_inference_default_region = ds_inference_default.isel(values=indices)
    ds_inference_PINNLoss_region = ds_inference_PINNLoss.isel(values=indices)
    
    # Compute MSSS for this region (returns dict with variable as keys)
    msss_PINN = mf.compute_msss(
        ds_dataset_region, 
        ds_inference_nofinetuning_region, 
        ds_inference_PINNLoss_region, 
        variable=var_list_metrics, 
        squash=False
    )
    msss_default = mf.compute_msss(
        ds_dataset_region, 
        ds_inference_nofinetuning_region, 
        ds_inference_default_region, 
        variable=var_list_metrics, 
        squash=False
    )
    
    #Compute MAE for this region
    mae_PINN = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_PINNLoss_region, variable=var_list_metrics, squash=False)
    mae_default = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_default_region, variable=var_list_metrics, squash=False)
    mae_nofinetuning = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_nofinetuning_region, variable=var_list_metrics, squash=False)
    
    # Organize by variable
    for var in var_list_metrics:
        msss_PINN_regional[var].append(msss_PINN[var])
        msss_default_regional[var].append(msss_default[var])
        
        mae_PINN_regional[var].append(mae_PINN[var])
        mae_default_regional[var].append(mae_default[var])
        mae_nofinetuning_regional[var].append(mae_nofinetuning[var])

#MSSS Boxplots
# Prepare data for boxplot - one subplot per variable
boxplot_msss = {}
for var in var_list_metrics:
    boxplot_msss[var] = {
        'PINNLoss Finetuning': msss_PINN_regional[var],
        'Default Finetuning': msss_default_regional[var]
    }

# Define colors
colors = {
    'PINNLoss Finetuning': '#FF6B6B',
    'Default Finetuning': '#4ECDC4'
}

# Create boxplot
pf.plot_boxplots(
    data_dict=boxplot_msss,
    colors=colors,
    savename='msss_regional_boxplot.png',
    ylabel='MSSS Score'
)


#MAE Boxplots
# Prepare data for boxplot - one subplot per variable
boxplot_mae = {}
for var in var_list_metrics:
    boxplot_mae[var] = {
        'PINNLoss Finetuning': mae_PINN_regional[var],
        'Default Finetuning': mae_default_regional[var],
        'No Finetuning': mae_nofinetuning_regional[var]
    }

# Define colors for MAE
colors_mae = {
    'PINNLoss Finetuning': '#FF6B6B',
    'Default Finetuning': '#4ECDC4',
    'No Finetuning': '#FFD93D'
}

# Create boxplot for MAE
pf.plot_boxplots(
    data_dict=boxplot_mae,
    colors=colors_mae,
    savename='mae_regional_boxplot.png',
    ylabel='Mean Absolute Error',
    sharey = False
)