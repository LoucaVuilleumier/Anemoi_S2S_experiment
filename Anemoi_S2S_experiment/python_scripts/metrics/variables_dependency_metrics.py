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

inference_nofinetuning_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_refmodel_6weeksjuly2022.nc"
inference_default_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_10k_default_6weeksjuly2022.nc"
inference_PINNLoss_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_10k_PINNLoss_6weeksjuly2022.nc"
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"

#open inference datasets
ds_inference_nofinetuning = xr.open_dataset(inference_nofinetuning_path)
ds_inference_default = xr.open_dataset(inference_default_path)
ds_inference_PINNLoss = xr.open_dataset(inference_PINNLoss_path)

#daily average inference datasets
ds_inference_nofinetuning_daily_avg = ds_inference_nofinetuning.coarsen(time=4, boundary='trim').mean()
ds_inference_default_daily_avg = ds_inference_default.coarsen(time=4, boundary='trim').mean()
ds_inference_PINNLoss_daily_avg = ds_inference_PINNLoss.coarsen(time=4, boundary='trim').mean()

#weekly average inference datasets
ds_inference_nofinetuning_weekly_avg = ds_inference_nofinetuning.coarsen(time=28, boundary='trim').mean()
ds_inference_default_weekly_avg = ds_inference_default.coarsen(time=28, boundary='trim').mean()
ds_inference_PINNLoss_weekly_avg = ds_inference_PINNLoss.coarsen(time=28, boundary='trim').mean()

#dictionary to access datasets easily
dict_inference = {
    "No Finetuning": {
        "6h": ds_inference_nofinetuning,
        "daily_avg": ds_inference_nofinetuning_daily_avg,
        "weekly_avg": ds_inference_nofinetuning_weekly_avg},
    "Default Finetuning": {
        "6h": ds_inference_default,
        "daily_avg": ds_inference_default_daily_avg,
        "weekly_avg": ds_inference_default_weekly_avg},
    "PINNLoss Finetuning": {
        "6h": ds_inference_PINNLoss,
        "daily_avg": ds_inference_PINNLoss_daily_avg,
        "weekly_avg": ds_inference_PINNLoss_weekly_avg}
}



#open dataset and align time with inference datasets
n_steps = len(ds_inference_default.time)
ds_whole_dataset = xr.open_zarr(dataset_path)
times = ds_whole_dataset.dates.values
var_names = ds_whole_dataset.attrs["variables"]
init_time = np.datetime64("2022-07-01T00:00")
t0 = np.where(times == init_time)[0][0]
ds_dataset_sliced = ds_whole_dataset.isel(time = slice(t0, t0 + n_steps))

var_list = ['2t', '2d', 'sp'] 
def get_var_dataset(dataset, variable):
    idx_variable = var_names.index(variable)
    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})


#create dataset with only required variables
ds_dataset = xr.Dataset({var: get_var_dataset(ds_dataset_sliced, var) for var in var_list})
#daily average dataset
ds_dataset_daily_avg = ds_dataset.coarsen(time=4, boundary='trim').mean()
#weekly average dataset
ds_dataset_weekly_avg = ds_dataset.coarsen(time=28, boundary='trim').mean()

#dictionary to access datasets easily
dict_dataset = {
    "6h": ds_dataset,
    "daily_avg": ds_dataset_daily_avg,
    "weekly_avg": ds_dataset_weekly_avg
}

#compute r_sur and rh_sur for all inference datasets
finetuning_types = ["No Finetuning", "Default Finetuning", "PINNLoss Finetuning"]
time_resolutions = ["6h", "daily_avg", "weekly_avg"]

for finetuning in finetuning_types:
    for time_res in time_resolutions:
        ds_inference = dict_inference[finetuning][time_res]
        t2m = ds_inference["2t"] - 273.15  # Convert from Kelvin to Celsius
        dp2m = ds_inference["2d"] - 273.15  # Convert from Kelvin to Celsius
        sp = ds_inference["sp"] / 100.0  # Convert from Pa to hPa
        
        #compute r_sur and add to corresponding inference dataset
        r_sur = compute_r_sur(t2m, dp2m, sp)
        ds_inference["r_sur"] = r_sur
        
        #compute rh_sur and add to corresponding inference dataset
        rh_sur = compute_rh_sur(t2m, dp2m, clip_for_plot=False)
        ds_inference["rh_sur"] = rh_sur
        
for time_res in time_resolutions:
    ds_dataset = dict_dataset[time_res]
    t2m = ds_dataset["2t"] - 273.15  # Convert from Kelvin to Celsius
    dp2m = ds_dataset["2d"] - 273.15  # Convert from Kelvin to Celsius
    sp = ds_dataset["sp"] / 100.0  # Convert from Pa to hPa
    
    #compute r_sur and add to corresponding dataset
    r_sur = compute_r_sur(t2m, dp2m, sp)
    ds_dataset["r_sur"] = r_sur
    
    #compute rh_sur and add to corresponding dataset
    rh_sur = compute_rh_sur(t2m, dp2m, clip_for_plot=False)
    ds_dataset["rh_sur"] = rh_sur
        
var_list_metrics = ["2t", "2d", "sp", "r_sur", "rh_sur"]

#Compute mae
mae= defaultdict(dict)
mae_time_series = defaultdict(dict)
for finetuning in finetuning_types:
    for time_res in time_resolutions:
        ds_inference = dict_inference[finetuning][time_res]
        ds_dataset = dict_dataset[time_res]
        
        mae[finetuning][time_res] = mf.compute_mae_from_dataset(ds_dataset, ds_inference, variable=var_list_metrics)
        mae_time_series[finetuning][time_res] = mf.compute_mae_from_dataset(ds_dataset, ds_inference, variable=var_list_metrics, keep_time=True)


            
# Convert to DataFrame and export
# Extract only the "6h" results from each finetuning method and convert to dict
for time_res in time_resolutions:
    mae_data = {}
    for finetuning in finetuning_types:
        mae_data[finetuning] = {var: mae[finetuning][time_res][var].values.item() for var in var_list_metrics}
    mae_df = pd.DataFrame(mae_data).T
    print("\nMAE Results:")
    print(mae_df.to_string())

    # Save to CSV
    mae_df.to_csv(f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/metrics/images/mae_results_{time_res}.csv')

#plot MAE time series for each variable
units = ["K", "K", "Pa", "kg/kg", "%"]
mae_time_res = {}
for time_res in time_resolutions:
    for finetuning in finetuning_types:
        mae_time_res[finetuning] = {time_res: mae_time_series[finetuning][time_res] for time_res in time_resolutions}
    
    for var, unit in zip(var_list_metrics, units):
        series_dict = {
            'No Finetuning': mae_time_res['No Finetuning'][time_res][var],
            'Default Finetuning': mae_time_res['Default Finetuning'][time_res][var],
            'PINNLoss Finetuning': mae_time_res['PINNLoss Finetuning'][time_res][var]
        }
        pf.plot_multiple_lines(
            series_dict,
            x=None,
            xlabel=f'Time Step {time_res}',
            ylabel=f'Mean Absolute Error ({unit})',
            title=f'MAE Time Series for {var}',
            savename=f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/metrics/images/mae_time_series_{var}_{time_res}.png",
            linestyle="-"
        )

#Compute MSSS for the different inferences and different variables
msss = defaultdict(dict)
msss_time_series= defaultdict(dict)

for finetuning in finetuning_types[1:]:  # Skip "No Finetuning"
    for time_res in time_resolutions:
        ds_inference = dict_inference[finetuning][time_res]
        ds_dataset = dict_dataset[time_res]
        
        msss[finetuning][time_res] = mf.compute_msss(ds_dataset,dict_inference["No Finetuning"][time_res], ds_inference, variable=var_list_metrics)
        msss_time_series[finetuning][time_res] = mf.compute_msss(ds_dataset, dict_inference["No Finetuning"][time_res], ds_inference, variable=var_list_metrics, keep_time=True)


# Convert to DataFrame and export
# Extract only the "6h" results from each finetuning method and convert to dict
for time_res in time_resolutions:
    msss_data = {}
    for finetuning in finetuning_types[1:]:  # Skip "No Finetuning"
        msss_data[finetuning] = {var: msss[finetuning][time_res][var].values.item() for var in var_list_metrics}
    msss_df = pd.DataFrame(msss_data).T
    print(f"\nMSSS Results for {time_res}:")
    print(msss_df.to_string())
    msss_df.to_csv(f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/metrics/images/msss_results_{time_res}.csv')

#Plot MSSS time series for each variable
msss_time_res = {}
for time_res in time_resolutions:
    for finetuning in finetuning_types[1:]:
        msss_time_res[finetuning] = {time_res: msss_time_series[finetuning][time_res] for time_res in time_resolutions}
    for var in var_list_metrics:
        series_dict = {
            'Default Finetuning': msss_time_res['Default Finetuning'][time_res][var],
            'PINNLoss Finetuning': msss_time_res['PINNLoss Finetuning'][time_res][var]
        }
        pf.plot_multiple_lines(
            series_dict,
            x=None,
            xlabel=f'Time Step ({time_res})',
            ylabel='MSSS Score',
            title=f'MSSS Time Series for {var}',
            savename=f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/metrics/images/msss_time_series_{var}_{time_res}.png",
            linestyle="-"
        )


print("MAE and MSSS computation completed.")

##################################################################################################
#Statistical analysis over random regions for regional MSSS boxplots
##################################################################################################
# Create sub-datasets with random lat/lon regions for regional MSSS analysis
n_regions = 100  # Number of random regions to sample
region_size = 5000  # Number of grid points per region
#reset ds_dataset t

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
#with time dimension
msss_time_PINN_regional = {var: [] for var in var_list_metrics}
msss_time_default_regional = {var: [] for var in var_list_metrics}

#Compute MAE for each region
mae_PINN_regional = {var: [] for var in var_list_metrics}
mae_default_regional = {var: [] for var in var_list_metrics}
mae_nofinetuning_regional = {var: [] for var in var_list_metrics}
#with time dimension
mae_time_PINN_regional = {var: [] for var in var_list_metrics}
mae_time_default_regional = {var: [] for var in var_list_metrics}
mae_time_nofinetuning_regional = {var: [] for var in var_list_metrics}

for i, indices in enumerate(region_indices):
    # Select sub-region for all datasets
    ds_dataset_region = dict_dataset["6h"].isel(values=indices)
    ds_inference_nofinetuning_region = dict_inference["No Finetuning"]["6h"].isel(values=indices)
    ds_inference_default_region = dict_inference["Default Finetuning"]["6h"].isel(values=indices)
    ds_inference_PINNLoss_region = dict_inference["PINNLoss Finetuning"]["6h"].isel(values=indices)
    
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
    #with time dimension
    msss_PINN_time = mf.compute_msss(ds_dataset_region, ds_inference_nofinetuning_region, ds_inference_PINNLoss_region, variable=var_list_metrics, squash=False, keep_time=True)
    msss_default_time = mf.compute_msss(ds_dataset_region, ds_inference_nofinetuning_region, ds_inference_default_region, variable=var_list_metrics, squash=False, keep_time=True)
    
    #Compute MAE for this region
    mae_PINN = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_PINNLoss_region, variable=var_list_metrics, squash=False)
    mae_default = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_default_region, variable=var_list_metrics, squash=False)
    mae_nofinetuning = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_nofinetuning_region, variable=var_list_metrics, squash=False)
    #with time dimension
    mae_time_PINN = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_PINNLoss_region, variable=var_list_metrics, squash=False, keep_time=True)
    mae_time_default = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_default_region, variable=var_list_metrics, squash=False, keep_time=True)
    mae_time_nofinetuning = mf.compute_mae_from_dataset(ds_dataset_region, ds_inference_nofinetuning_region, variable=var_list_metrics, squash=False, keep_time=True)
    
    # Organize by variable
    for var in var_list_metrics:
        msss_PINN_regional[var].append(msss_PINN[var])
        msss_default_regional[var].append(msss_default[var])
        
        msss_time_PINN_regional[var].append(msss_PINN_time[var])
        msss_time_default_regional[var].append(msss_default_time[var])
        
        mae_PINN_regional[var].append(mae_PINN[var])
        mae_default_regional[var].append(mae_default[var])
        mae_nofinetuning_regional[var].append(mae_nofinetuning[var])
        
        mae_time_PINN_regional[var].append(mae_time_PINN[var])
        mae_time_default_regional[var].append(mae_time_default[var])
        mae_time_nofinetuning_regional[var].append(mae_time_nofinetuning[var])

#MSSS Boxplots
# Prepare data for boxplot - one subplot per variable
boxplot_msss = {}
for var in var_list_metrics:
    boxplot_msss[var] = {
        'PINNLoss Finetuning': msss_PINN_regional[var],
        'Default Finetuning': msss_default_regional[var]
    }
# Prepare data for boxplot - one subplot per variable (last time step)
boxplot_msss_last_time = {}
for var in var_list_metrics:
    boxplot_msss_last_time[var] = {
        'PINNLoss Finetuning': [msss_time_PINN_regional[var][i][-1] for i in range(n_regions)],
        'Default Finetuning': [msss_time_default_regional[var][i][-1] for i in range(n_regions)]
    }

# Define colors
colors = {
    'PINNLoss Finetuning': '#FF6B6B',
    'Default Finetuning': '#4ECDC4'
}

# Create boxplots for MSSS
pf.plot_boxplots(
    data_dict=boxplot_msss,
    colors=colors,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/metrics/images/msss_regional_boxplot.png',
    ylabel='MSSS Score'
)

#Same for last timestep
pf.plot_boxplots(
    data_dict=boxplot_msss_last_time,
    colors=colors,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/metrics/images/msss_regional_boxplot_last_time.png',
    ylabel='MSSS Score (Last Time Step)'
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

# Prepare data for boxplot - one subplot per variable (last time step)
boxplot_mae_time_last = {}
for var in var_list_metrics:
    boxplot_mae_time_last[var] = {
        'PINNLoss Finetuning': [mae_time_PINN_regional[var][i][-1] for i in range(n_regions)],
        'Default Finetuning': [mae_time_default_regional[var][i][-1] for i in range(n_regions)],
        'No Finetuning': [mae_time_nofinetuning_regional[var][i][-1] for i in range(n_regions)]
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
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/metrics/images/mae_regional_boxplot.png',
    ylabel='Mean Absolute Error',
    sharey = False
)

#Same for last timestep
pf.plot_boxplots(
    data_dict=boxplot_mae_time_last,
    colors=colors_mae,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/metrics/images/mae_regional_boxplot_last_time.png',
    ylabel='Mean Absolute Error (Last Time Step)',
    sharey = False
)

print("Metrics computation and plotting completed.")