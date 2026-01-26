import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LogNorm
from scipy.spatial import cKDTree
from scipy import stats
import earthkit.data as ekd 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import plotting_functions as pf
from utils import physics_functions as phfun
from utils import metrics_function as mf
from collections import defaultdict
from collections import defaultdict
import importlib
import torch
from anemoi.training.losses.mse import MSELoss
from anemoi.training.losses.PINNmse import PINNMSELoss
importlib.reload(pf)
importlib.reload(phfun)


##Paths
inference_output_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_inference_refmodel_6weeksjuly2022.nc"
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"

#inference output
ds_inference = xr.open_dataset(inference_output_path, engine="netcdf4")
n_steps = len(ds_inference.time)

#dataset 
ds_dataset = xr.open_zarr(dataset_path)
times = ds_dataset.dates.values
var_names = ds_dataset.attrs["variables"]
init_time = np.datetime64("2022-07-01T00:00")
t0 = np.where(times == init_time)[0][0]
ds_dataset_sliced = ds_dataset.isel(time = slice(t0, t0 + n_steps))

def get_var_dataset(dataset, variable):
    idx_variable = var_names.index(variable)
    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})


def hydrostatic_residual(z_pred, z_hydro):
    return z_pred - z_hydro

def T_hydrostatic_residual(t_pred, t_hydro):
    return t_pred[:, :, 1:-1] - t_hydro[:, :, 1:-1]

def get_antarctica_indices(dataset, lat_threshold=-80.0):
    latitudes = dataset['latitudes'].values
    antarctica_indices = np.where(latitudes <= lat_threshold)[0]
    return antarctica_indices

def mask_antarctica_data(data, antarctica_indices, mask_value=np.nan):
    data_masked = data.copy()
    data_masked[:, antarctica_indices] = mask_value
    return data_masked

p = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]) * 100  # Pa

variables = ["z", "t", "q"]
dict_var = {}
dict_var_era5 = {}
for var in variables:
    
    #var : array (..., nlev)     specific humidity [kg/kg]
    vars_p_level = [v for v in var_names if v.startswith(f"{var}_")]
    vars_p_level_sorted = sorted(vars_p_level, key=lambda x: int(x.split("_")[1]), reverse=True)
    print(vars_p_level_sorted)
    dict_var[var] = np.transpose(np.array([ds_inference[v].values for v in vars_p_level_sorted]), (1,2,0))
    dict_var_era5[var] = np.transpose(np.array([get_var_dataset(ds_dataset_sliced, v).values for v in vars_p_level_sorted]), (1,2,0))



T_hydro = phfun.compute_T_hydro(p, dict_var["z"], dict_var["q"])
T_hydro_era5 = phfun.compute_T_hydro(p, dict_var_era5["z"], dict_var_era5["q"])

T_residuals = T_hydrostatic_residual(dict_var["t"], T_hydro)
T_residuals_era5 = T_hydrostatic_residual(dict_var_era5["t"], T_hydro_era5)

# Get Antarctic indices for analysis
antarctica_indices = get_antarctica_indices(ds_dataset, lat_threshold=-80.0)
most_south_indices = get_antarctica_indices(ds_dataset, lat_threshold=-85.0)

print(f"\n=== ANTARCTIC GRID POINT ANALYSIS ===")
print(f"Total grid points: {T_residuals.shape[1]}")
print(f"Antarctic points (lat < -80°): {len(antarctica_indices)} indices")
print(f"Most southern points (lat < -85°): {len(most_south_indices)} indices")
print(f"Antarctic index range: {antarctica_indices.min()} to {antarctica_indices.max()}")
print(f"Most southern index range: {most_south_indices.min()} to {most_south_indices.max()}")

# Analyze residuals in Antarctic region
T_residuals_antarctica = T_residuals[:, antarctica_indices]
T_residuals_most_south = T_residuals[:, most_south_indices]

print(f"\n=== ANTARCTIC RESIDUAL STATISTICS ===")
print(f"Max residual (all points): {np.max(np.abs(T_residuals))} K")
print(f"Max residual (Antarctica, lat < -80°): {np.max(np.abs(T_residuals_antarctica))} K")
print(f"Max residual (Most southern, lat < -85°): {np.max(np.abs(T_residuals_most_south))} K")
print(f"Mean absolute residual (all points): {np.mean(np.abs(T_residuals))} K")
print(f"Mean absolute residual (Antarctica): {np.mean(np.abs(T_residuals_antarctica))} K")
print(f"Mean absolute residual (Most southern): {np.mean(np.abs(T_residuals_most_south))} K")

# Create masked version excluding Antarctic artifacts
T_residuals_masked = mask_antarctica_data(T_residuals, most_south_indices)

print(f"\n=== RESIDUAL STATISTICS AFTER MASKING MOST SOUTHERN POINTS ===")
print(f"Max residual (after masking): {np.nanmax(np.abs(T_residuals_masked))} K")
print(f"Mean absolute residual (after masking): {np.nanmean(np.abs(T_residuals_masked))} K")

#Sanity check at 
print(f"index of 500hPa: {np.where(p==50000)}")
print(f"q at 500hPa, first point: {dict_var['q'][0,0,np.where(p==50000)]}, typicall value 0.001 to 0.01 kg/kg")
print(f"Geopotential height * g at 500hPa, first point: {dict_var['z'][0,0,np.where(p==50000)]} m, typicall value ~55000 m/s^2")
print(f"T_hydro at 500hPa, first point: {T_hydro[0,0,np.where(p==50000)]} K")
print(f"T_pred at 500hPa, first point: {dict_var['t'][0,0,np.where(p==50000)]} K")
print(f"Max residual: {np.max(np.abs(T_residuals))} K")
print(f"Mean absolute residual: {np.mean(np.abs(T_residuals))} K")

#Plotting
# Mean absolute Temperature hydrostatic residuals over time
pf.plot_multiple_lines(
    {
        "Original (all points)": np.abs(T_residuals).mean(axis=(1,2)),
        "Masked (excluding Antarctic artifacts)": np.nanmean(np.abs(T_residuals_masked), axis=(1,2)),
        "Era5 (all points)": np.abs(T_residuals_era5).mean(axis=(1,2)),
    },
    xlabel="Time step",
    ylabel="Mean Absolute Temperature Residual (K)",
    title="Mean Absolute Temperature Residuals over Time",
    savename="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_var_dep_hydro_balance/images/mean_absolute_temperature_residuals_over_time.png",
    linestyle= "-",
)

#Mean absolute Temperature hydrostatic vertical distribution
pf.plot_multiple_lines(
    {
        "Original (all points)": np.abs(T_residuals).mean(axis=(0,1)),
        "Masked (excluding Antarctic artifacts)": np.nanmean(np.abs(T_residuals_masked), axis=(0,1)),
        "Era5 (all points)": np.abs(T_residuals_era5).mean(axis=(0,1)),
    },
    x = p[1:-1]/100,  # hPa
    xlabel="Mean Absolute Temperature Residual (K)",
    ylabel="Pressure Level (hPa)",
    title="Mean Absolute Temperature Residuals per Pressure Level",
    savename="/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_var_dep_hydro_balance/images/mean_absolute_temperature_residuals_per_pressure_level.png",
    linestyle= "-",
    transpose=True,
    flip_y=True
    
)



# Original surface field plots
pf.plot_surface_field(ds_dataset, T_residuals[:,:, np.where(p==50000)] ,120, "Temperature Residual (K) at 500hPa (Original)", "K", "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_var_dep_hydro_balance/images/temperature_residual_500hPa_original.png")
pf.plot_surface_field(ds_dataset, T_residuals_era5[:,:, np.where(p==50000)] ,120, "Temperature Residual (K) at 500hPa (ERA5)", "K", "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_var_dep_hydro_balance/images/temperature_residual_500hPa_era5.png")

pf.plot_surface_field(ds_dataset, T_residuals[:,:, np.where(p==85000)] ,120, "Temperature Residual (K) at 850hPa (Original)", "K", "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_var_dep_hydro_balance/images/temperature_residual_850hPa_original.png")

# Masked surface field plots
pf.plot_surface_field(ds_dataset, np.clip(T_residuals_masked[:,:, np.where(p==50000)], -100, 100),120, "Temperature Residual (K) at 500hPa (Masked)", "K", "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_var_dep_hydro_balance/images/temperature_residual_500hPa_masked.png")

pf.plot_surface_field(ds_dataset, np.clip(T_residuals_masked[:,:, np.where(p==85000)], -100, 100),120, "Temperature Residual (K) at 850hPa (Masked)", "K", "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/bench_var_dep_hydro_balance/images/temperature_residual_850hPa_masked.png")