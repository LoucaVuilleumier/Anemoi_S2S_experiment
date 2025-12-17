import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import earthkit.data as ekd 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import plotting_functions as pf
from utils.physics_functions import compute_r_sur, compute_rh_sur
from utils import metrics_function as mf
from collections import defaultdict
from collections import defaultdict
import importlib
import torch
from anemoi.training.losses.mse import MSELoss
from anemoi.training.losses.PINNmse import PINNMSELoss
importlib.reload(pf)


##Paths
inference_output_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_benchmark_july2022.nc"
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"

#Variables of interest
var_list = ['2t', '2d', 'sp'] 

#inference output
ds_inference = xr.open_dataset(inference_output_path, engine="netcdf4")
n_steps = len(ds_inference.time)

#Daily average (4 time steps = 24h at 6h resolution)
ds_inference_daily_avg = ds_inference.coarsen(time=4, boundary='trim').mean()

#Weekly average (28 time steps = 7 days at 6h resolution)
ds_inference_weekly_avg = ds_inference.coarsen(time=28, boundary='trim').mean()

#Create new dataset with only variables of interest
ds_inference_6h = ds_inference[var_list]
ds_inference_daily_avg = ds_inference_daily_avg[var_list]
ds_inference_weekly_avg = ds_inference_weekly_avg[var_list]

#dataset 
ds_dataset = xr.open_zarr(dataset_path)
times = ds_dataset.dates.values
var_names = ds_dataset.attrs["variables"]
init_time = np.datetime64("2022-07-01T00:00")
t0 = np.where(times == init_time)[0][0]
ds_dataset_sliced = ds_dataset.isel(time = slice(t0, t0 + n_steps))

#Daily, weekly, and monthly averages for dataset
ds_dataset_daily_avg = ds_dataset_sliced.coarsen(time=4, boundary='trim').mean()
ds_dataset_weekly_avg = ds_dataset_sliced.coarsen(time=28, boundary='trim').mean()

#get the variables of interest for the dataset
def get_var_dataset(dataset, variable):
    idx_variable = var_names.index(variable)
    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})


ds_dataset_6h = xr.Dataset({var: get_var_dataset(ds_dataset_sliced, var) for var in var_list})
ds_dataset_daily_avg = xr.Dataset({var: get_var_dataset(ds_dataset_daily_avg, var) for var in var_list})
ds_dataset_weekly_avg = xr.Dataset({var: get_var_dataset(ds_dataset_weekly_avg, var) for var in var_list})

#t2m_dataset = ds_dataset_6h["2t"]
#dp2m_dataset = ds_dataset_6h["2d"]
#sp_dataset = ds_dataset_6h["sp"]


#get the variables of interest for the inference ouput
#t2m_inference = ds_inference["2t"]
#dp2m_inference = ds_inference["2d"]
#sp_inference = ds_inference["sp"]

# Convert temperature from Kelvin to Celsius
#t2m_dataset_C = t2m_dataset - 273.15
#dp2m_dataset_C = dp2m_dataset - 273.15
#t2m_inference_C = t2m_inference - 273.15
#dp2m_inference_C = dp2m_inference - 273.15

#Conversion to use physics equations
list_datasets = [ds_dataset_6h, ds_dataset_daily_avg, ds_dataset_weekly_avg]
for dataset in list_datasets:
    dataset["2t"].values = dataset["2t"].values - 273.15
    dataset["2d"].values = dataset["2d"].values - 273.15
    dataset["sp"].values = dataset["sp"].values / 100.0  # Convert Pa to hPa

list_inferences = [ds_inference_6h, ds_inference_daily_avg, ds_inference_weekly_avg]
for inference in list_inferences:
    inference["2t"].values = inference["2t"].values - 273.15
    inference["2d"].values = inference["2d"].values - 273.15
    inference["sp"].values = inference["sp"].values / 100.0  # Convert Pa to hPa

# Convert surface pressure from Pa to hPa
#sp_dataset_hPa = sp_dataset / 100.0
#sp_inference_hPa = sp_inference / 100.0


#plot all surface fields
pf.plot_surface_field(ds_dataset, ds_dataset_6h["2t"], 1, "2m Temperature Dataset (°C)", "°C", "t2m_dataset.png", colormap='coolwarm')
pf.plot_surface_field(ds_dataset, ds_inference_6h["2t"], 1, "2m Temperature Inference  (°C)", "°C", "t2m_inference.png", colormap='coolwarm')
pf.plot_surface_field(ds_dataset, ds_dataset_6h["2d"], 1, "2m Dew Point Temperature Dataset (°C)", "°C", "dp2m_dataset.png", colormap='coolwarm')
pf.plot_surface_field(ds_dataset, ds_inference_6h["2d"], 1, "2m Dew Point Temperature Inference (°C)", "°C", "dp2m_inference.png", colormap='coolwarm')
pf.plot_surface_field(ds_dataset, ds_dataset_6h["sp"], 1, "Surface Pressure Dataset (hPa)", "hPa", "sp_dataset.png", colormap='winter')
pf.plot_surface_field(ds_dataset, ds_inference_6h["sp"], 1, "Surface Pressure Inference (hPa)", "hPa", "sp_inference.png", colormap='winter')

#compute rh and r
timesteps_list = ["6h", "daily_avg", "weekly_avg"]
rh_sur_datasets = {}
r_sur_datasets = {}
for dataset, timestep in zip(list_datasets, timesteps_list):
    rh_sur_datasets[timestep] = compute_rh_sur(dataset["2t"], dataset["2d"])
    r_sur_datasets[timestep] = compute_r_sur(dataset["2t"], dataset["2d"], dataset["sp"])


rh_sur_inferences = {}
r_sur_inferences = {}
for inference, timestep in zip(list_inferences, timesteps_list):
    rh_sur_inferences[timestep] = compute_rh_sur(inference["2t"], inference["2d"], clip_for_plot=False)
    r_sur_inferences[timestep] = compute_r_sur(inference["2t"], inference["2d"], inference["sp"])


#plot rh and r

pf.plot_surface_field(ds_dataset, rh_sur_datasets["6h"], 1, "RH Dataset", "%", "rh_dataset.png")
pf.plot_surface_field(ds_dataset, rh_sur_inferences["6h"], 1, "RH Inference", "%", "rh_inference.png")

pf.plot_surface_field(ds_dataset, r_sur_datasets["6h"], 1, "r Dataset", "g/kg", "r_dataset.png")
pf.plot_surface_field(ds_dataset, r_sur_inferences["6h"], 1, "r Inference", "g/kg", "r_inference.png")
pf.plot_surface_field(ds_dataset, rh_sur_inferences["daily_avg"], 1, "RH Dataset", "%", "rh_dataset_daily_avg.png")


#compute residuals
r_res_sur = {}
rh_res_sur = {}

for timestep in timesteps_list:
    r_res_sur[timestep] = r_sur_datasets[timestep] - r_sur_inferences[timestep]
    rh_res_sur[timestep] = rh_sur_datasets[timestep] - rh_sur_inferences[timestep]


#t2m_res = (t2m_dataset_C - t2m_inference_C)
#dp2m_res = (dp2m_dataset_C - dp2m_inference_C)
#sp_res = (sp_dataset_hPa - sp_inference_hPa)

#sanity checks
def sanity_check(name, arr):
    print(f"--- {name} ---")
    print("shape:", arr.shape)
    print("min:", np.nanmin(arr))
    print("max:", np.nanmax(arr))
    print("mean:", np.nanmean(arr))
    print("median:", np.nanmedian(arr))
    print("std:", np.nanstd(arr))
    print("number of NaNs:", np.isnan(arr).sum())
    print()
    
sanity_check("t2m_dataset_C", t2m_dataset_C.values)
sanity_check("t2m_inference_C", t2m_inference_C.values)

sanity_check("dp2m_datase_C", dp2m_dataset_C.values)
sanity_check("dp2m_inference_C", dp2m_inference_C.values)

sanity_check("sp_dataset_hPa", sp_dataset_hPa.values)
sanity_check("sp_inference_hPa", sp_inference_hPa.values)

sanity_check("r_sur_dataset", r_sur_dataset.values)
sanity_check("r_sur_inference", r_sur_inference.values)

sanity_check("rh_sur_dataset", rh_sur_dataset)
sanity_check("rh_sur_inference", rh_sur_inference)

sanity_check("r_res_sur", r_res_sur.values)
sanity_check("rh_res_sur", rh_res_sur)

#Plot surface residues
last_indices =[120,29,3]  #last time step for 6h, daily avg, weekly avg
for timestep, last in zip(timesteps_list, last_indices):
    
    plot_surface_field(ds_dataset, r_res_sur[timestep], last, f"r surface residual, {timestep}", "g/kg", f"r_residue_last_time_{timestep}.png")
    plot_surface_field(ds_dataset, rh_res_sur[timestep], last, f"RH surface residual, {timestep}", "%", f"rh_residue_last_time_{timestep}.png")
    plot_surface_field(ds_dataset, r_res_sur[timestep], 1, f"r surface residual, {timestep}", "g/kg", f"r_residue_first_time_{timestep}.png")
    plot_surface_field(ds_dataset, rh_res_sur[timestep], 1, f"RH surface residual, {timestep}", "%", f"rh_residue_first_time_{timestep}.png")

#plot_surface_field(ds_dataset, t2m_res, 120, "2m Temperature residual", "°C", "t2m_residue_last_time.png", colormap='coolwarm')
#plot_surface_field(ds_dataset, dp2m_res, 120, "2m Dew Point Temperature residual", "°C", "dp2m_residue_last_time.png", colormap='coolwarm')
#plot_surface_field(ds_dataset, sp_res, 120, "Surface Pressure residual", "hPa", "sp_residue_last_time.png", colormap='winter')

#plot_surface_field(ds_dataset, t2m_res, 1, "2m Temperature residual", "°C", "t2m_residue_first_time.png", colormap='coolwarm')
#plot_surface_field(ds_dataset, dp2m_res, 1, "2m Dew Point Temperature residual", "°C", "dp2m_residue_first_time.png", colormap='coolwarm')
#plot_surface_field(ds_dataset, sp_res, 1, "Surface Pressure residual", "hPa", "sp_residue_first_time.png", colormap='winter')

##Compute relative MSE
relative_mse = defaultdict(dict)

for var in var_list:
    for timestep, dataset, inference in zip(timesteps_list, list_datasets, list_inferences):
        relative_mse[var][timestep] = mf.compute_relative_mse(
            dataset[var],
            inference[var]
        )

squashed_mse_6h = 1/3*(relative_mse['2t']['6h'] + relative_mse['2d']['6h'] + relative_mse['sp']['6h'] )

#add the rh_sur variable
for timestep in timesteps_list:
    relative_mse["rh_sur"][timestep] = mf.compute_relative_mse(
        rh_sur_datasets[timestep],
        rh_sur_inferences[timestep]
    )
    
#add the r_sur variable
for timestep in timesteps_list:
    relative_mse["r_sur"][timestep] = mf.compute_relative_mse(
        r_sur_datasets[timestep],
        r_sur_inferences[timestep]
    )
    
squashed_mse_6h_physics = 1/2*(relative_mse['rh_sur']['6h'] + relative_mse['r_sur']['6h'] )

#plot relative MSE time series
for timestep in timesteps_list:
    pf.plot_multiple_lines(
        {
            "Relative MSE t2m": relative_mse["2t"][timestep],
            "Relative MSE dp2m": relative_mse["2d"][timestep],
            "Relative MSE sp": relative_mse["sp"][timestep],
            "Relative MSE r_sur": relative_mse["r_sur"][timestep],
            "Relative MSE rh_sur": relative_mse["rh_sur"][timestep]
        },
        linestyle={
        "Relative MSE t2m": "-",
        "Relative MSE dp2m": "--",
        "Relative MSE sp": "-.",
        "Relative MSE r_sur": ":",
        "Relative MSE rh_sur": "-"
        },
        xlabel="Time step",
        ylabel="Relative MSE",
        title=f"Relative MSE time series, {timestep}",
        savename=f"relative_mse_time_series_{timestep}.png"
    )
##################################################################################################################################################################
#Reload the dataset to reset Kevlin and Pa to their original units
ds_dataset = xr.open_zarr(dataset_path)
times = ds_dataset.dates.values
var_names = ds_dataset.attrs["variables"]
init_time = np.datetime64("2022-07-01T00:00")
t0 = np.where(times == init_time)[0][0]
ds_dataset_sliced = ds_dataset.isel(time = slice(t0, t0 + n_steps))

ds_inference = xr.open_dataset(inference_output_path, engine="netcdf4")
ds_inference_6h = ds_inference[var_list]

#prepare the pred and targe to use anemoi.training.losses.mse.MSELoss 
var_list = ['2t', '2d', 'sp'] 
pred_arrays = [ds_inference_6h[v].values for v in var_list]
target_arrays = [get_var_dataset(ds_dataset_sliced,v).values for v in var_list]


# Stack along new variable dimension
pred_stacked = np.stack(pred_arrays, axis=-1)      # (time, values, n_vars)
target_stacked = np.stack(target_arrays, axis=-1)  # (time, values, n_vars)

pred_stacked = pred_stacked[:, np.newaxis, :, :]    # (time, 1, values, n_vars) 
target_stacked = target_stacked[:, np.newaxis, :, :] # (time, 1, values, n_vars)

pred_tensor = torch.from_numpy(pred_stacked).float()
target_tensor = torch.from_numpy(target_stacked).float()

#Instantiate the loss
loss_fn = MSELoss(ignore_nans=False)

#create scaler for grid weights (uniform weights here)
n_grid = pred_tensor.shape[2]
grid_weights = torch.ones(n_grid) / n_grid #Uniform weights
loss_fn.add_scaler(2, grid_weights, name="grid_weights")  

# Variable normalization (inverse variance scaling)
# Compute std for each variable across all data
var_variances = []
for i in range(len(var_list)):
    variance = target_stacked[:, :, :, i].var()  # Variance over values (=spatial dimension)
    var_variances.append(1/variance)


var_scales = torch.tensor(var_variances, dtype=torch.float32)
loss_fn.add_scaler(3, var_scales, name="variable_scales") 

#Compute loss
mse_value = loss_fn(pred_tensor, target_tensor)
print(f"MSE loss: {mse_value.item()}")

#Compute The loss in a loop over time steps to get time series
mse_time_series = []
n_time_steps = pred_tensor.shape[0]
for t in range(n_time_steps):
    pred_t = pred_tensor[t:t+1, :, :, :]   # (1, 1, values, n_vars)
    target_t = target_tensor[t:t+1, :, :, :] # (1, 1, values, n_vars)
    mse_t = loss_fn(pred_t, target_t)
    mse_time_series.append(mse_t.item())


mse_time_series




#############################
#PINN Loss
#Instantiate the PINN loss
pinn_loss_pure_data = PINNMSELoss(physics_weight=1.0, alpha=0.0)  # Pure data loss
pinn_loss_pure_physics = PINNMSELoss(physics_weight=1.0, alpha=1.0)  # Pure physics loss

#Create scalers as before
pinn_loss_pure_data.add_scaler(2, grid_weights, name="grid_weights")
pinn_loss_pure_data.add_scaler(3, var_scales, name="variable_scales")

pinn_loss_pure_physics.add_scaler(2, grid_weights, name="grid_weights")
#pinn_loss_pure_physics.add_scaler(3, var_scales, name="variable_scales")

# Set the variable indices manually for testing
# var_list = ['2t', '2d', 'sp']
pinn_loss_pure_data.set_indices_manually(idx_2t=0, idx_2d=1, idx_sp=2)
pinn_loss_pure_physics.set_indices_manually(idx_2t=0, idx_2d=1, idx_sp=2)

#Compute PINN loss time series
pinn_mse_pure_data_time_series = []
pinn_mse_pure_physics_time_series = []
for t in range(n_time_steps):
    pred_t = pred_tensor[t:t+1, :, :, :]   # (1, 1, values, n_vars)
    target_t = target_tensor[t:t+1, :, :, :] # (1, 1, values, n_vars)
    
    pinn_mse_data_t = pinn_loss_pure_data(pred_t, target_t)
    pinn_mse_physics_t = pinn_loss_pure_physics(pred_t, target_t)
    
    pinn_mse_pure_data_time_series.append(pinn_mse_data_t.item())
    pinn_mse_pure_physics_time_series.append(pinn_mse_physics_t.item())
### Plotting
pf.plot_multiple_lines(
    {
        "Hand-made, data Loss": squashed_mse_6h,
        "Anemoi MSE": mse_time_series,
        "Hand-made, Physics Loss": squashed_mse_6h_physics,
        "PINN Loss (pure data)": pinn_mse_pure_data_time_series,
        "PINN Loss (pure physics)": pinn_mse_pure_physics_time_series,

    },
        linestyle={
        "Hand-made, data Loss": "-",
        "Anemoi MSE": "--",
        "Hand-made, Physics Loss": "--",
        "PINN Loss (pure data)": "-.",
        "PINN Loss (pure physics)": ":"
        },    
    xlabel="Time step",
    ylabel="Squashed Relative MSE",
    title=f"Relative MSE time series comparison, 6h",
    savename=f"comparison_relative_mse_time_series_6h.png"
)
##################################################################################################################
#### 3D fields


#retrieving 3D variable of interest from the dataset
n_levels = 13
n_cells = 40320

p_levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],)
p_reshaped = p_levels.reshape(1, n_levels, 1)

def get_3D_field_from_dataset(variable): 
    
        vars = [v for v in var_names if v.startswith(variable)]
        vars_sorted = sorted(vars, key=lambda x: int(x.split("_")[1]))
        indices_sorted = [var_names.index(v) for v in vars_sorted]
        print(indices_sorted)
        return ds_dataset_sliced["data"].isel(variable = indices_sorted).squeeze(dim="ensemble").rename({"cell": "values"}), vars_sorted
    
t_dataset, t_vars_sorted = get_3D_field_from_dataset("t_")
t_dataset_C = t_dataset - 273.15
q_dataset, q_vars_sorted = get_3D_field_from_dataset("q_")


#retrieving 3D variable of interest from the inference 
t_inference = ds_inference[t_vars_sorted].to_array(dim="variable")
t_inference_C = t_inference - 273.15
t_inference_C = t_inference_C.transpose("time", "variable", "values")

q_inference = ds_inference[q_vars_sorted].to_array(dim="variable")
q_inference = q_inference.transpose("time", "variable", "values")
q_inference_clipped = np.clip(q_inference, 0, None) #remove unphysicall value under zero



#### rh in 3D
def compute_rh_3D(t, q, p_levels):
    nt, nl, nc = t.shape
    rh = np.empty((nt, nl, nc))

    for i, p in enumerate(p_levels):
        #  vapor pressure
        e = q[:, i, :] * p / 0.622

        # saturation vapor pressure over water (Tetens)
        e_s = 6.112 * np.exp(17.67 * t[:, i, :] / (t[:, i, :] + 243.5))

        rh[:, i, :] = 100 * e / e_s

    return rh


#Compute for dataset
rh_3d_dataset   = compute_rh_3D(t_dataset_C,   q_dataset,   p_levels)



#Compute for inference
rh_3d_inference = compute_rh_3D(t_inference_C, q_inference_clipped, p_levels)


## check 
sanity_check("rh_3d_dataset ",rh_3d_dataset[:, 12, :])
sanity_check("rh_3d_inference ",rh_3d_inference[:, 12, :])
sanity_check("rh_dataset surface", rh_sur_dataset)


#water vapor mixing ratio
def compute_r_from_q(q, to_g_per_kg=True):
    """
    Compute water vapor mixing ratio r from specific humidity q.

    Parameters
    ----------
    q : np.ndarray or xarray.DataArray
        Specific humidity (kg/kg), shape (time, level, cell) or (time, cell)
    to_g_per_kg : bool
        If True, convert r to g/kg

    Returns
    -------
    r : np.ndarray
        Water vapor mixing ratio (kg/kg or g/kg), same shape as q
    """
    r = q / (1.0 - q)
    if to_g_per_kg:
        r = r * 1000.0
    return r

r_3d_dataset = compute_r_from_q(q_dataset)  
r_3d_inference = compute_r_from_q(q_inference_clipped)

sanity_check("r_3d_dataset ",r_3d_dataset[:, 12, :])
sanity_check("r_3d_inference ",r_3d_inference[:, 12, :])
sanity_check("rh_dataset surface", r_sur_dataset) 

#compute 3D residuals
r_res_3D =  ((r_3d_dataset - r_3d_inference)).values
rh_res_3D = (rh_3d_dataset -rh_3d_inference)


#plot
def plot_3D_residues(r_res_3D, rh_res_3D, timestep, p_level, savename):
    lons = ds_dataset["longitudes"].values.ravel()
    lats = ds_dataset["latitudes"].values.ravel()
    date = ds_dataset.dates.isel(time = (t0+ timestep)).values
    p = p_levels[p_level]
    
    r_plot = r_res_3D[timestep, p_level, :]
    rh_plot = rh_res_3D[timestep, p_level, :]
    

    # remove NaNs/infs
    mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(r_plot) & np.isfinite(rh_plot)
    lons, lats, r_plot, rh_plot = lons[mask], lats[mask], r_plot[mask], rh_plot[mask]

    proj = ccrs.PlateCarree()

    
    # --- r residuals ---
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    cf = ax.tricontourf(lons, lats, r_plot, 40, transform=proj)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title(f"r residual at {p} hPa & {str(date)[:19]}")
    cbar = plt.colorbar(cf, ax=ax, pad=0.05)
    cbar.set_label("g/kg")
    plt.tight_layout()
    plt.savefig(f"/ec/res4/hpcperm/nld4584/python_scripts/benchmark_variables_dependency/images/r_{savename}.png", dpi =150)
    plt.close()

    # --- RH residuals ---
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = ax.tricontourf(lons, lats, rh_plot, 40, transform=proj)
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title(f"RH  residual at {p} hPa & {str(date)[:19]}")
    cbar = plt.colorbar(cf, ax=ax, pad=0.05)
    cbar.set_label("%")
    plt.tight_layout()
    plt.savefig(f"/ec/res4/hpcperm/nld4584/python_scripts/benchmark_variables_dependency/images/rh_{savename}.png", dpi =150)
    plt.close()

plot_3D_residues(r_res_3D, rh_res_3D, 120, 5, "residue_3D")



#clip rh inference to plausible value for plotting
rh_3d_inference_clip = np.clip(rh_3d_inference, 0, 120)
rh_res_3D_plot = (rh_3d_dataset -rh_3d_inference_clip)
plot_3D_residues(r_res_3D, rh_res_3D_plot, 120, 5, "residue__clipped_3D")


#RMSE plot

rmse_r_sur = np.sqrt((r_res_sur**2).mean(dim="values"))
rmse_rh_sur = np.sqrt(np.mean(rh_res_sur**2, axis=1))



rmse_r_3D = np.sqrt(np.mean(r_res_3D**2, axis = (1,2)))
rmse_rh_3D = np.sqrt(np.mean(rh_res_3D **2, axis = (1,2)))





def plot_rmse_time(rmse_r, rmse_rh, savename, times=None, title="RMSE over time"):

    if times is None:
        times = range(len(rmse_r))
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Left axis for RMSE of r
    color_r = "tab:blue"
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("RMSE r [g/kg]", color=color_r)
    ax1.plot(times, rmse_r, color=color_r, label="RMSE r")
    ax1.tick_params(axis="y", labelcolor=color_r)
    
    # Right axis for RMSE of RH
    ax2 = ax1.twinx()
    color_rh = "tab:red"
    ax2.set_ylabel("RMSE RH [%]", color=color_rh)
    ax2.plot(times, rmse_rh, color=color_rh, label="RMSE RH")
    ax2.tick_params(axis="y", labelcolor=color_rh)
    
    # Title
    plt.title(title)
    
    # Optional: combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f"/ec/res4/hpcperm/nld4584/python_scripts/benchmark_variables_dependency/images/{savename}.png", dpi =150)
    plt.close()
    
plot_rmse_time(rmse_r_sur, rmse_rh_sur, "rmse_sur", times=np.arange(121), title="Surface RMSE")
plot_rmse_time(rmse_r_3D, rmse_rh_3D, "rmse_3D", times = np.arange(121), title = "Global RMSE")

