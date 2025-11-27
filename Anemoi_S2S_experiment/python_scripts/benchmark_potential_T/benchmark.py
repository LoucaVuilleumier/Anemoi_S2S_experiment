# %%

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import earthkit.data as ekd 
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%
output_path = "/ec/res4/hpcperm/nld4584/output_inference/output_benchmark_inference.grib"
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"

# %%
ds_dataset = xr.open_zarr(dataset_path)
times = ds_dataset.dates.values
times

source = ekd.from_source("file", str(output_path)) 

# %%

t2m_model = source.sel(shortName="2t")

#print(t2m_model.values)

var_names = ds_dataset.attrs["variables"]
idx_2t = var_names.index("2t")
t2m_dataset= ds_dataset["data"].isel(variable=idx_2t)

init_time = np.datetime64("2022-09-01T00:00")
t0 = np.where(times == init_time)[0][0]
print("Index of init time:", t0)

n_steps = source.sel(shortName="2t").values.shape[0]
idx_2t = ds_dataset.attrs["variables"].index("2t")

t2m_truth_slice = ds_dataset["data"].isel(
    variable=idx_2t,
    time=slice(t0, t0 + n_steps)
)

#%%
mse_all = []
for i in range(n_steps):
    mse = (t2m_truth_slice[i][0].values - t2m_model.values[i])**2
    mse_all.append(mse)
    

mse_all = np.array(mse_all)  # shape: (n_steps, 40320)

# Compute global mean per step
mse_mean = mse_all.mean(axis=1)

plt.plot(mse_mean)
plt.xlabel("Forecast step")
plt.ylabel("MSE")
plt.title("Global Mean MSE over Forecast Lead Time, for 2t")
plt.savefig("/ec/res4/hpcperm/nld4584/python_scripts/benchmark_potential_T/images/mse_of_2t.png", dpi=150)
# %%
ds_model = ekd.from_source("file", output_path)
llv = ds_model[0].data() #longetude, latitude, values
#constant
p0 = 1000 # hPa
R_over_cp = 0.286
t_model = ds_model.sel(shortName="t").values

#%%

n_timesteps = 101
n_levels = 13
n_cells = 40320

# reshape: (time, level, cells)
T = t_model.reshape(n_timesteps, n_levels, n_cells)
p_levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],)

p0 = 1000.0  # reference hPa
R_over_cp = 0.286

# reshape pressure levels to broadcast: (1, level, 1)
p_reshaped = p_levels.reshape(1, n_levels, 1)

# potential temperature
theta = T * (p0 / p_reshaped)**R_over_cp


# %%
u_model = ds_model.sel(shortName="u").values
v_model = ds_model.sel(shortName="v").values
w_model = ds_model.sel(shortName="w").values

u = u_model.reshape(n_timesteps, n_levels, n_cells)
v = v_model.reshape(n_timesteps, n_levels, n_cells)
w = w_model.reshape(n_timesteps, n_levels, n_cells) 
# %%

lat = llv[0]
lon = llv[1]

# Convert lat/lon to Cartesian coordinates (meters)
R_earth = 6371000.0
lat_rad = np.deg2rad(lat)
lon_rad = np.deg2rad(lon)
x = R_earth * np.cos(lat_rad) * np.cos(lon_rad)
y = R_earth * np.cos(lat_rad) * np.sin(lon_rad)
z = R_earth * np.sin(lat_rad)
coords = np.stack([x, y, z], axis=1)

# Build KDTree
tree = cKDTree(coords)
k = 6
distances, indices = tree.query(coords, k=k+1)  # +1 includes self
distances = distances[:,1:]  # remove self
indices = indices[:,1:]
# %%

dtheta_dx = np.zeros_like(theta)
dtheta_dy = np.zeros_like(theta)

for i in range(n_cells):
    neighbor_idx = indices[i]
    dist = distances[i]  # shape (k,)

    # differences
    dtheta = theta[:, :, neighbor_idx] - theta[:, :, i][:, :, np.newaxis]  # (time, level, k)
    
    # simple finite difference: divide by distance and average over neighbors
    dtheta_dx[:, :, i] = np.mean(dtheta / dist, axis=2)
    dtheta_dy[:, :, i] = np.mean(dtheta / dist, axis=2)
# %%

dtheta_dz = np.zeros_like(theta)
dp = np.diff(p_levels)  # hPa
for k_level in range(n_levels-1):
    dtheta_dz[:, k_level, :] = (theta[:, k_level+1, :] - theta[:, k_level, :]) / dp[k_level]
dtheta_dz[:, -1, :] = dtheta_dz[:, -2, :]  # copy last level derivative


R_d = 287.0  # J/(kg*K)
g = 9.81     # m/s^2
p_levels_pa = p_levels * 100  # hPa → Pa
p_reshaped_pa = p_levels_pa.reshape(1, n_levels, 1)  # broadcast

w_meters = - w * R_d * T / (p_reshaped_pa * g)


dtheta_dt = np.zeros_like(theta)
dtheta_dt[1:, :, :] = theta[1:, :, :] - theta[:-1, :, :]
dtheta_dt[0, :, :] = dtheta_dt[1, :, :]


DthetaDt = dtheta_dt + u*dtheta_dx + v*dtheta_dy + w_meters*dtheta_dz

print("DthetaDt shape:", DthetaDt.shape)
# %%

# Average over cells and levels
mean_DthetaDt_time = DthetaDt.mean(axis=(1,2))  # shape: (time,)

plt.figure(figsize=(10,4))
plt.plot(mean_DthetaDt_time, marker='o')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Time step")
plt.ylabel("Average Dθ/Dt (K/s)")
plt.title("Net tendency of θ over time")
plt.grid(True)
plt.savefig("/ec/res4/hpcperm/nld4584/python_scripts/benchmark_potential_T/images/DThetaDt.png", dpi=150)
# %%


dataset_slice = ds_dataset["data"].isel(
    time=slice(t0, t0 + n_steps)
)
timestep_slice = dataset_slice.time.values
## Dataset

vars_t = [v for v in var_names if v.startswith("t_")]
vars_u = [v for v in var_names if v.startswith("u_")]
vars_v = [v for v in var_names if v.startswith("v_")] 
vars_w = [v for v in var_names if v.startswith("w_")]

vars_t_sorted = sorted(vars_t, key=lambda x: int(x.split("_")[1]))
vars_u_sorted = sorted(vars_u, key=lambda x: int(x.split("_")[1]))
vars_v_sorted = sorted(vars_v, key=lambda x: int(x.split("_")[1]))
vars_w_sorted = sorted(vars_w, key=lambda x: int(x.split("_")[1]))

t_indices_sorted = [var_names.index(v) for v in vars_t_sorted]
u_indices_sorted = [var_names.index(v) for v in vars_u_sorted]
v_indices_sorted = [var_names.index(v) for v in vars_v_sorted]
w_indices_sorted = [var_names.index(v) for v in vars_w_sorted]

t_dataset = dataset_slice.isel(variable=t_indices_sorted)
u_dataset = dataset_slice.isel(variable=u_indices_sorted)
v_dataset = dataset_slice.isel(variable=v_indices_sorted)
w_dataset = dataset_slice.isel(variable=w_indices_sorted)

t_dataset = t_dataset.squeeze(dim="ensemble")
u_dataset = u_dataset.squeeze(dim="ensemble")
v_dataset = v_dataset.squeeze(dim="ensemble")
w_dataset = w_dataset.squeeze(dim="ensemble")


theta_data = t_dataset * (p0 / p_reshaped)**R_over_cp
theta_data = theta_data.values
dtheta_dx_data = np.zeros_like(theta_data)
dtheta_dy_data = np.zeros_like(theta_data)

for i in range(n_cells):
    neighbor_idx = indices[i]
    dist = distances[i]  # shape (k,)

    # differences
    dtheta = theta_data[:, :, neighbor_idx] - theta_data[:, :, i][:, :, np.newaxis]  # (time, level, k)
    
    # simple finite difference: divide by distance and average over neighbors
    dtheta_dx_data[:, :, i] = np.mean(dtheta / dist, axis=2)
    dtheta_dy_data[:, :, i] = np.mean(dtheta / dist, axis=2)
    
dtheta_dz_data = np.zeros_like(theta_data)
dp = np.diff(p_levels)  # hPa
for k_level in range(n_levels-1):
    dtheta_dz_data[:, k_level, :] = (theta_data[:, k_level+1, :] - theta_data[:, k_level, :]) / dp[k_level]
dtheta_dz_data[:, -1, :] = dtheta_dz_data[:, -2, :]  # copy last level derivative

w_meters_data = - w_dataset * R_d * T / (p_reshaped_pa * g)

dtheta_dt_data = np.zeros_like(theta_data)
dtheta_dt_data[1:, :, :] = theta_data[1:, :, :] - theta_data[:-1, :, :]
dtheta_dt_data[0, :, :] = dtheta_dt_data[1, :, :]

DthetaDt_data = dtheta_dt_data + u*dtheta_dx_data + v*dtheta_dy_data + w_meters_data*dtheta_dz_data
mean_DthetaDt_time_data = DthetaDt_data.mean(axis=(1,2))  # shape: (time,)


plt.figure(figsize=(10,4))
plt.plot(mean_DthetaDt_time_data, marker='o')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Time step")
plt.ylabel("Average Dθ/Dt (K/s)")
plt.title("Net tendency of θ over time, for Era5 dataset")
plt.grid(True)
plt.savefig("/ec/res4/hpcperm/nld4584/python_scripts/benchmark_potential_T/images/DThetaDt_dataset.png", dpi=150)

##Comparison between the 2

diff_mean_DthetaDt = mean_DthetaDt_time_data - mean_DthetaDt_time

plt.figure(figsize=(10,4))
plt.plot(diff_mean_DthetaDt, marker='o')
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Time step")
plt.ylabel("Residual (K/s)")
plt.title(r"Residual of $\frac{D\theta}{Dt} - \widehat{\frac{D\theta}{Dt}}$")
plt.grid(True)
plt.savefig("/ec/res4/hpcperm/nld4584/python_scripts/benchmark_potential_T/images/DThetaDt_diff.png", dpi=150)




#Map of the difference

def plot_map_diff(DthetaDt_data, DthetaDt, p_level, timestep):
    diff_global = DthetaDt_data - DthetaDt
    idx_p_level = np.where(p_levels == p_level)[0]
    if len(idx_p_level) == 0:
        print("Pressure level not found, valid p levels are 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000")
        

    date = ds_dataset.dates.isel(time = (t0+ timestep)).values

    idx_p_level = idx_p_level[0]

    # Extract field to plot
    field = diff_global[timestep, idx_p_level, :]
    
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.set_global()
    
    # tricontourf for unstructured grid
    cs = ax.tricontourf(lon, lat, field, 40, transform=ccrs.PlateCarree(), alpha = 0.5)
    plt.title(f"Residual Dθ/Dt difference at {p_level} hPa\n{str(date)[:19]}")
    cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05)
    cbar.set_label("Residual (K/s)")
    plt.savefig("/ec/res4/hpcperm/nld4584/python_scripts/benchmark_potential_T/images/MapResidual.png", dpi=150)



plot_map_diff(
    DthetaDt_data, 
    DthetaDt,
    p_level=500,
    timestep=100
)


#diff_global = DthetaDt_data - DthetaDt
#field = diff_global[100,0, :]
#date = ds_dataset.dates.isel(time = (t0+ 100)).values

#fig = plt.figure(figsize=(10,6))
#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.add_feature(cfeature.COASTLINE)
#ax.add_feature(cfeature.BORDERS, linestyle=':')
#ax.add_feature(cfeature.LAND, edgecolor='black')
#ax.set_global()
#cs = ax.tricontourf(lon, lat, field, 40, transform=ccrs.PlateCarree(), alpha = 0.5)
#plt.title(f"Residual Dθ/Dt difference at 50 hPa\n{str(date)[:19]}")
#cbar = plt.colorbar(cs, orientation='horizontal', pad=0.05)
#cbar.set_label("Residual (K/s)")
#plt.savefig("/ec/res4/hpcperm/nld4584/python_scripts/benchmark_potential_T/images/MapResidual.png", dpi=10)
#plt.close()