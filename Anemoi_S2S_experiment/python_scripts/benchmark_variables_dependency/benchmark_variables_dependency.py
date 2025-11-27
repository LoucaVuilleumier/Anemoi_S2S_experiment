import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import earthkit.data as ekd 
import cartopy.crs as ccrs
import cartopy.feature as cfeature

##Paths
inference_output_path = "/ec/res4/hpcperm/nld4584/output_inference/output_benchmark_july2022.nc"
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




#get the variables of interest for the dataset
def get_var_dataset(variable):
    idx_variable = var_names.index(variable)
    return ds_dataset_sliced["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})

t2m_dataset = get_var_dataset("2t")
dp2m_dataset = get_var_dataset("2d")
sp_dataset = get_var_dataset("sp")


#get the variables of interest for the inference ouput
t2m_inference = ds_inference["2t"]
dp2m_inference = ds_inference["2d"]
sp_inference = ds_inference["sp"]

# Convert temperature from Kelvin to Celsius
t2m_dataset_C = t2m_dataset - 273.15
dp2m_dataset_C = dp2m_dataset - 273.15
t2m_inference_C = t2m_inference - 273.15
dp2m_inference_C = dp2m_inference - 273.15

# Convert surface pressure from Pa to hPa
sp_dataset_hPa = sp_dataset / 100.0
sp_inference_hPa = sp_inference / 100.0

#compute rh and r

def compute_r_sur(t2m, dp2m, sp):       #mixing ratio
    e = np.where(                       #water vapor partial pressure, in hPa  
    t2m >= 0.0,
    6.107 * np.exp((17.368 * dp2m) / (dp2m + 238.83)),
    6.108 * np.exp((17.856 * dp2m) / (dp2m + 245.52)),
    )
    r_sur = 622.0 * (e / (sp - e))
    return r_sur

def compute_rh_sur(t2m, dp2m):       #relative humidity
    e = np.where(                   #water vapor partial pressure, in hPa  
    t2m >= 0.0,
    6.107 * np.exp((17.368 * dp2m) / (dp2m + 238.83)),
    6.108 * np.exp((17.856 * dp2m) / (dp2m + 245.52)),
    )
    e_sat =  np.where(          #saturation water vapor, in hPa
    t2m >= 0.0,
    6.107 * np.exp((17.368 * t2m) / (t2m + 238.83)),
    6.108 * np.exp((17.856 * t2m) / (t2m + 245.52)),
    )
    rh_sur = 100 * e/e_sat
    return rh_sur

r_sur_inference = compute_r_sur(t2m_inference_C, dp2m_inference_C, sp_inference_hPa)
r_sur_dataset = compute_r_sur(t2m_dataset_C, dp2m_dataset_C, sp_dataset_hPa)

rh_sur_inference = compute_rh_sur(t2m_inference_C,dp2m_inference_C)
rh_sur_dataset = compute_rh_sur(t2m_dataset_C, dp2m_dataset_C)


#compute residuals
r_res_sur =  (r_sur_dataset - r_sur_inference)
rh_res_sur = (rh_sur_dataset -rh_sur_inference)

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

def plot_surface_residues(r_res_sur, rh_res_sur, timestep):
    lons = ds_dataset["longitudes"].values.ravel()
    lats = ds_dataset["latitudes"].values.ravel()
    date = ds_dataset.dates.isel(time = (t0+ timestep)).values

    r_plot = r_res_sur.isel(time=timestep).values.ravel()
    rh_plot = rh_res_sur[timestep].ravel() 
    
    # remove duplicates
    #coords = np.column_stack((lons, lats))
    #_, idx_unique = np.unique(coords, axis=0, return_index=True)
    #lons, lats, r_plot, rh_plot = lons[idx_unique], lats[idx_unique], r_plot[idx_unique], rh_plot[idx_unique]

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
    ax.set_title(f"r surface residual at {str(date)[:19]}")
    cbar = plt.colorbar(cf, ax=ax, pad=0.05)
    cbar.set_label("(g/kg)", fontsize=12)
    plt.tight_layout()
    plt.savefig("/ec/res4/hpcperm/nld4584/python_scripts/benchmark_variables_dependency/images/r_residue", dpi =150)

    # --- RH residuals ---
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = ax.tricontourf(lons, lats, rh_plot, 40, transform=proj)
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title(f"RH surface residual at {str(date)[:19]}")
    cbar = plt.colorbar(cf, ax=ax, pad=0.05)
    cbar.set_label("%", fontsize=12)
    plt.tight_layout()
    plt.savefig("/ec/res4/hpcperm/nld4584/python_scripts/benchmark_variables_dependency/images/rh_residue", dpi =150)
    plt.close()
    
    
plot_surface_residues(r_res_sur, rh_res_sur, 120)



#### 3D fields


#retrieving 3D variable of interest from the dataset
n_levels = 13
n_cells = 40320

p_levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],)
p_reshaped = p_levels.reshape(1, n_levels, 1)

def get_3D_field_from_dataset(variable): #variable in form t_
    
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

#github test
test = "github test successful"
