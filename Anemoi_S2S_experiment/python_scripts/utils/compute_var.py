import xarray as xr
import numpy as np
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import plotting_functions as pf
from utils.physics_functions import compute_r_sur, compute_rh_sur
from utils import metrics_function as mf
from collections import defaultdict
import importlib
importlib.reload(pf)

##Path
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
#dataset 
ds_dataset = xr.open_zarr(dataset_path)
var_names = ds_dataset.attrs["variables"]

#get the variables of interest for the dataset
def get_var_dataset(dataset, variable):
    idx_variable = var_names.index(variable)
    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})

#needed to compute r and rh stats over the entire dataset
t2m_dataset = get_var_dataset(ds_dataset, "2t")
dp2m_dataset = get_var_dataset(ds_dataset, "2d")
sp_dataset = get_var_dataset(ds_dataset, "sp")

# Convert temperature from Kelvin to Celsius
t2m_dataset_C = t2m_dataset - 273.15
dp2m_dataset_C = dp2m_dataset - 273.15
sp_dataset_hPa = sp_dataset / 100.0  # Convert Pa to hPa

r_sur_whole_dataset = compute_r_sur(t2m_dataset_C, dp2m_dataset_C, sp_dataset_hPa)
rh_sur_whole_dataset = compute_rh_sur(t2m_dataset_C, dp2m_dataset_C)

var_r = r_sur_whole_dataset.var().item()
var_rh = rh_sur_whole_dataset.var().item()

print("Variance of r over the whole dataset: ", var_r)
print("Variance of rh over the whole dataset: ", var_rh)