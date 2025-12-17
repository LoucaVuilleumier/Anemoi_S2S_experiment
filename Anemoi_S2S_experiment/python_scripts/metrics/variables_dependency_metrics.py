import xarray as xr
import numpy as np
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

inference_output_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_inference/output_benchmark_july2022.nc"
dataset_path = "/home/mlx/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"

#open datasets
ds_inference = xr.open_dataset(inference_output_path)
ds_dataset = xr.open_zarr(dataset_path)

mae_2t = mf.compute_mae_from_dataset(ds_dataset, ds_inference, variable="2t")

msss = mf.compute_msss(ds_dataset, ds_inference, ds_inference, variable=["2t", "10u"], squash=False)

np.random.seed(0)

data = {
    "T": {
        "Unconstrained": np.random.normal(1.35, 0.05, 50),
        "Architecture constrained": np.random.normal(1.32, 0.04, 50),
        "Loss constrained": np.random.normal(1.31, 0.03, 50),
        "Offline constrained": np.random.normal(1.38, 0.06, 50),
    },
    "Td": {
        "Unconstrained": np.random.normal(1.52, 0.06, 50),
        "Architecture constrained": np.random.normal(1.50, 0.05, 50),
        "Loss constrained": np.random.normal(1.47, 0.04, 50),
        "Offline constrained": np.random.normal(1.55, 0.07, 50),
    },
    "P": {
        "Unconstrained": np.random.normal(0.53, 0.02, 50),
        "Architecture constrained": np.random.normal(0.52, 0.02, 50),
        "Loss constrained": np.random.normal(0.51, 0.02, 50),
        "Offline constrained": np.random.normal(0.54, 0.03, 50),
    },
}

colors = {
    "Unconstrained": "#4C9BE8",
    "Architecture constrained": "#C03A6B",
    "Loss constrained": "#E0B11C",
    "Offline constrained": "#1B8A6B",
}

pf.plot_boxplots(data, colors, savename="test.png")