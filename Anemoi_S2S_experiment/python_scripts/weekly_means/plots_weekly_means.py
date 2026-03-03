import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
sys.path.append('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts')
from utils import plotting_functions as pf
from utils import metrics_function as mf
from collections import defaultdict
import importlib
importlib.reload(pf)

from matplotlib.colors import TwoSlopeNorm, Normalize





# Colors for variables
color_vars = {
    "2m Temperature": "#d62728",  # red
    "Total Precipitation": "#1f77b4",  # blue
    "10m U Wind": "#2ca02c",  # green
    "10m V Wind": "#ff7f0e",  # orange
}

#ACC path and loading
acc_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/ACC_weekly_anomalies_AIFS.nc"
ACC_ds = xr.open_dataset(acc_path)

R_t_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/R_t_weekly_anomalies_AIFS.nc"
R_t_ds = xr.open_dataset(R_t_path)

#RMSE path and loading
rmse_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/RMSE_weekly_AIFS.nc"
RMSE_ds = xr.open_dataset(rmse_path)

spatial_rmse_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Spatial_RMSE_weekly_AIFS.nc"
SPATIAL_RMSE_ds = xr.open_dataset(spatial_rmse_path)



#Roc path and loading
roc_datasets = {}
for i in range(8):
    path = f"/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/ROC_week_{i}_AIFS.nc"
    roc_datasets[f"week_{i}"] = xr.open_dataset(path)
    
#CRPS path and loading
crps_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/CRPS_weekly_AIFS.nc"
CRPS_ds = xr.open_dataset(crps_path)

#SKill/Spread path and loading
spead_skill_path = "/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/output_metrics/AIFS/Spread_Skill_weekly_AIFS.nc"
spead_skill_ds = xr.open_dataset(spead_skill_path)

# Line plot data for ACC (ensemble mean metrics)
lineplot_acc = {
    "2m Temperature": ACC_ds["2t"].mean(dim="init_date").values,
    "Total Precipitation": ACC_ds["tp"].mean(dim="init_date").values,
    "10m U Wind": ACC_ds["10u"].mean(dim="init_date").values,
    "10m V Wind": ACC_ds["10v"].mean(dim="init_date").values
}

# Line plot for ACC
pf.plot_weekly_lines(
    data_dict=lineplot_acc,
    title='ACC for Weekly Forecasts of Reference Model (Ensemble Mean)',
    colors=color_vars,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/acc_AIFS.png',
    ylabel='Anomaly Correlation Coefficient',
    ylim=(-.2, 1)
)

# Line plot data for RMSE (ensemble mean metrics)
lineplot_rmse = {
    "2m Temperature": RMSE_ds["2t"].mean(dim="init_date").values,
    "Total Precipitation": RMSE_ds["tp"].mean(dim="init_date").values,
    "10m U Wind": RMSE_ds["10u"].mean(dim="init_date").values,
    "10m V Wind": RMSE_ds["10v"].mean(dim="init_date").values
}

# Line plot for RMSE
pf.plot_weekly_lines(
    data_dict=lineplot_rmse,
    title='RMSE for Weekly Forecasts of Reference Model (Ensemble Mean)',
    colors=color_vars,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/rmse_AIFS.png',
    ylabel='Root Mean Square Error',
    ylim=None  # Auto scale for RMSE since different variables have different scales
)


#line plot of crps
pf.plot_weekly_lines(
    data_dict={
        "2m Temperature": CRPS_ds["2t"].mean(dim="init_date").values,
        "Total Precipitation": CRPS_ds["tp"].mean(dim="init_date").values,
        "10m U Wind": CRPS_ds["10u"].mean(dim="init_date").values,
        "10m V Wind": CRPS_ds["10v"].mean(dim="init_date").values
    },
    title='fair CRPS for Weekly Forecasts of Reference Model',
    colors=color_vars,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/crps_AIFS.png',
    ylabel='Continuous Ranked Probability Score',
    ylim=None  # Auto scale for CRPS since different variables have different scales
)

#line plot of spread/skill ratio
pf.plot_weekly_lines(
    data_dict={
        "2m Temperature": spead_skill_ds["2t"].mean(dim="init_date").values,
        "Total Precipitation": spead_skill_ds["tp"].mean(dim="init_date").values,
        "10m U Wind": spead_skill_ds["10u"].mean(dim="init_date").values,
        "10m V Wind": spead_skill_ds["10v"].mean(dim="init_date").values
    },
    title='Spread/Skill Ratio for Weekly Forecasts of Reference Model',
    colors=color_vars,
    savename='/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/spread_skill_AIFS.png',
    ylabel='Spread/Skill Ratio',
    ylim=None  # Auto scale for Spread/Skill Ratio
)


#spatial maps of RMSE — one figure per variable, 4 weeks each
spatial_vars = ['2t', 'tp', '10u', '10v']
spatial_var_labels = {"2t": "2m Temperature", "tp": "Total Precipitation",
                      "10u": "10m U Wind", "10v": "10m V Wind"}
spatial_var_units = {"2t": "K", "tp": "m", "10u": "m/s", "10v": "m/s"}
selected_weeks = [0, 2, 4, 6]

for var in spatial_vars:
    pf.plot_single_var_spatial_rmse(
        dataset=SPATIAL_RMSE_ds,
        var=var,
        weeks=selected_weeks,
        var_label=spatial_var_labels[var],
        unit=spatial_var_units[var],
        suptitle=f'Ensemble Mean RMSE — {spatial_var_labels[var]}',
        savename=f'/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/spatial_rmse_{var}_AIFS.png',
    )

#Spatial map of R_t
pf.plot_weekly_spatial_maps(R_t_ds, ['2t', 'tp', '10u', '10v'], [0, 2, 4, 6], "Temporal CC",
                            'Temporal Correlation Coefficient of Reference Model',
                            '/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/R_t_AIFS.png')


# ROC curves — one subplot per week, all variables overlaid
var_short = ["2t", "tp", "10u", "10v"]
var_labels = {"2t": "2m Temperature", "tp": "Total Precipitation",
              "10u": "10m U Wind", "10v": "10m V Wind"}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, (week_key, roc_ds) in enumerate(roc_datasets.items()):
    ax = axes[i]
    for var in var_short:
        auc_val = roc_ds[f"{var}_AUC"].item()
        ax.plot(roc_ds[f"{var}_POFD"], roc_ds[f"{var}_POD"],
                color=color_vars[var_labels[var]],
                label=f"{var_labels[var]} (AUC={auc_val:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("POFD")
    ax.set_ylabel("POD")
    ax.set_title(f"ROC — Week {i}")
    ax.legend(loc="lower right", fontsize=7)

fig.suptitle("Receiver Operating Characteristic per Week", fontsize=14)
plt.tight_layout()
plt.savefig('/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images/roc_AIFS.png', dpi=150)
plt.show()

