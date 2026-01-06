import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.spatial import cKDTree
import earthkit.data as ekd 
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_surface_field(ds_dataset, sur_field, timestep, title, unit, savename, colormap='RdBu_r'):
    lons = ds_dataset["longitudes"].values.ravel()
    lats = ds_dataset["latitudes"].values.ravel()
    
    times = ds_dataset.dates.values
    init_time = np.datetime64("2022-07-01T00:00")
    t0 = np.where(times == init_time)[0][0]
    
    # Extract date and field data
    if hasattr(sur_field, 'time'):
        # xarray DataArray with time coordinate
        date = sur_field.time[timestep].values
        field_plot = sur_field.isel(time=timestep).values.ravel()
    else:
        # numpy array - fall back to original date indexing
        date = ds_dataset.dates.isel(time=(t0 + timestep)).values
        field_plot = sur_field[timestep].ravel()
    
    
    # remove NaNs/infs
    mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(field_plot)
    lons, lats, field_plot = lons[mask], lats[mask], field_plot[mask]
    # Handle antimeridian: convert from 0-360 to -180 to 180 to avoid white band at 0 longitude
    lons = np.where(lons > 180, lons - 360, lons)

    proj = ccrs.PlateCarree()
    
    # Choose normalization and colormap safely.
    # TwoSlopeNorm requires vmin < vcenter < vmax. For fields that don't
    # cross zero (e.g., surface pressure), fall back to a linear Normalize
    # and use a suitable sequential colormap to avoid the ValueError.
    vmin, vmax = np.min(field_plot), np.max(field_plot)
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap_use = colormap
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Pick a sensible sequential colormap when data are all positive
        # or all negative to avoid misleading diverging maps.
        if vmin >= 0:
            cmap_use = 'winter'
        elif vmax <= 0:
            cmap_use = 'Blues_r'
        else:
            cmap_use = colormap

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    cf = ax.tricontourf(lons, lats, field_plot, 40, transform=proj, norm=norm, cmap=cmap_use)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title(f"{title} at {str(date)[:13]}", fontsize=14)
    cbar = plt.colorbar(cf, ax=ax, pad=0.05)
    cbar.set_label(f"({unit})", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"images/{savename}", dpi =150)
    plt.close()
    
    
def plot_multiple_lines(series_dict, x=None, xlabel="", ylabel="", title="", savename=None, linestyle="--"):
    """
    series_dict: dict where keys are labels and values are 1D arrays.
    x: optional x-array. If None, use index of values.
    linestyle: can be:
        - a string (e.g., "--") to use same style for all lines
        - a dict mapping labels to linestyles for per-line control
    """
    plt.figure()

    for label, y in series_dict.items():
        # Determine linestyle for this line
        if isinstance(linestyle, dict):
            ls = linestyle.get(label, "-")  # default to solid if label not in dict
        else:
            ls = linestyle
        
        if x is None:
            plt.plot(y, label=label, linestyle=ls)
        else:
            plt.plot(x, y, label=label, linestyle=ls)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/{savename}", dpi =150)
    plt.close()
   
    
    
def plot_boxplots(data_dict, colors, savename, ylabel="Mean absolute error", figsize_per_subplot=(2.2, 5), sharey = True):
    """
    Parameters
    ----------
    data_dict : dict
        {subplot_name: {box_label: array_like}}
    colors : dict
        {box_label: color}
    ylabel : str
        Shared y-axis label
    figsize_per_subplot : tuple
        Size per subplot (width, height)
    show : bool
        Whether to call plt.show()
    """

    n_subplots = len(data_dict)
    fig_width = figsize_per_subplot[0] * n_subplots
    fig_height = figsize_per_subplot[1]

    fig, axes = plt.subplots(
        1, n_subplots,
        figsize=(fig_width, fig_height),
        sharey=sharey
    )

    # Make axes iterable if only one subplot
    if n_subplots == 1:
        axes = [axes]

    # Track legend handles
    legend_handles = []
    legend_labels = []

    for ax, (subplot_name, box_data) in zip(axes, data_dict.items()):
        labels = list(box_data.keys())
        values = [box_data[label] for label in labels]

        positions = np.arange(len(labels)) + 1

        bp = ax.boxplot(
            values,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False
        )

        # Color boxes and create legend entries (only for first subplot)
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors.get(label, "gray"))
            patch.set_edgecolor("black")
            
            # Add to legend only once
            if len(legend_handles) < len(labels):
                legend_handles.append(patch)
                legend_labels.append(label)

        # Median styling
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.set_xticks([])
        ax.set_xlabel(subplot_name, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(ylabel, fontsize=12)
    
    # Add legend to the figure
    fig.legend(legend_handles, legend_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=len(legend_labels), 
               frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(f"./images/{savename}", dpi=150)
    plt.close()

    