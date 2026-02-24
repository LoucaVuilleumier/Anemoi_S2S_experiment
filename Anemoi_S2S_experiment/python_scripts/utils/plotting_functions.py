import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.spatial import cKDTree
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
    plt.savefig(f"{savename}", dpi =150)
    plt.close()
    
    
def plot_multiple_lines(series_dict, x=None, xlabel="", ylabel="", title="", savename=None, linestyle="--", transpose=False, flip_y= False):
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
            if transpose==False:
                plt.plot(x, y, label=label, linestyle=ls)
            elif transpose==True:
                plt.plot(y, x, label=label, linestyle=ls)

    # Apply y-axis inversion ONCE after all series are plotted
    if transpose and flip_y:
        plt.gca().invert_yaxis()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
        
    plt.savefig(f"{savename}", dpi =150)
    plt.close()
   

def plot_weekly_lines(data_dict, title, colors, savename, ylabel="Metric", xlabel="Forecast Week", 
                      figsize=(10, 6), ylim=None, add_markers=True):
    """
    Plot line graphs showing metric evolution across forecast weeks.
    
    Parameters
    ----------
    data_dict : dict
        {variable_name: array_like} where array has length equal to number of weeks
    title : str
        Plot title
    colors : dict
        {variable_name: color}
    savename : str
        Path to save the figure
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label (default: "Forecast Week")
    figsize : tuple
        Figure size (width, height)
    ylim : tuple, optional
        Y-axis limits (ymin, ymax)
    add_markers : bool
        Whether to add markers to the lines
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get number of weeks from first variable
    n_weeks = len(next(iter(data_dict.values())))
    weeks = np.arange(1, n_weeks + 1)
    
    # Plot each variable
    for var_name, values in data_dict.items():
        if add_markers:
            ax.plot(weeks, values, label=var_name, color=colors.get(var_name, 'gray'), 
                   marker='o', markersize=6, linewidth=2)
        else:
            ax.plot(weeks, values, label=var_name, color=colors.get(var_name, 'gray'), 
                   linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(weeks)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight')
    plt.close()

    
def plot_boxplots(data_dict, title, colors, savename, ylabel="Mean absolute error", figsize_per_subplot=(2.2, 5), sharey = True):
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
            showfliers=False,
            manage_ticks=False  # Faster rendering
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
    
    # Add title
    fig.suptitle(title, fontsize=16, y=1.08)
    
    # Add legend below the title
    fig.legend(legend_handles, legend_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=len(legend_labels), 
               frameon=True, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{savename}", dpi=150, bbox_inches='tight')
    plt.close()



def plot_weekly_spatial_maps(dataset, list_variables, list_weeks, label, subtitle, savename):
    
    
    fig = plt.figure(figsize=(20, 16))

    variables = list_variables
    #var_names_full = ['2m Temperature', 'Total Precipitation', '10m U Wind', '10m V Wind']
    weeks = list_weeks  # Weeks 1, 3, 5, 7 (0-indexed)

    proj = ccrs.PlateCarree()
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for i, var in enumerate(variables):
        for j, week in enumerate(weeks):
            ax = fig.add_subplot(len(list_weeks), len(list_variables), i*len(list_weeks) + j + 1, projection=proj)
            ax.set_global()
            
            # Get data for this variable and week
            data_week = dataset[var].isel(leadtime=week).values.ravel()
            lons = dataset['longitude'].values.ravel()
            lats = dataset['latitude'].values.ravel()
            
            # Handle antimeridian
            lons = np.where(lons > 180, lons - 360, lons)
            
            # Remove NaNs/infs
            mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(data_week)
            lons_plot, lats_plot, data_plot = lons[mask], lats[mask], data_week[mask]
            
            # Plot using tricontourf
            im = ax.tricontourf(lons_plot, lats_plot, data_plot, 40, 
                            transform=proj, norm=norm, cmap='RdBu_r')
            
            # Add coastlines
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            # Set title
            ax.set_title(f'{var}\nWeek {week+1}', fontsize=10, fontweight='bold')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'{label}', fontsize=12)

    plt.suptitle(subtitle, 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])

    plt.savefig(f'{savename}.png', 
                dpi=300, bbox_inches='tight')
    print("R_t spatial maps saved!")


def plot_single_var_spatial_rmse(dataset, var, weeks, var_label, unit, suptitle, savename, cmap='YlOrRd'):
    """Plot spatial RMSE maps for a single variable across several weeks.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with spatial RMSE values, longitude, latitude, and a leadtime dim.
    var : str
        Short variable name (key in *dataset*), e.g. '2t', 'tp'.
    weeks : list[int]
        0-based leadtime indices to plot.
    var_label : str
        Human-readable variable name for title.
    unit : str
        Physical unit string shown on the colour-bar.
    suptitle : str
        Figure super-title.
    savename : str
        Full path (with extension) where the figure is saved.
    cmap : str, optional
        Matplotlib sequential colourmap name (default 'YlOrRd').
    """
    proj = ccrs.PlateCarree()
    n_weeks = len(weeks)
    fig, axes = plt.subplots(1, n_weeks, figsize=(5.5 * n_weeks, 5),
                             subplot_kw={"projection": proj})
    if n_weeks == 1:
        axes = [axes]

    # Shared colour range across the selected weeks
    all_data = []
    for week in weeks:
        d = dataset[var].isel(leadtime=week).values.ravel()
        all_data.append(d[np.isfinite(d)])
    all_data = np.concatenate(all_data)
    vmin, vmax = 0, np.nanpercentile(all_data, 99)
    norm = Normalize(vmin=vmin, vmax=vmax)

    lons = dataset['longitude'].values.ravel()
    lats = dataset['latitude'].values.ravel()
    lons = np.where(lons > 180, lons - 360, lons)

    for j, week in enumerate(weeks):
        ax = axes[j]
        ax.set_global()

        data_week = dataset[var].isel(leadtime=week).values.ravel()
        mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(data_week)
        lons_plot, lats_plot, data_plot = lons[mask], lats[mask], data_week[mask]

        im = ax.tricontourf(lons_plot, lats_plot, data_plot, 40,
                            transform=proj, norm=norm, cmap=cmap)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_title(f'Week {week + 1}', fontsize=11, fontweight='bold')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'RMSE [{unit}]', fontsize=11)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    plt.savefig(savename, dpi=300, bbox_inches='tight')
    print(f"Spatial RMSE map for {var_label} saved!")