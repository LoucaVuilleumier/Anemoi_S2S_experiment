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



def plot_weekly_spatial_maps(dataset, list_variables, list_weeks, label, subtitle, savename, 
                            norm=None, cmap='RdBu_r'):
    """
    Plot spatial maps for multiple variables and forecast weeks.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with variables, longitude, latitude, and leadtime dimensions
    list_variables : list
        List of variable names to plot
    list_weeks : list
        List of week indices (0-based) to plot
    label : str or dict
        Colorbar label(s). Can be:
        - str: single label for all variables (single colorbar)
        - dict: {var_name: label} for per-variable colorbars
    subtitle : str
        Figure title
    savename : str
        Path to save figure (without extension)
    norm : Normalize object, dict, or None
        Colorbar normalization. Can be:
        - None: auto-compute TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        - Normalize object: single norm for all variables
        - dict: {var_name: norm_object} for per-variable norms
    cmap : str or dict
        Colormap. Can be:
        - str: single colormap for all variables (default 'RdBu_r')
        - dict: {var_name: cmap_string} for per-variable colormaps
    """
    
    fig = plt.figure(figsize=(20, 16))

    variables = list_variables
    weeks = list_weeks
    
    # Determine if we're using per-variable colorbars
    use_multiple_colorbars = isinstance(label, dict)
    
    # Convert single norm/cmap/label to dicts if needed
    if not isinstance(norm, dict):
        norm_dict = {var: norm if norm is not None else TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1) 
                     for var in variables}
    else:
        norm_dict = norm
    
    if not isinstance(cmap, dict):
        cmap_dict = {var: cmap for var in variables}
    else:
        cmap_dict = cmap
    
    if not isinstance(label, dict):
        label_dict = {var: label for var in variables}
    else:
        label_dict = label

    proj = ccrs.PlateCarree()
    
    # Store plot handles for colorbars
    var_images = {}

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
            
            # Plot using tricontourf with per-variable norm and cmap
            im = ax.tricontourf(lons_plot, lats_plot, data_plot, 40, 
                            transform=proj, norm=norm_dict[var], cmap=cmap_dict[var])
            
            # Store image for colorbar (one per variable)
            if var not in var_images:
                var_images[var] = im
            
            # Add coastlines
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            
            # Set title
            ax.set_title(f'{var}\nWeek {week+1}', fontsize=10, fontweight='bold')

    # Add colorbar(s)
    if use_multiple_colorbars:
        # Multiple colorbars - one per variable row
        # First apply tight_layout to get final subplot positions
        plt.tight_layout(rect=[0, 0, 0.91, 0.97])
        
        for idx, var in enumerate(variables):
            # Get the axes for this variable's row
            # Each variable occupies a row of subplots (one per week)
            first_subplot_idx = idx * len(weeks)
            
            # Get position of the first subplot in this row
            first_ax = fig.axes[first_subplot_idx]
            pos = first_ax.get_position()
            
            # Place colorbar to the right of this row, matching its height
            cbar_x = 0.92
            cbar_y = pos.y0
            cbar_width = 0.012
            cbar_height = pos.height
            
            cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
            
            # Create colorbar from a ScalarMappable with explicit norm instead of from image data
            # This ensures identical colorbar for the same norm, independent of actual data
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(norm=norm_dict[var], cmap=cmap_dict[var])
            sm.set_array([])  # Dummy array
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(label_dict[var], fontsize=9)
            cbar.ax.tick_params(labelsize=8)
    else:
        # Single colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        
        # Create colorbar from ScalarMappable for consistency
        from matplotlib.cm import ScalarMappable
        norm_to_use = norm_dict[variables[0]]
        cmap_to_use = cmap_dict[variables[0]]
        sm = ScalarMappable(norm=norm_to_use, cmap=cmap_to_use)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(label_dict[variables[0]], fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.91, 0.97])

    plt.suptitle(subtitle, 
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(f'{savename}.png', 
                dpi=300, bbox_inches='tight')
    print("spatial maps saved!")


def plot_single_var_spatial_rmse(dataset, var, weeks, var_label, unit, suptitle, savename, cmap='YlOrRd', models=None):
    """Plot spatial RMSE maps for a single variable across several weeks and models.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with spatial RMSE values, longitude, latitude, and a leadtime dim.
        Can optionally have a model dimension.
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
    models : list[str], optional
        List of model names to plot. If None, plots all models or assumes no model dim.
    """
    proj = ccrs.PlateCarree()
    n_weeks = len(weeks)
    
    # Check if dataset has model dimension
    has_model_dim = 'model' in dataset[var].dims
    
    if has_model_dim:
        if models is None:
            models = dataset['model'].values
        n_models = len(models)
        
        # Create grid: rows=models, columns=weeks
        fig, axes = plt.subplots(n_models, n_weeks, figsize=(5.5 * n_weeks, 5 * n_models),
                                 subplot_kw={"projection": proj})
        
        # Ensure axes is always 2D
        if n_models == 1 and n_weeks == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = axes.reshape(1, -1)
        elif n_weeks == 1:
            axes = axes.reshape(-1, 1)
        
        # Shared colour range across all selected weeks and models
        all_data = []
        for model in models:
            for week in weeks:
                d = dataset[var].sel(model=model).isel(leadtime=week).values.ravel()
                all_data.append(d[np.isfinite(d)])
        all_data = np.concatenate(all_data)
        vmin, vmax = 0, np.nanpercentile(all_data, 99)
        norm = Normalize(vmin=vmin, vmax=vmax)

        lons = dataset['longitude'].values.ravel()
        lats = dataset['latitude'].values.ravel()
        lons = np.where(lons > 180, lons - 360, lons)

        for i, model in enumerate(models):
            for j, week in enumerate(weeks):
                ax = axes[i, j]
                ax.set_global()

                data_week = dataset[var].sel(model=model).isel(leadtime=week).values.ravel()
                mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(data_week)
                lons_plot, lats_plot, data_plot = lons[mask], lats[mask], data_week[mask]

                im = ax.tricontourf(lons_plot, lats_plot, data_plot, 40,
                                    transform=proj, norm=norm, cmap=cmap)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                
                # Add title: week number for top row, model name for leftmost column
                if i == 0:
                    ax.set_title(f'Week {week + 1}', fontsize=11, fontweight='bold')
                if j == 0:
                    ax.text(-0.1, 0.5, model.replace('_', ' ').title(), 
                           transform=ax.transAxes, fontsize=11, fontweight='bold',
                           rotation=90, va='center', ha='right')

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'RMSE [{unit}]', fontsize=11)

        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 0.91, 0.95])
        
    else:
        # Original single-model behavior
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
    plt.close()
    print(f"Spatial RMSE map for {var_label} saved!")


def plot_model_comparison_subplots(dataset, variables, var_labels, ylabel, title, savename, 
                                   ylim=None, figsize=(14, 10), model_colors=None, 
                                   add_zero_line=False, secondary_dataset=None, 
                                   secondary_ylabel=None, secondary_linestyle='--',
                                   var_units=None, secondary_ylim=None, use_dual_axes=False):
    """
    Plot metric comparison across models with subplots for each variable.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with metric values, must have dimensions (model, init_date, leadtime)
        and model coordinate with model names
    variables : list of tuple
        List of (var_short, var_label) tuples, e.g., [("2t", "2m Temperature"), ...]
    var_labels : dict
        Mapping from variable short names to display labels (can be same as variables)
    ylabel : str
        Y-axis label (metric name)
    title : str
        Figure super-title
    savename : str
        Full path where the figure is saved (including extension)
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). If None, auto-scale per subplot
    figsize : tuple, optional
        Figure size (width, height), default (14, 10)
    model_colors : dict, optional
        Mapping from model names to colors. If None, uses default blue/orange
    add_zero_line : bool, optional
        Whether to add a horizontal line at y=0, default False
    secondary_dataset : xr.Dataset, optional
        Optional second dataset to overlay on the same plot with different linestyle
    secondary_ylabel : str, optional
        Label for the secondary metric (shown in legend or on right y-axis if use_dual_axes=True)
    secondary_linestyle : str, optional
        Linestyle for secondary dataset lines, default '--'
    var_units : dict, optional
        Mapping from variable short names to units, e.g., {"2t": "K", "tp": "m"}
    secondary_ylim : tuple, optional
        Y-axis limits for secondary axis (ymin, ymax). Only used if use_dual_axes=True
    use_dual_axes : bool, optional
        If True and secondary_dataset is provided, use dual y-axes. If False, plot on same axis. Default False.
    
    Returns
    -------
    None
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Default model colors if not provided
    if model_colors is None:
        model_colors = {
            'reference': '#1f77b4',  # blue
            'finetuning': '#ff7f0e'   # orange
        }
    
    for idx, (var_short, var_label) in enumerate(variables):
        ax = axes[idx]
        
        # Get unit for this variable if available
        unit = var_units.get(var_short, '') if var_units else ''
        unit_str = f' [{unit}]' if unit else ''
        
        # Plot each model for primary dataset
        for model in dataset.model.values:
            # Mean over init_date dimension
            metric_values = dataset[var_short].sel(model=model).mean(dim="init_date").values
            weeks = range(1, len(metric_values) + 1)
            # Add metric name to label only if secondary dataset is present
            label = f'{model.capitalize()} - {ylabel}' if secondary_dataset is not None else model.capitalize()
            ax.plot(weeks, metric_values, 
                    color=model_colors.get(model, 'gray'),
                    marker='o',
                    label=label,
                    linewidth=2,
                    linestyle='-')
        
        # Plot secondary dataset if provided
        if secondary_dataset is not None and use_dual_axes:
            # Use dual y-axis for secondary dataset
            ax2 = ax.twinx()  # Create secondary y-axis
            for model in secondary_dataset.model.values:
                metric_values = secondary_dataset[var_short].sel(model=model).mean(dim="init_date").values
                weeks = range(1, len(metric_values) + 1)
                label_suffix = f' - {secondary_ylabel}' if secondary_ylabel else ' (secondary)'
                ax2.plot(weeks, metric_values, 
                        color=model_colors.get(model, 'gray'),
                        marker='s',
                        label=f'{model.capitalize()}{label_suffix}',
                        linewidth=2,
                        linestyle=secondary_linestyle)
            
            # Set secondary y-axis label with units
            ax2.set_ylabel(secondary_ylabel + unit_str if secondary_ylabel else 'Secondary' + unit_str, 
                          fontsize=10, color='black')
            if secondary_ylim is not None:
                ax2.set_ylim(secondary_ylim)
            
            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
        elif secondary_dataset is not None:
            # Plot on same axis (for spread-skill comparison)
            for model in secondary_dataset.model.values:
                metric_values = secondary_dataset[var_short].sel(model=model).mean(dim="init_date").values
                weeks = range(1, len(metric_values) + 1)
                label_suffix = f' - {secondary_ylabel}' if secondary_ylabel else ' (secondary)'
                ax.plot(weeks, metric_values, 
                        color=model_colors.get(model, 'gray'),
                        marker='s',
                        label=f'{model.capitalize()}{label_suffix}',
                        linewidth=2,
                        linestyle=secondary_linestyle)
            ax.legend(loc='best', fontsize=9)
        else:
            ax.legend(loc='best', fontsize=9)
        
        # Set primary y-axis label with units
        # If secondary dataset is on same axis, combine the labels
        if secondary_dataset is not None and not use_dual_axes and secondary_ylabel:
            combined_ylabel = f'{ylabel} & {secondary_ylabel}{unit_str}'
        else:
            combined_ylabel = ylabel + unit_str
        ax.set_ylabel(combined_ylabel, fontsize=10, color='black')
        if ylim is not None:
            ax.set_ylim(ylim)
        
        ax.set_xlabel('Week', fontsize=10)
        ax.set_title(var_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if add_zero_line:
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(savename, dpi=150)
    plt.close()
    print(f"Model comparison plot saved to {savename}")