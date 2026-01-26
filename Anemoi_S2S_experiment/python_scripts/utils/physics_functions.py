import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from scipy.spatial import cKDTree
import earthkit.data as ekd 
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def compute_r_sur(t2m, dp2m, sp):
    """
    Compute water-vapor mixing ratio at the surface (g/kg).

    Parameters:
    -----------
    t2m : array-like or xr.DataArray
        Temperature in Celsius
    dp2m : array-like or xr.DataArray
        Dew point temperature in Celsius
    sp : array-like or xr.DataArray
        Surface pressure in hPa

    Returns:
    --------
    r_sur : array-like or xr.DataArray
        Mixing ratio in g/kg. Same type as input (numpy array or DataArray).
        If input is DataArray, output preserves coordinates and dimensions.
    """
    # Detect if inputs are xarray.DataArray
    is_xarray = isinstance(t2m, xr.DataArray)
    
    # Work with numpy arrays internally
    t2m_np = t2m.values if is_xarray else t2m
    dp2m_np = dp2m.values if is_xarray else dp2m
    sp_np = sp.values if is_xarray else sp
    
    # water vapor partial pressure (hPa) using Magnus formula
    e = np.where(
        t2m_np >= 0.0,
        6.107 * np.exp((17.368 * dp2m_np) / (dp2m_np + 238.83)),
        6.108 * np.exp((17.856 * dp2m_np) / (dp2m_np + 245.52)),
    )
    # Avoid division by zero: ensure sp > e
    e = np.maximum(e, 0.0)
    sp_np = np.maximum(sp_np, 1e-6)
    denom = sp_np - e
    denom = np.where(denom > 0, denom, np.nan)
    r_sur_np = 622.0 * (e / denom)
    
    # Return as xarray.DataArray if input was DataArray
    if is_xarray:
        r_sur = xr.DataArray(r_sur_np, coords=t2m.coords, dims=t2m.dims, name='r_sur')
        return r_sur
    else:
        return r_sur_np

def compute_rh_sur(t2m, dp2m, clip_for_plot=False):
    """
    Compute relative humidity at the surface (%).

    Parameters:
    -----------
    t2m : array-like or xr.DataArray
        Temperature in Celsius
    dp2m : array-like or xr.DataArray
        Dew point temperature in Celsius
    clip_for_plot : bool, default False
        If True, clip RH to [0, 100] for visualization only.
        If False (default), return raw RH which may exceed 100% due to
        numerical noise; use this for loss functions and residuals.

    Returns:
    --------
    rh_sur : array-like or xr.DataArray
        Relative humidity in percent (%).
        Raw values (can be > 100%) if clip_for_plot=False,
        clipped to [0, 100] if clip_for_plot=True.
        Same type as input (numpy array or DataArray).
        If input is DataArray, output preserves coordinates and dimensions.
    """
    # Detect if inputs are xarray.DataArray
    is_xarray = isinstance(t2m, xr.DataArray)
    
    # Work with numpy arrays internally
    t2m_np = t2m.values if is_xarray else t2m
    dp2m_np = dp2m.values if is_xarray else dp2m
    
    # water vapor partial pressure (hPa)
    e = np.where(
        t2m_np >= 0.0,
        6.107 * np.exp((17.368 * dp2m_np) / (dp2m_np + 238.83)),
        6.108 * np.exp((17.856 * dp2m_np) / (dp2m_np + 245.52)),
    )
    # saturation water vapor pressure (hPa)
    e_sat = np.where(
        t2m_np >= 0.0,
        6.107 * np.exp((17.368 * t2m_np) / (t2m_np + 238.83)),
        6.108 * np.exp((17.856 * t2m_np) / (t2m_np + 245.52)),
    )
    # Avoid division by zero
    e = np.maximum(e, 0.0)
    e_sat = np.maximum(e_sat, 1e-12)
    rh_sur_np = 100.0 * e / e_sat
    
    if clip_for_plot:
        rh_sur_np = np.clip(rh_sur_np, 0.0, 100.0)
    
    # Return as xarray.DataArray if input was DataArray
    if is_xarray:
        rh_sur = xr.DataArray(rh_sur_np, coords=t2m.coords, dims=t2m.dims, name='rh_sur')
        return rh_sur
    else:
        return rh_sur_np
    
def virtual_temperature(T, q):
    """
    T : temperature [K]
    q : specific humidity [kg/kg]
    """
    return T * (1.0 + 0.61 * q)


def compute_T_hydro(p, zg, q):
    """
    Compute temperature profile from geopotential height profile
    using hydrostatic balance.

    Parameters
    ----------
    p : array (nlev,)         pressure levels [Pa], descending (surface -> top)
    zg : array (..., nlev)    geopotential [m²/s²] (ERA5 standard)
    q : array (..., nlev)     specific humidity [kg/kg]

    Returns
    -------
    T_hydro : array (..., nlev)
        Temperature profile [K] computed from hydrostatic balance.
        NaN for boundary levels (first and last) where centered difference cannot be computed.
    """
    Rd = 287.05      # J kg-1 K-1
    T_hydro = np.zeros_like(zg)

    # Centered differences for interior points
    for k in range(1, len(p)-1):
        dp = p[k+1] - p[k-1]
        dzg = zg[..., k+1] - zg[..., k-1]
        denom = Rd * (1.0 + 0.61 * q[..., k])
        T_hydro[..., k] = - (p[k] / denom) * (dzg / dp)
    return T_hydro