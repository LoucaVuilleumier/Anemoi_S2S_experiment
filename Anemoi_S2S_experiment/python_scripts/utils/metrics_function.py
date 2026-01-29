import xarray as xr
import numpy as np

def compute_relative_mse(sur_field_dataset, sur_field_inference):
    """
    Compute time-resolved relative MSE:
        rMSE(t) = MSE(t) / Var_dataset(t)
    preserving the time dimension.
    """
    # Difference field
    diff = sur_field_dataset - sur_field_inference

    # MSE over the spatial dimension
    mse = (diff ** 2).mean(dim="values")

    # Variance of the truth over the same spatial dimension
    var = sur_field_dataset.var()

    # Return time series (xarray DataArray)
    return mse / var

def compute_mae_from_dataset(ds_dataset, ds_inference, variable, squash = False, keep_time = False):
    """
    Compute MAE between dataset and inference for a given variable. variable can be a list of variables.
    If squash is True and variable is a list, the MAE is computed on the squashed variables (divided by their standard deviation).
    """
    #n_steps = len(ds_inference.time)
    #init_time = ds_inference.time[0].values
    #t0 = np.where(ds_dataset.dates.values == init_time)[0][0]
    #ds_dataset = ds_dataset.isel(time = slice(t0, t0 + n_steps))
    
    #select variable(s) from dataset
    #var_names = ds_dataset.attrs["variables"]
    
    #def get_var_dataset(dataset, variable):
    #    idx_variable = var_names.index(variable)
    #    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})

    #if isinstance(variable, list):
    #    variable_dataset = xr.Dataset({var: get_var_dataset(ds_dataset, var) for var in variable})
    #else:        
    #    idx_variable = var_names.index(variable)
    #    variable_dataset = get_var_dataset(ds_dataset, variable)
    
    #select standard deviation from dataset (only useful for squash option)
    #stdev = ds_dataset["stdev"].values
    
    variable_dataset = ds_dataset[variable]
    
    #select variable(s) from inference
    variable_inference = ds_inference[variable]
    
    
    
    #compute MAE
    if isinstance(variable, list) and squash == True:
        mae = []
        for var in variable:
            var_dataset = get_var_dataset(ds_dataset, var)
            var_inference = variable_inference[var]
            stdev_var = stdev[var_names.index(var)]
            mae_var = np.mean(np.abs((var_dataset - var_inference) / stdev_var))
            mae.append(mae_var)
        mae = np.mean(mae)
    elif isinstance(variable, list) and squash == False:
        if keep_time ==False:
            mae = (np.abs(variable_dataset - variable_inference)).mean()
        elif keep_time == True:
            mae = (np.abs(variable_dataset - variable_inference)).mean(dim="values")  #keep time dimension
        
    else:
        if keep_time ==False:
            mae = (np.abs(variable_dataset - variable_inference)).mean().values.item()
        elif keep_time == True:
            mae = (np.abs(variable_dataset - variable_inference)).mean(dim="values").values  #keep time dimension
        
    return mae


def compute_msss(ds_dataset, ds_inference, ds_inference_finetuned, variable, squash = False, keep_time = False):
    """
    Compute MSSS as 1-(MSE model finetuned / MSE model). variable can be a list of variables.
    If squash is True and variable is a list, the MSSS is computed on the squashed variables 
    """
    #n_steps = len(ds_inference.time)
    #init_time = ds_inference.time[0].values
    #t0 = np.where(ds_dataset.dates.values == init_time)[0][0]
    #ds_dataset = ds_dataset.isel(time = slice(t0, t0 + n_steps))
    
    #select variable(s) from dataset
    #var_names = ds_dataset.attrs["variables"]
    
    #def get_var_dataset(dataset, variable):
    #    idx_variable = var_names.index(variable)
    #    return dataset["data"].isel(variable = idx_variable).squeeze(dim="ensemble").rename({"cell": "values"})

    #if isinstance(variable, list):
    #    variable_dataset = xr.Dataset({var: get_var_dataset(ds_dataset, var) for var in variable})
    #else:        
    #    idx_variable = var_names.index(variable)
    #    variable_dataset = get_var_dataset(ds_dataset, variable)
    
    #select variable()s from dataset
    variable_dataset = ds_dataset[variable]
        
    #select variable(s) from inference
    variable_inference = ds_inference[variable]
    variable_inference_finetuned = ds_inference_finetuned[variable]
    
    #compute msss
    if isinstance(variable, list) and squash == False:
        if keep_time ==False:
            mse_ref = (np.square(variable_dataset - variable_inference)).mean()
            mse_finetuned = (np.square(variable_dataset - variable_inference_finetuned)).mean()
            msss = 1 - (mse_finetuned / mse_ref)
        elif keep_time == True:
            mse_ref = np.square(variable_dataset - variable_inference).mean(dim="values")  #keep time dimension
            mse_finetuned = np.square(variable_dataset - variable_inference_finetuned).mean(dim="values")  #keep time dimension
            msss = 1 - (mse_finetuned / mse_ref)
        
    elif isinstance(variable, list) and squash == True:
        msss_list = []
        for var in variable:
            var_dataset = ds_dataset[var]
            var_inference = ds_inference[var]
            var_inference_finetuned = ds_inference_finetuned[var]
            #compute MSE
            mse_ref = np.mean(np.square((var_dataset - var_inference)))
            mse_finetuned = np.mean(np.square((var_dataset - var_inference_finetuned)))
            msss_var = 1 - (mse_finetuned / mse_ref)
            msss_list.append(msss_var)
        msss = float(np.mean(msss_list))
        
    else:
        if keep_time == False:
            mse_ref = np.mean(np.square(variable_dataset - variable_inference))
            mse_finetuned = np.mean(np.square(variable_dataset - variable_inference_finetuned))
            msss = 1 - (mse_finetuned / mse_ref).values.item()
        elif keep_time == True:
            mse_ref = np.square(variable_dataset - variable_inference).mean(dim="values")  #keep time dimension
            mse_finetuned = np.square(variable_dataset - variable_inference_finetuned).mean(dim="values")  #keep time dimension
            msss = 1 - (mse_finetuned / mse_ref)

    return msss
    
    
def compute_acc(anomaly_dataset, anomaly_inference, weights=None):
    """
    Compute Anomaly Correlation Coefficient (ACC) between dataset and inference anomalies.
    
    Parameters:
    -----------
    anomaly_dataset : xarray.DataArray
        Dataset anomalies with dimensions 'time' and 'values' (spatial)
    anomaly_inference : xarray.DataArray
        Inference anomalies with dimensions 'time' and 'values' (spatial)
    weights : xarray.DataArray, optional
        Weights for spatial averaging (e.g., cos(latitude) for area weighting)

        
    Returns:
    --------
    acc_spatial : xarray.DataArray
        Spatially-averaged ACC with dimension 'time'
    acc_pointwise : xarray.DataArray
        Point-wise ACC with dimension 'values' (ACC at each grid point, computed over time)
    """
    
    # Compute spatially-averaged ACC
    if weights is not None:
        numerator_spatial = (anomaly_dataset * anomaly_inference).weighted(weights).mean(dim="values")
        denominator_spatial = np.sqrt(
            (anomaly_dataset ** 2).weighted(weights).mean(dim="values") * 
            (anomaly_inference ** 2).weighted(weights).mean(dim="values")
        )
    else:
        numerator_spatial = (anomaly_dataset * anomaly_inference).mean(dim="values")
        denominator_spatial = np.sqrt((anomaly_dataset ** 2).mean(dim="values") * (anomaly_inference ** 2).mean(dim="values"))
    
    acc_spatial = numerator_spatial / denominator_spatial
     
    return acc_spatial


def compute_rt(anomaly_dataset, anomaly_inference):
    """
    Compute temporal correlation coefficient (R_t) at each grid point for each lead time.
    Correlation is computed across runs (temporal dimension).
    
    Parameters:
    -----------
    anomaly_dataset : xarray.DataArray
        Dataset anomalies with dimensions (run, leadtime, values)
    anomaly_inference : xarray.DataArray
        Inference anomalies with dimensions (run, leadtime, values)
        
    Returns:
    --------
    rt : xarray.DataArray
        Temporal correlation coefficient with dimensions (leadtime, values)
        Shows correlation at each grid point for each lead time, computed across runs
    """
    
    # Compute R_t: correlation over 'run' dimension for each (leadtime, values) combination
    numerator = (anomaly_dataset * anomaly_inference).mean(dim="run")
    denominator = np.sqrt(
        (anomaly_dataset ** 2).mean(dim="run") * 
        (anomaly_inference ** 2).mean(dim="run")
    )
    
    rt = numerator / denominator
    
    return rt
