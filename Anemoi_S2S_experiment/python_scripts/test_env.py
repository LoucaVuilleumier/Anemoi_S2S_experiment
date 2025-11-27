import xarray as xr
import matplotlib.pyplot as plt

# Path to your NetCDF file
file_path = "/ec/res4/hpcperm/nld4584/output_inference/output_test_inference.nc"

# Load the NetCDF file
ds = xr.open_dataset(file_path)

# Inspect the variables
print(ds)

# Suppose your variable of interest is the first data variable in the dataset
var_name = list(ds.data_vars)[0]
data = ds[var_name]

# Quick visualization: 2D plot of the first timestep (if time exists)
if "time" in data.dims:
    data_slice = data.isel(time=0)
else:
    data_slice = data

plt.figure(figsize=(8,6))
data_slice.plot()
plt.title(f"{var_name} - first timestep")
plt.show()

