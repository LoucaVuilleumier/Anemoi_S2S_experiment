from omegaconf import OmegaConf
from anemoi.datasets import open_dataset
import numpy as np

        
ds = open_dataset(
    {"concat": [
        {'dataset': 'aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8',
        'start': 2000,
        'end': 2006},
        {'dataset': 'aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8',
        'start': 2012,
        'end': 2023}
    ], "fill_missing_gaps": True},
    rolling_average=(-28, 0, "freq")
)
date = "2012-01-01T00:00:00"
idx_date = int(np.where(np.array(ds.dates) == np.datetime64(date))[0][0])
ds[idx_date, ...]
