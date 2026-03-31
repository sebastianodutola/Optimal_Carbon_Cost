import pandas as pd 
import numpy as np
from pathlib import Path
from optimiser import Load, pandas_to_forecast, optimiser, naive_charge, charging_stats

# Define Reasonable Loads


DATAPATH = Path(__file__).parent.joinpath("Data/backtest_data.py")
data = pd.read_parquet(DATAPATH)

# validate data
ts = np.datetime64(data["from"].to_nump()))
ts_diff = ts[1:] - ts[:-1]

# deduplicate if necessary (backup)
data.drop_duplicates()

# check there are no gaps in settlement times
assert (ts_diff == ts_diff[0]).all()





