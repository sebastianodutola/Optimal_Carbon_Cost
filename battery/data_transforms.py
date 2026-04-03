from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .models import CarbonTS


def parquet_to_carbon_series(
    data_path: str | Path,
    settlement_interval: np.timedelta64 = np.timedelta64(30, "m"),
) -> tuple[CarbonTS, CarbonTS]:
    """Load a parquet file and return (forecast, actual) as CarbonTS objects."""
    df = pq.read_table(data_path).to_pandas()
    time_arr = df["from"].apply(np.datetime64).to_numpy()
    f = df["intensity"].apply(lambda i: i["forecast"]).to_numpy()
    a = df["intensity"].apply(lambda i: i["actual"]).to_numpy()
    return (
        CarbonTS(time_arr, settlement_interval, f),
        CarbonTS(time_arr, settlement_interval, a),
    )


def pandas_to_carbon_series(
    df: pd.DataFrame,
    settlement_interval: np.timedelta64 = np.timedelta64(30, "m"),
) -> tuple[CarbonTS, CarbonTS]:
    """Convert a DataFrame with 'from', 'forecast', 'actual' columns to (forecast, actual)."""
    time_arr = df["from"].apply(np.datetime64).to_numpy()
    f = df["forecast"].to_numpy()
    a = df["actual"].to_numpy()
    return (
        CarbonTS(time_arr, settlement_interval, f),
        CarbonTS(time_arr, settlement_interval, a),
    )


def daily_to_datetime(
    t_intervals: list[tuple[time, time]],
    d: date | np.datetime64,
) -> np.ndarray:
    """Convert (start_time, end_time) time-of-day pairs to absolute datetime64 interval array."""
    if isinstance(d, np.datetime64):
        d = d.astype("datetime64[D]").item().date()
    intervals = [
        (datetime.combine(d, s), datetime.combine(d, e)) for s, e in t_intervals
    ]
    return np.array(intervals, dtype=np.datetime64)
