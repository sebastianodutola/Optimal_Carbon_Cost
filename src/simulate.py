from pathlib import Path
from datetime import time, date, timedelta
from multiprocessing import Pool
from battery import run_comparative_simulation, CarbonTS
import pandas as pd
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
DATA_PATH = _ROOT / "Data" / "daily"
DATA_PATH_ACTUAL = _ROOT / "Data" / "actual" / "2025.parquet"
OUTPUT_PATH = _ROOT / "Data" / "results"
START = date(2025, 1, 1)
END = date(2026, 1, 1)

# Actual Carbon Data
df_actual = pd.read_parquet(DATA_PATH_ACTUAL)
actual_carbon_ts = CarbonTS(
    time=df_actual["period_start"]
    .dt.tz_localize(None)
    .to_numpy()
    .astype("datetime64[m]"),
    intensity=df_actual["actual"].to_numpy(),
    settlement_interval=np.timedelta64(30, "m"),
)

EV_SCHEDULE = [
    (time(8, 30), time(9, 0), 6),
    (time(17, 45), time(18, 30), 6),
]

LAPTOP_SCHEDULE = [
    (time(8, 0), time(12, 0), 0.020),
    (time(13, 0), time(16, 0), 0.020),
    (time(18, 0), time(20, 0), 0.060),
]
LAPTOP_AVAILABILITY = [
    (time(0, 0), time(10, 0)),
    (time(17, 0), time(23, 59, 59)),
]


def create_scenarios(horizon_type: str):
    if horizon_type not in ("48h", "24h"):
        raise ValueError("horizon type must be either '48h' or '24h'")
    scenarios = [
        dict(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            output_file_name=f"ev_home_work_small_cap_{horizon_type}.parquet",
            actual_carbon_ts=actual_carbon_ts,
            start=START,
            end=END,
            schedule=EV_SCHEDULE,
            availability=[
                (time(0, 0), time(8, 0)),
                (time(9, 0), time(17, 30)),
                (time(18, 30), time(23, 59, 59)),
            ],
            capacity=50.0,
            charging_rate=10.0,
            efficiency=0.95,
            initial_charge=0.0,
            horizon_type=horizon_type,
        ),
        dict(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            output_file_name=f"ev_home_work_{horizon_type}.parquet",
            actual_carbon_ts=actual_carbon_ts,
            start=START,
            end=END,
            schedule=EV_SCHEDULE,
            availability=[
                (time(0, 0), time(8, 0)),
                (time(9, 0), time(17, 30)),
                (time(18, 30), time(23, 59, 59)),
            ],
            capacity=100.0,
            charging_rate=10.0,
            efficiency=0.95,
            initial_charge=0.0,
            horizon_type=horizon_type,
        ),
        dict(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            output_file_name=f"ev_home_only_{horizon_type}.parquet",
            actual_carbon_ts=actual_carbon_ts,
            start=START,
            end=END,
            schedule=EV_SCHEDULE,
            availability=[
                (time(0, 0), time(8, 0)),
                (time(18, 30), time(23, 59, 59)),
            ],
            capacity=100.0,
            charging_rate=10.0,
            efficiency=0.95,
            initial_charge=0.0,
            horizon_type=horizon_type,
        ),
        dict(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            output_file_name=f"laptop_{horizon_type}.parquet",
            actual_carbon_ts=actual_carbon_ts,
            start=START,
            end=END,
            schedule=LAPTOP_SCHEDULE,
            availability=LAPTOP_AVAILABILITY,
            capacity=0.1,
            charging_rate=0.06,
            efficiency=0.95,
            initial_charge=0.0,
            horizon_type=horizon_type,
        ),
    ]
    return scenarios


def _run(kwargs):
    run_comparative_simulation(**kwargs)
    print(f"Done: {kwargs['output_file_name']}")


if __name__ == "__main__":
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    long_forecast_scene = create_scenarios("48h")
    short_forecast_scene = create_scenarios("24h")
    scenarios = long_forecast_scene + short_forecast_scene
    with Pool(processes=len(scenarios)) as pool:
        pool.map(_run, tqdm(scenarios))
