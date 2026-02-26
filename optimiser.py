from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Optional
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd


@dataclass
class Charge:
    start_time: np.datetime64
    end_time: np.datetime64
    delta_E: float
    carbon_cost: float  # gC0_2eq


@dataclass(order=True)
class Discharge:
    start_time: np.datetime64
    end_time: np.datetime64
    power: float  # kW


@dataclass
class Load:
    capacity: float  # kWh
    charging_rate: float  # kW
    discharges: np.ndarray[Discharge]
    efficiency: float = 1  # (0,1]


@dataclass
class Forecast:
    time: np.ndarray[np.datetime64]
    settlement_interval: np.timedelta64
    # forecast carbon_intensity gCO_2eq/kWh
    f_carbon_intensity: np.ndarray[float]
    # actual carbon intensity if it exists
    a_carbon_intensity: np.ndarray[float]

    def __len__():
        return time.shape(0)


def parquet_to_forecast(
    data_path: str, settlement_interval=np.timedelta64(30, "m")
) -> Forecast:
    table = pq.read_table(data_path)
    df = table.to_pandas()
    time = df["from"].apply(np.datetime64).to_numpy()
    f_carbon_intensity = df["intensity"].apply(lambda i: i["forecast"])
    a_carbon_intensity = df["intensity"].apply(lambda i: i["actual"])
    return Forecast(time, settlement_interval, f_carbon_intensity, a_carbon_intensity)


def optimiser(load: Load, forecast: Forecast) -> list[Charge]:
    list = []
    i_events = 0
    i_forecast = 0
    effective_charging_rate = load.efficiency * load.charging_rate
    discharges = load.discharges

    events = sorted(
        [(d.start, False) for d in discharges] + [(d.end, True) for d in discharges],
        key=lambda x: x[0],
    )

    f_times = forecast.time
    available = True

    state = []
    while i_events < len(events) and i_forecast < len(forecast):
        if events[i_events][0] < f_times[i_forecast]:
            time = events[i_events]
            available = events[i_events][1]
            state.append(time, available)
            i_events += 1
        else:
            time = f_times[i_forecast]
            state.append((time, available))
            i_forecast += 1

    # drain events
    if i_events < len(events):
        state.extend(events[i_events:])
    # drain f_times
    for t in f_times[i_forecast:]:
        state.append((t, available))
