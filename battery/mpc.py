from datetime import datetime, time, date, timedelta
from pathlib import Path
from typing import Callable
import pandas as pd
import numpy as np
from .models import Load, CarbonTS
from .optimisers import lp_naive, lp_optimal
from .utils import forecast_range, recurring_availability, recurring_discharge, merge
import matplotlib.pyplot as plt


def MPC(
    data_path: Path,
    start: date,
    end: date,
    optimiser: Callable,
    schedule: list[tuple[time, time, float]],  # daily discharge periods
    availability: list[tuple[time, time]],  # daily availability periods
    capacity: float,
    charging_rate: float,
    efficiency: float = 1.0,
    initial_charge: float = 0.0,
    intensity_col: str = "forecast",
    horizon_type: str = "24h",
) -> pd.DataFrame:
    x = initial_charge
    S = charging_rate * efficiency
    SI = np.timedelta64(30, "m")
    if horizon_type == "48h":
        forecast_length = timedelta(hours=48)
    elif horizon_type == "24h":
        forecast_length = timedelta(hours=24)
    else:
        raise ValueError("horizon_type must be '24h' or '48h'")

    load = Load(
        capacity=capacity,
        charging_rate=charging_rate,
        discharges=[],
        efficiency=efficiency,
        initial_charge=x,
    )

    history = []

    for curr, f in forecast_range(data_path, start, end, horizon_type):
        curr_ts = pd.Timestamp(curr, tz="UTC")
        next_ts = curr_ts + pd.Timedelta("30min")

        load.discharges = recurring_discharge(schedule, curr, forecast_length)
        load.availability = recurring_availability(availability, curr, forecast_length)
        load.initial_charge = x

        fwd = f[f["period_start"] >= curr_ts]
        carbon_ts = CarbonTS(
            time=fwd["period_start"]
            .dt.tz_localize(None)
            .to_numpy()
            .astype("datetime64[m]"),
            settlement_interval=SI,
            intensity=fwd[intensity_col].to_numpy(dtype=float),
        )

        ts = merge(load, carbon_ts)
        result = optimiser(x, load, ts)
        u = result[0] if result is not None else np.zeros(ts.delta.size)

        step_mask = ts.t < next_ts.to_datetime64()
        for i in np.where(step_mask)[0]:
            dx = S * u[i] * ts.delta[i] - ts.d[i] * ts.delta[i]
            x = float(np.clip(x + dx, 0.0, capacity))
            energy_drawn = charging_rate * u[i] * ts.delta[i]
            _append_to_history(ts.t[i], x, u[i], energy_drawn, history)

    return pd.DataFrame(history)


def _append_to_history(
    t: np.datetime64, x: float, u: float, energy_drawn: float, history: list
):
    history.append(
        {
            "t": t,
            "charge": x,
            "u": u,
            "energy_drawn": energy_drawn,
        }
    )


def clairvoyant_charging_schedule(
    actual_carbon_ts: CarbonTS,
    start: date,
    end: date,
    optimiser: Callable,
    schedule: list[tuple[time, time, float]],
    availability: list[tuple[time, time]],
    capacity: float,
    charging_rate: float,
    efficiency: float,
    initial_charge: float,
):
    x = initial_charge
    SI = np.timedelta64(30, "m")
    S = efficiency * charging_rate

    start_dt64 = np.datetime64(datetime.combine(start, time.min), "m")
    end_dt64 = np.datetime64(datetime.combine(end, time.max), "m")
    mask = (actual_carbon_ts.time >= start_dt64) & (actual_carbon_ts.time <= end_dt64)
    actual_carbon_ts = CarbonTS(
        time=actual_carbon_ts.time[mask],
        settlement_interval=actual_carbon_ts.settlement_interval,
        intensity=actual_carbon_ts.intensity[mask],
    )

    load = Load(
        capacity=capacity,
        charging_rate=charging_rate,
        discharges=[],
        efficiency=efficiency,
        initial_charge=initial_charge,
    )
    end_dt = datetime.combine(end, time.max)
    start_dt = datetime.combine(start, time.min)
    load.availability = recurring_availability(
        times=availability, start=start_dt, window=end_dt - start_dt
    )
    load.discharges = recurring_discharge(
        times=schedule, start=start_dt, window=end_dt - start_dt
    )
    ts = merge(load, actual_carbon_ts)
    result = optimiser(x, load, ts)
    u = result[0] if result is not None else np.zeros(ts.delta.size)
    history = []
    for t, delta, d_i, u_i in zip(ts.t, ts.delta, ts.d, u):
        x = float(np.clip(x + S * u_i * delta - d_i * delta, 0.0, capacity))
        energy_drawn = u_i * charging_rate * delta
        history.append(
            {
                "t": t,
                "charge": x,
                "u": u_i,
                "energy_drawn": energy_drawn,
            }
        )
    return pd.DataFrame(history)


def run_comparative_simulation(
    data_path: Path,
    output_path: Path,
    output_file_name: str,
    actual_carbon_ts: CarbonTS,
    start: date,
    end: date,
    schedule: list[tuple[time, time, float]],
    availability: list[tuple[time, time]],
    capacity: float,
    charging_rate: float,
    efficiency: float,
    initial_charge: float,
    horizon_type: str = "48h",
) -> pd.DataFrame:
    shared = dict(
        start=start,
        end=end,
        schedule=schedule,
        availability=availability,
        capacity=capacity,
        charging_rate=charging_rate,
        efficiency=efficiency,
        initial_charge=initial_charge,
    )

    optimal = MPC(
        data_path=data_path, **shared, optimiser=lp_optimal, horizon_type=horizon_type
    )
    naive = MPC(
        data_path=data_path, **shared, optimiser=lp_naive, horizon_type=horizon_type
    )
    clairvoyant_mpc = MPC(
        data_path=data_path,
        **shared,
        optimiser=lp_optimal,
        horizon_type=horizon_type,
        intensity_col="actual",
    )
    clairvoyant = clairvoyant_charging_schedule(
        actual_carbon_ts=actual_carbon_ts, **shared, optimiser=lp_optimal
    )

    optimal = optimal.rename(
        columns={
            "charge": "charge_optimal",
            "u": "u_optimal",
            "energy_drawn": "energy_drawn_optimal",
        }
    )
    naive = naive.rename(
        columns={
            "charge": "charge_naive",
            "u": "u_naive",
            "energy_drawn": "energy_drawn_naive",
        }
    )
    clairvoyant_mpc = clairvoyant_mpc.rename(
        columns={
            "charge": "charge_clair_mpc",
            "u": "u_clair_mpc",
            "energy_drawn": "energy_drawn_clair_mpc",
        }
    )
    clairvoyant = clairvoyant.rename(
        columns={
            "charge": "charge_clairvoyant",
            "u": "u_clairvoyant",
            "energy_drawn": "energy_drawn_clairvoyant",
        }
    )
    df_actual_carbon = pd.DataFrame(
        {"t": actual_carbon_ts.time, "actual_intensity": actual_carbon_ts.intensity}
    )
    df = optimal.merge(naive, "left", on="t")
    df = df.merge(clairvoyant, "left", on="t")
    df = df.merge(clairvoyant_mpc, "left", on="t")
    df = df.merge(df_actual_carbon, "left", on="t")
    df["actual_intensity"] = df["actual_intensity"].ffill()

    for name in ["optimal", "naive", "clair_mpc", "clairvoyant"]:
        df[f"carbon_cost_{name}"] = df[f"energy_drawn_{name}"] * df["actual_intensity"]

    df.to_parquet(output_path / output_file_name)
    return df


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent.parent / "Data" / "daily"
    schedule = [(time(10, 0), time(11, 15), 10)]
    availability = [(time(0, 0), time(23, 30))]
    history = MPC(
        data_path=DATA_PATH,
        start=date(2025, 2, 1),
        end=date(2025, 2, 2),
        optimiser=lp_optimal,
        schedule=schedule,
        availablility=availability,
        capacity=100,
        charging_rate=10,
    )
