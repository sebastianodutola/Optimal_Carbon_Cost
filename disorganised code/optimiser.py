from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class ChargeSlot:
    start_time: np.datetime64
    duration: np.timedelta64
    carbon_intensity: float  # gCO_2eq/kWh
    power_drawn: float


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
    initial_charge_percent: float = 0  # (0,1] %


@dataclass
class Forecast:
    time: np.ndarray[np.datetime64]
    settlement_interval: np.timedelta64
    # forecast carbon_intensity gCO_2eq/kWh
    f_carbon_intensity: np.ndarray[float]
    # actual carbon intensity if it exists
    a_carbon_intensity: np.ndarray[float]

    def __len__(self):
        return self.time.shape[0]


def parquet_to_forecast(
    data_path: str, settlement_interval=np.timedelta64(30, "m")
) -> Forecast:
    table = pq.read_table(data_path)
    df = table.to_pandas()
    time = df["from"].apply(np.datetime64).to_numpy()
    f_carbon_intensity = df["intensity"].apply(lambda i: i["forecast"]).to_numpy()
    a_carbon_intensity = df["intensity"].apply(lambda i: i["actual"]).to_numpy()
    return Forecast(time, settlement_interval, f_carbon_intensity, a_carbon_intensity)


def pandas_to_forecast(
    df: pd.DataFrame, settlement_interval=np.timedelta64(30, "m")
) -> Forecast:
    time = df["from"].apply(np.datetime64).to_numpy()
    f_carbon_intensity = df["forecast"].to_numpy()
    a_carbon_intensity = df["actual"].to_numpy()
    return Forecast(time, settlement_interval, f_carbon_intensity, a_carbon_intensity)


def state_util(
    load: Load,
    f_times: np.ndarray[np.datetime64],
    f_carbon_intensity: np.ndarray[float],
) -> tuple[list[tuple]]:
    "Takes load and forecast as input and outputs a state vector and a discharge vector used by the optimiser."
    i_events = 0
    i_forecast = 0
    sorted_discharges = sorted(load.discharges)

    transformed_discharges = [
        (
            d.start_time,
            d.end_time,
            d.power
            * (d.end_time - d.start_time).astype("timedelta64[s]").astype("int64")
            / 3600,
        )
        for d in sorted_discharges
    ]

    # Check to see if required energy doesn't exceed load capacity
    max_required = max(transformed_discharges, key=lambda x: x[1])[2]
    if max_required > load.capacity:
        raise ValueError(
            f"Required energy {max_required:.2f} kWh exceeds usable capacity {load.capacity:.2f} kWh."
        )

    d_events = [
        (t, is_start)
        for d in sorted_discharges
        for t, is_start in ((d.start_time, False), (d.end_time, True))
    ]

    available = True
    state = []

    while i_events < len(d_events) and i_forecast < len(f_times):
        if d_events[i_events][0] < f_times[i_forecast]:
            time = d_events[i_events][0]
            # update energy available in state
            if state:
                state[-1][1] = time - state[-1][0]
            available = d_events[i_events][1]
            state.append([time, 0, available, f_carbon_intensity[i_forecast]])
            i_events += 1
        else:
            time = f_times[i_forecast]
            if state:
                state[-1][1] = time - state[-1][0]
            state.append([time, 0, available, f_carbon_intensity[i_forecast]])
            i_forecast += 1

    # drain events
    while i_events < len(d_events):
        time = d_events[i_events][0]
        if state:
            state[-1][1] = time - state[-1][0]
        available = d_events[i_events][1]
        state.append([time, 0, available, f_carbon_intensity[-1]])
        i_events += 1

    # drain f_times
    while i_forecast < len(f_times):
        dt = f_times[1] - f_times[0]
        time = f_times[i_forecast]
        state.append([time, dt, available, f_carbon_intensity[i_forecast]])
        i_forecast += 1

    return transformed_discharges, state


def optimiser(
    load: Load,
    f_times: np.ndarray[np.datetime64],
    f_carbon_intensity: np.ndarray[float],
) -> list[ChargeSlot]:
    transformed_discharges, state = state_util(load, f_times, f_carbon_intensity)
    effective_charging_rate = load.efficiency * load.charging_rate  # kW
    # optimisation logic
    slots = sorted(state, key=lambda x: x[3])
    # charging (time, duration, carbon_intensity)
    charging = []
    stored_E = load.initial_charge_percent * load.capacity
    for d_start, _, req_E in transformed_discharges:
        for slot in slots:
            # Sufficiently Charged
            if stored_E > req_E:
                break

            t, dt, available, c = slot
            if t > d_start or not available:
                continue
            dt_s = dt.astype("timedelta64[s]").astype("int64")
            dE = dt_s * effective_charging_rate / 3600

            # Partial fill
            if stored_E + dE > req_E:
                partial_dt = np.timedelta64(int(dt_s * (req_E - stored_E) / dE), "s")
                charging.append(ChargeSlot(t, partial_dt, c, load.charging_rate))
                slot[0] = t + partial_dt
                slot[1] = dt - partial_dt
                # Sufficiently Charged
                stored_E = req_E
                break

            slot[2] = False
            stored_E += dE
            charging.append(ChargeSlot(t, dt, c, load.charging_rate))

        if stored_E < req_E:
            print(f"Undercharged at {d_start} by {req_E - stored_E:.2f} kWh")

        stored_E = max(0.0, stored_E - req_E)

    return charging


def naive_charge(
    load: Load,
    f_times: np.ndarray[np.datetime64],
    f_carbon_intensity: np.ndarray[float],
) -> list[ChargeSlot]:
    transformed_discharges, state = state_util(load, f_times, f_carbon_intensity)
    effective_charging_rate = load.efficiency * load.charging_rate
    req_E_q = deque([d[2] for d in transformed_discharges])
    req_E = req_E_q.leftpop()
    stored_E = load.initial_charge_percent * load.capacity
    charging = []
    in_discharge = False
    for slot in state:
        t, dt, available, c = slot
        if not available:
            if not in_discharge:
                if stored_E < req_E:
                    print(f"Undercharged at {t} by {req_E - stored_E:.2f} kWh")
                stored_E = max(0, stored_E - req_E)
                if req_E_q:
                    req_E = req_E_q.pop(0)
            in_discharge = True
            continue
        in_discharge = False
        if stored_E >= load.capacity:
            continue
        dt_s = dt.astype("timedelta64[s]").astype("int64")
        dE = dt_s * effective_charging_rate / 3600
        if stored_E + dE > load.capacity:
            partial_dt = np.timedelta64(
                int(dt_s * (load.capacity - stored_E) / dE), "s"
            )
            charging.append(ChargeSlot(t, partial_dt, c, load.charging_rate))
            stored_E = load.capacity
        else:
            stored_E += dE
            charging.append(ChargeSlot(t, dt, c, load.charging_rate))

    return charging


def realised_charging_stats(
    charging: list[ChargeSlot],
    times: np.ndarray[np.datetime64],
    carbon_intensity: np.ndarray[float],
):
    dt = times[1:] - times[:-1]
    assert (dt == dt[0]).all()
    dt = dt[0]

    carbon_cost = 0
    max_carbon_intensity = 0
    total_energy_drawn = 0

    for slot in charging:
        t = slot.start_time
        d = slot.duration
        # [t, t+dt) should be within a settlement period.
        # Calculate the settlement time index before t
        idx = ((t - times[0]) // np.timedelta64(30, "m")).astype(int)
        # check the inclusion
        assert t >= times[idx]
        if idx + 1 < len(times):
            assert t + d <= times[idx + 1]

        s = d / np.timedelta64(1, "s")
        energy_drawn = slot.power_drawn * s
        carbon_cost += carbon_intensity[idx] * energy_drawn
        max_carbon_intensity = max(max_carbon_intensity, carbon_intensity[idx])
        total_energy_drawn += energy_drawn
    return {
        "carbon cost": carbon_cost,
        "max carbon intensity": max_carbon_intensity,
        "total energy drawn": total_energy_drawn,
    }


def forecast_charging_stats(charging: list[ChargeSlot]) -> dict:
    carbon_cost = 0
    max_carbon_intensity = 0
    total_energy_drawn = 0
    for slot in charging:
        s = slot.duration / np.timedelta64(1, "s")
        energy_drawn = s * slot.power_drawn
        carbon_cost += slot.carbon_intensity * energy_drawn
        max_carbon_intensity = max(max_carbon_intensity, slot.carbon_intensity)
        total_energy_drawn += slot.power_drawn * s
    return {
        "carbon cost": carbon_cost,
        "max carbon intensity": max_carbon_intensity,
        "total energy drawn": total_energy_drawn,
    }


if __name__ == "__main__":
    DATAPATH = Path(__file__).parent.joinpath("intensity.parquet")
    forecast = parquet_to_forecast(DATAPATH)
    # Define EV Load
    # Most EVs have very high charging efficiency
    efficiency = 0.95
    capacity = 55
    charging_rate = 5  # somewhere in between fast chargers and home chargers

    # EV driving efficiency is often about 0.3 kWh/mile, for the purposes of demonstration we assume 30 mph for use time
    power = 9
    # We use standard driving times for a reasonable workin' jane
    morning_drive_start, morning_drive_end = np.datetime64(
        "2026-02-27T07:30"
    ), np.datetime64("2026-02-27T08:30")
    evening_drive_start, evening_drive_end = np.datetime64(
        "2026-02-27T17:30"
    ), np.datetime64("2026-02-27T18:30")

    morning_discharge1 = Discharge(morning_drive_start, morning_drive_end, power)
    evening_discharge1 = Discharge(evening_drive_start, evening_drive_end, power)
    morning_discharge2 = Discharge(
        morning_drive_start + np.timedelta64(1, "D"),
        morning_drive_end + np.timedelta64(1, "D"),
        power,
    )
    evening_discharge2 = Discharge(
        evening_drive_start + np.timedelta64(1, "D"),
        evening_drive_end + np.timedelta64(1, "D"),
        power,
    )
    discharges = np.array(
        [morning_discharge1, evening_discharge1, morning_discharge2, evening_discharge2]
    )

    EV = Load(capacity, charging_rate, discharges, efficiency, 0.0)
    print(EV.discharges[0].start_time)

    optimal_charging_times = optimiser(EV, forecast.time, forecast.f_carbon_intensity)
    optimal_charging_times_clairvoyant = optimiser(
        EV, forecast.time, forecast.a_carbon_intensity
    )

    # quick comparison between clairvoyant and optimal forecast charging
    fcs = realised_charging_stats(
        optimal_charging_times, forecast.time, forecast.a_carbon_intensity
    )
    ccs = forecast_charging_stats(optimal_charging_times_clairvoyant)
    carbon_regret = fcs["carbon cost"] / ccs["carbon cost"] - 1
    print(carbon_regret)

    # Cursory plots
    fx = forecast.time
    fy = forecast.f_carbon_intensity
    plt.plot(fx, fy)
    for charge in optimal_charging_times:
        x1 = charge.start_time
        x2 = charge.start_time + charge.duration
        plt.fill_betweenx([0, 250], x1, x2, alpha=0.3, color="red", linewidth=0)

    plt.show()
