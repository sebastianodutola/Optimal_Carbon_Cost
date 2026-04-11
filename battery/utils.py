from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .models import CarbonTS, Discharge, Load, MergedTimeSeries


class PiecewiseConstant:
    def __init__(self, default: float, change_times, values):
        self.events = list(zip(change_times, values))
        self.default = default
        self._idx = 0
        self.current = default

    def advance(self, t: np.datetime64) -> None:
        if self._idx < len(self.events) and self.events[self._idx][0] == t:
            self.current = self.events[self._idx][1]
            self._idx += 1

    def next_change(self, t_end: np.datetime64) -> np.datetime64:
        if self._idx < len(self.events):
            return self.events[self._idx][0]
        return t_end


def _intervals_to_switch(
    t_intervals: np.ndarray,
    values: np.ndarray,
    default: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (N, 2) interval array to flat switch-event arrays.

    At each interval start the value switches to values[i]; at the end it reverts to default.
    """
    t_switch = t_intervals.flat  # [start0, end0, start1, end1, ...]
    v_switch = np.ones(2 * values.size) * default
    v_switch[::2] = values
    return t_switch, v_switch


def merge(load: Load, carbon: CarbonTS) -> MergedTimeSeries:
    """Line sweep over Load and CarbonTS signals, returning a unified interval timeseries.

    Each row in the output corresponds to a time interval [t_i, t_{i+1}) during which all
    signals (carbon intensity, discharge power, availability) are constant.

    Pass a forecast or actual CarbonTS — the optimisers and stats functions are indifferent.
    """
    carbon_t = carbon.time
    carbon_c = carbon.intensity
    t_end = carbon.t_end

    # Discharge signal: power during discharge intervals, 0 otherwise
    if load.discharges:
        d_intervals = np.array(
            [(d.start_time, d.end_time) for d in load.discharges], dtype=np.datetime64
        )
        d_powers = np.array([d.power for d in load.discharges])
        d_s, d_v = _intervals_to_switch(d_intervals, d_powers, 0.0)
    else:
        d_s, d_v = np.array([]), np.array([])
    d_pw = PiecewiseConstant(0.0, d_s, d_v)

    # Carbon intensity signal
    c_pw = PiecewiseConstant(0.0, carbon_t, carbon_c)

    # Availability signal: 1 during charging windows, 0 outside
    if load.availability is not None:
        a_s, a_v = _intervals_to_switch(
            load.availability, np.ones(load.availability.shape[0]), 0.0
        )
        a_pw = PiecewiseConstant(0.0, a_s, a_v)
    else:
        a_pw = PiecewiseConstant(1.0, np.array([]), np.array([]))

    t_list, delta, d, carbon_cost, available = [], [], [], [], []

    curr_t = carbon_t[0]
    signals = [d_pw, c_pw, a_pw]
    for s in signals:
        s.advance(curr_t)

    while curr_t < t_end:
        next_t = min(s.next_change(t_end) for s in signals)
        delta_t = (next_t - curr_t) / np.timedelta64(1, "h")

        t_list.append(curr_t)
        delta.append(delta_t)
        d.append(d_pw.current)
        carbon_cost.append(c_pw.current)
        available.append(a_pw.current)

        curr_t = next_t
        for s in signals:
            s.advance(curr_t)

    return MergedTimeSeries(
        t=np.array(t_list),
        delta=np.array(delta),
        d=np.array(d),
        carbon_cost=np.array(carbon_cost),
        a=np.array(available),
    )


def recurring_discharge(
    times: list[tuple[time, time, float]],
    start: datetime,
    window: timedelta,
) -> list[Discharge]:
    """Instantiate Discharge objects for all occurrences within [start, start+window).

    Handles the case where start falls mid-discharge (case 1): the discharge is clipped
    to start at `start`.
    """
    end = start + window
    discharges = []
    d = start.date()
    while d <= end.date():
        for t_start, t_end, power in times:
            dt_start = datetime.combine(d, t_start)
            dt_end = datetime.combine(d, t_end)
            if dt_start < start < dt_end:
                discharges.append(
                    Discharge(
                        np.datetime64(start), np.datetime64(min(dt_end, end)), power
                    )
                )
            if start <= dt_start < end:
                discharges.append(
                    Discharge(
                        np.datetime64(dt_start), np.datetime64(min(dt_end, end)), power
                    )
                )
        d += timedelta(days=1)
    return discharges


def recurring_availability(
    times: list[tuple[time, time]],
    start: datetime,
    window: timedelta,
) -> np.ndarray:
    """Return (N, 2) datetime64 array of charging-allowed windows within [start, start+window).

    Handles the case where start falls mid-window (case 1): the window is clipped to start
    at `start`.
    """
    end = start + window
    available = []
    d = start.date()
    while d <= end.date():
        for t_start, t_end in times:
            dt_start = datetime.combine(d, t_start)
            dt_end = datetime.combine(d, t_end)
            if dt_start < start < dt_end:
                available.append(
                    (np.datetime64(start), np.datetime64(min(dt_end, end)))
                )
            if start <= dt_start < end:
                available.append(
                    (np.datetime64(dt_start), np.datetime64(min(dt_end, end)))
                )
        d += timedelta(days=1)
    return np.array(available)


def forecast_range(data_path: Path, start: date, end: date, horizon_type: str):
    """Yield (curr: datetime, forecast_slice: DataFrame) for every 30-min step in [start, end].

    forecast_slice contains only rows where issued_at == curr (UTC), i.e. the forecast
    as it was known at that exact settlement period — suitable for use in an MPC loop.
    Reloads the parquet partition at midnight boundaries.
    """

    if horizon_type not in ("24h", "48h"):
        raise ValueError("Horizon type must be either '24h' or '48h'")
    curr = datetime.combine(start, time.min)
    df = pd.read_parquet(data_path / f"{curr.date()}-{horizon_type}.parquet")
    while curr.date() <= end:
        yield curr, df[df["issued_at"] == pd.Timestamp(curr, tz="UTC")]
        curr += timedelta(minutes=30)
        if curr.time() == time(0, 0) and curr.date() <= end:
            df = pd.read_parquet(data_path / f"{curr.date()}-{horizon_type}.parquet")
