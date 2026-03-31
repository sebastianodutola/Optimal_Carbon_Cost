from __future__ import annotations

import numpy as np

from .models import CarbonTS, Load, MergedTimeSeries


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
