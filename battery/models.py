from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(order=True)
class Discharge:
    start_time: np.datetime64
    end_time: np.datetime64
    power: float  # kW


@dataclass
class Load:
    capacity: float                    # kWh
    charging_rate: float               # kW
    discharges: list[Discharge]
    availability: np.ndarray | None = None  # (N, 2) datetime64; windows when charging is possible
                                            # None = always available
    efficiency: float = 1.0
    initial_charge: float = 0.0        # kWh


@dataclass
class CarbonTS:
    """A single carbon intensity timeseries — either forecast or actual.

    Keeping forecast and actual as separate CarbonTS instances means every
    function in the pipeline is indifferent to which one it receives, making
    EVPI comparisons straightforward at the call site.
    """
    time: np.ndarray                # datetime64 settlement period start times
    settlement_interval: np.timedelta64
    intensity: np.ndarray           # gCO2eq/kWh

    def __len__(self) -> int:
        return self.time.shape[0]

    @property
    def t_end(self) -> np.datetime64:
        return self.time[-1] + self.settlement_interval


@dataclass
class MergedTimeSeries:
    t: np.ndarray            # datetime64 start time of each interval
    delta: np.ndarray        # hours per interval
    d: np.ndarray            # discharge power (kW) during each interval
    carbon_cost: np.ndarray  # gCO2eq/kWh during each interval
    a: np.ndarray            # availability: 1.0 if charging possible, 0.0 otherwise
