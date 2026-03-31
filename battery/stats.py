from __future__ import annotations

import numpy as np

from .models import Load, MergedTimeSeries


def stats(ts: MergedTimeSeries, u: np.ndarray, load: Load) -> dict:
    """Carbon cost and energy for a given control signal against ts.carbon_cost.

    Pass a MergedTimeSeries built from a forecast CarbonTS for forecast stats,
    or from an actual CarbonTS for realised stats. The function is indifferent.

    Example — EVPI:
        ts_f = merge(load, forecast)
        ts_a = merge(load, actual)
        u, _ = lp_optimal(x0, load, ts_f)
        forecast_cost = stats(ts_f, u, load)["carbon_cost"]
        realised_cost = stats(ts_a, u, load)["carbon_cost"]
    """
    S = load.charging_rate * load.efficiency
    energy = S * u * ts.delta          # kWh per interval
    carbon_cost = float(np.dot(ts.carbon_cost, energy))
    charging_mask = u > 0
    return {
        "carbon_cost": carbon_cost,
        "energy_charged": float(energy.sum()),
        "peak_carbon_intensity": float(ts.carbon_cost[charging_mask].max()) if charging_mask.any() else 0.0,
    }
