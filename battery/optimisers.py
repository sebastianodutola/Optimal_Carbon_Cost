from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .models import Load, MergedTimeSeries


# ---------------------------------------------------------------------------
# All optimisers share the signature: (x0, load, ts) -> (u, cost)
#
# LP-based: allow concurrent charging and discharging
# Greedy:   no concurrent charging and discharging
#
# State equation: x_{k+1} = x_k + S_eff · u_k · δ_k − d_k · δ_k
# where S_eff = charging_rate · efficiency, u_k ∈ [0, a_k], a_k ∈ {0, 1}
# ---------------------------------------------------------------------------

def lp_optimal(x0: float, load: Load, ts: MergedTimeSeries) -> tuple[np.ndarray, float]:
    """Minimise carbon cost subject to battery state constraints via linear programming.

    Returns (u, cost) where u[i] ∈ [0, 1] is the charging fraction during interval i.
    """
    S = load.charging_rate * load.efficiency
    C = load.capacity
    delta, d, carbon_intensity, a = ts.delta, ts.d, ts.carbon_cost, ts.a

    if delta.size == 0:
        return np.array([]), 0.0

    n = delta.size
    Delta = np.tri(n, dtype=delta.dtype) * delta  # lower-triangular cumulative time matrix

    # Constraints: 0 ≤ x_k ≤ C for all k, derived from cumulative state equation
    A = np.zeros((2 * n, n), dtype=delta.dtype)
    A[:n, :] = Delta
    A[n:, :] = -Delta

    b = np.zeros(2 * n, dtype=delta.dtype)
    b[:n] = (C - x0) * np.ones(n) + Delta @ d
    b[n:] = x0 * np.ones(n) - Delta @ d
    b /= S

    c = delta * carbon_intensity
    bounds = np.zeros((n, 2))
    bounds[:, 1] = a  # u_i ∈ [0, a_i]

    res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds)

    if not res.success:
        _lp_error(res.status)
        return None

    return res.x, float(res.fun)


def lp_naive(x0: float, load: Load, ts: MergedTimeSeries) -> tuple[np.ndarray, float]:
    """Greedy charge-to-capacity (allows concurrent charging/discharging).

    At each available interval: charge at maximum rate unless that would overshoot capacity,
    in which case charge at the rate that exactly reaches C at t_{k+1}.
    """
    S = load.charging_rate * load.efficiency
    C = load.capacity
    delta, d, carbon_intensity, a = ts.delta, ts.d, ts.carbon_cost, ts.a

    if delta.size == 0:
        return np.array([]), 0.0

    n = delta.size
    u = np.zeros(n)
    x = x0

    for i in range(n):
        if a[i] > 0:
            x_at_max = x + S * a[i] * delta[i] - d[i] * delta[i]
            if x_at_max < C:
                u[i] = a[i]
            else:
                # Exact rate to reach C at t_{i+1}; guaranteed ≤ a[i]
                u[i] = max(0.0, (C - x + d[i] * delta[i]) / (S * delta[i]))
        x = x + S * u[i] * delta[i] - d[i] * delta[i]

    cost = float(np.dot(delta * carbon_intensity, u))
    return u, cost


def greedy_optimal(x0: float, load: Load, ts: MergedTimeSeries) -> tuple[np.ndarray, float]:
    """Fill required energy per discharge by preferring lowest-carbon available slots.

    Slots are sorted globally by carbon intensity and allocated greedily per discharge event
    in chronological order. No concurrent charging and discharging.
    """
    S = load.charging_rate * load.efficiency
    n = ts.delta.size
    u = np.zeros(n)
    remaining = np.ones(n)  # fraction of each slot still unallocated

    # Eligible: available and not discharging
    eligible = np.where((ts.a > 0) & (ts.d == 0))[0]
    sorted_by_carbon = eligible[np.argsort(ts.carbon_cost[eligible])]

    stored_E = x0

    for discharge in sorted(load.discharges):
        req_E = discharge.power * (discharge.end_time - discharge.start_time) / np.timedelta64(1, "h")

        for idx in sorted_by_carbon:
            if stored_E >= req_E:
                break
            if ts.t[idx] >= discharge.start_time or remaining[idx] <= 0:
                continue

            slot_E = S * remaining[idx] * ts.delta[idx]
            needed = req_E - stored_E

            if slot_E <= needed:
                u[idx] = remaining[idx]
                remaining[idx] = 0.0
                stored_E += slot_E
            else:
                frac = needed / slot_E
                u[idx] += frac * remaining[idx]
                remaining[idx] *= (1.0 - frac)
                stored_E = req_E

        if stored_E < req_E:
            print(f"Undercharged at {discharge.start_time} by {req_E - stored_E:.2f} kWh")

        stored_E = max(0.0, stored_E - req_E)

    cost = float(np.dot(ts.delta * ts.carbon_cost, u))
    return u, cost


def greedy_naive(x0: float, load: Load, ts: MergedTimeSeries) -> tuple[np.ndarray, float]:
    """Charge to capacity as early as possible. No concurrent charging and discharging."""
    S = load.charging_rate * load.efficiency
    C = load.capacity
    n = ts.delta.size
    u = np.zeros(n)
    x = x0

    for i in range(n):
        if ts.a[i] > 0 and ts.d[i] == 0:
            x_at_max = x + S * ts.delta[i]
            if x_at_max < C:
                u[i] = 1.0
            else:
                u[i] = max(0.0, (C - x) / (S * ts.delta[i]))
        x = x + S * u[i] * ts.delta[i] - ts.d[i] * ts.delta[i]

    cost = float(np.dot(ts.delta * ts.carbon_cost, u))
    return u, cost


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lp_error(status: int) -> None:
    messages = {
        1: "iteration limit reached",
        2: "infeasible: verify the discharge series is feasible",
        3: "problem is unbounded",
        4: "numerical difficulties encountered",
    }
    print(f"LP failed: {messages.get(status, f'unknown status {status}')}")
