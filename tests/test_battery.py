"""
Unit tests and visualisation for the battery optimisers.

Load fixture design
-------------------
load_clean     — availability (00:00–07:00, 19:00–24:00) is disjoint from discharge
                 (07:30–08:30, 17:30–18:30).  All four algorithms are valid here.
                 With a single carbon signal, greedy-by-carbon is provably LP-optimal
                 for independent discharges drawn from the same slot pool, so
                 lp_optimal and greedy_optimal must agree on cost.

load_concurrent — availability (06:00–19:00) overlaps both discharge windows.
                  Only the LP algorithms are meaningful (they can exploit the overlap);
                  greedy silently skips concurrent slots, so only lp_optimal and
                  lp_naive are tested here.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe in CI and headless envs

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest

from battery import (
    CarbonTS,
    Discharge,
    Load,
    MergedTimeSeries,
    greedy_naive,
    greedy_optimal,
    lp_naive,
    lp_optimal,
    merge,
    stats,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

DATE = np.datetime64("2024-01-15T00:00")
SI = np.timedelta64(30, "m")  # settlement interval


def _t(hour: float) -> np.datetime64:
    """Absolute datetime64 for a given decimal hour offset from DATE."""
    return DATE + np.timedelta64(int(hour * 60), "m")


def _battery_state(x0: float, load: Load, ts: MergedTimeSeries, u: np.ndarray):
    """Reconstruct (t_axis, state) arrays for plotting.

    t_axis has length n+1 (interval boundaries); state likewise.
    """
    S = load.charging_rate * load.efficiency
    delta_td = (ts.delta * 3.6e12).astype("timedelta64[ns]")
    t_axis = np.empty(ts.delta.size + 1, dtype=ts.t.dtype)
    t_axis[:-1] = ts.t
    t_axis[-1] = ts.t[-1] + delta_td[-1]

    state = np.empty(ts.delta.size + 1)
    state[0] = x0
    for i in range(ts.delta.size):
        state[i + 1] = np.clip(
            state[i] + S * u[i] * ts.delta[i] - ts.d[i] * ts.delta[i],
            0.0,
            load.capacity,
        )
    return t_axis, state


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def carbon() -> CarbonTS:
    """48 half-hour periods with a sinusoidal carbon profile (low overnight, peak ~14:00)."""
    n = 48
    times = DATE + np.arange(n) * SI
    hours = np.arange(n) * 0.5
    intensity = 150.0 + 100.0 * np.sin(2 * np.pi * (hours - 14.0) / 24.0)
    return CarbonTS(times, SI, intensity)


@pytest.fixture
def load_clean() -> Load:
    """EV with overnight-only availability; no overlap with discharge windows."""
    discharges = [
        Discharge(_t(7.5), _t(8.5), 5.0),  # morning commute  (5 kWh)
        Discharge(_t(17.5), _t(18.5), 5.0),  # evening commute  (5 kWh)
    ]
    availability = np.array(
        [
            (_t(0.0), _t(7.0)),  # 00:00–07:00  (before morning departure)
            (_t(19.0), _t(24.0)),  # 19:00–24:00  (after evening return)
        ],
        dtype="datetime64",
    )
    return Load(
        capacity=40.0,
        charging_rate=7.0,
        discharges=discharges,
        availability=availability,
        efficiency=0.95,
        initial_charge=0.0,
    )


@pytest.fixture
def load_concurrent() -> Load:
    """EV plugged in at a workplace charger 06:00–19:00; availability overlaps commutes."""
    discharges = [
        Discharge(_t(7.5), _t(8.5), 5.0),
        Discharge(_t(17.5), _t(18.5), 5.0),
    ]
    availability = np.array(
        [
            (_t(6.0), _t(19.0)),  # 06:00–19:00  (overlaps both discharge windows)
        ],
        dtype="datetime64",
    )
    return Load(
        capacity=40.0,
        charging_rate=7.0,
        discharges=discharges,
        availability=availability,
        efficiency=0.95,
        initial_charge=0.0,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_all_four_optimisers_coincide(carbon, load_clean):
    """
    With disjoint availability and discharge windows, greedy-by-carbon is provably
    LP-optimal: both discharges draw from the same overnight pool, so sorting by
    carbon intensity is globally optimal.  lp_optimal and greedy_optimal must agree
    on cost within floating-point tolerance.
    """
    x0 = load_clean.initial_charge
    ts = merge(load_clean, carbon)

    _, cost_lp_opt = lp_optimal(x0, load_clean, ts)
    _, cost_lp_naive = lp_naive(x0, load_clean, ts)
    _, cost_g_opt = greedy_optimal(x0, load_clean, ts)
    _, cost_g_naive = greedy_naive(x0, load_clean, ts)

    # The two optimal algorithms must agree
    assert (
        abs(cost_lp_opt - cost_g_opt) < 1e-2
    ), f"lp_optimal ({cost_lp_opt:.4f}) ≠ greedy_optimal ({cost_g_opt:.4f})"

    # Naive variants must cost at least as much as their optimal counterparts
    assert cost_lp_naive >= cost_lp_opt - 1e-6
    assert cost_g_naive >= cost_g_opt - 1e-6


def test_concurrent_lp_algorithms(carbon, load_concurrent):
    """
    With overlapping availability and discharge windows, only LP algorithms are
    applicable.  Verify they complete without error and that lp_optimal ≤ lp_naive.
    """
    x0 = load_concurrent.initial_charge
    ts = merge(load_concurrent, carbon)

    result_opt = lp_optimal(x0, load_concurrent, ts)
    result_naive = lp_naive(x0, load_concurrent, ts)

    assert result_opt is not None, "lp_optimal failed to converge"
    assert result_naive is not None, "lp_naive failed to converge"

    _, cost_opt = result_opt
    _, cost_naive = result_naive

    assert (
        cost_opt <= cost_naive + 1e-6
    ), f"lp_optimal ({cost_opt:.4f}) should be ≤ lp_naive ({cost_naive:.4f})"


# ── Visualisation ──────────────────────────────────────────────────────────────


def test_visualisation(carbon, load_concurrent, tmp_path):
    """
    Produce a two-panel figure of battery state vs time for lp_optimal and lp_naive
    on load_concurrent.  Charging periods are highlighted in green, discharge windows
    in red.  Saved to tmp_path/battery_state.png.
    """
    load = load_concurrent
    x0 = load.initial_charge
    ts = merge(load, carbon)

    u_opt, _ = lp_optimal(x0, load, ts)
    u_naive, _ = lp_naive(x0, load, ts)

    delta_td = (ts.delta * 3.6e12).astype("timedelta64[ns]")
    t_ends = (ts.t + delta_td).astype("datetime64[ms]").astype("O")
    t_starts = ts.t.astype("datetime64[ms]").astype("O")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Carbon intensity on the merged interval start times
    c_t = ts.t.astype("datetime64[ms]").astype("O")

    for ax, u, title in [
        (axes[0], u_opt, "LP Optimal"),
        (axes[1], u_naive, "LP Naive"),
    ]:
        t_axis, state = _battery_state(x0, load, ts, u)
        t_plot = t_axis.astype("datetime64[ms]").astype("O")

        ax.plot(t_plot, state, color="steelblue", linewidth=2)
        ax.axhline(
            load.capacity,
            color="grey",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

        # Charging intervals (green)
        for i in range(ts.delta.size):
            if u[i] > 1e-6:
                ax.axvspan(t_starts[i], t_ends[i], alpha=0.25, color="green")

        # Discharge intervals (red)
        for d in load.discharges:
            ax.axvspan(
                d.start_time.astype("datetime64[ms]").astype("O"),
                d.end_time.astype("datetime64[ms]").astype("O"),
                alpha=0.25,
                color="red",
            )

        ax.set_ylabel("State of charge (kWh)")
        ax.set_title(title)
        ax.set_ylim(0, load.capacity * 1.1)

        # Carbon intensity on a right-hand axis
        ax2 = ax.twinx()
        ax2.step(c_t, ts.carbon_cost, where="post", color="darkorange", linewidth=1)
        ax2.set_ylabel("Carbon intensity (gCO₂eq/kWh)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax2.set_ylim(0, ts.carbon_cost.max() * 1.4)

        legend_elements = [
            plt.Line2D([0], [0], color="steelblue", linewidth=2, label="Battery state"),
            plt.Line2D(
                [0], [0], color="darkorange", linewidth=1, label="Carbon intensity"
            ),
            mpatches.Patch(facecolor="green", alpha=0.5, label="Charging"),
            mpatches.Patch(facecolor="red", alpha=0.5, label="Discharge"),
            plt.Line2D([0], [0], color="grey", linestyle="--", label="Capacity"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    axes[-1].set_xlabel("Time of day")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate()
    fig.suptitle("EV Battery State — 2024-01-15", fontsize=13)
    plt.tight_layout()

    out = tmp_path / "battery_state.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    assert out.exists()

    print(out)
