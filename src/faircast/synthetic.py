"""Planted panels with known ground truth, for validation and demos.

These generators exist so FairCast's claims can be checked against data
where the truth is fixed in advance — the validation protocol's rule
that a forecaster must *demonstrate* skill on planted structure before
its numbers on real data are quoted, and must stay silent on panels
planted to have no exploitable structure. All panels are tidy counts
frames (period, group, n, positives) for two groups, "A" and "B"; the
forecast metric is the demographic-parity gap between them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _frame(latent: dict[str, np.ndarray], ns: dict[str, np.ndarray],
           rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    T = len(next(iter(latent.values())))
    for g, rates in latent.items():
        for t in range(T):
            n = int(ns[g][t])
            rows.append({"period": t, "group": g, "n": n,
                         "positives": int(rng.binomial(n, rates[t]))})
    return pd.DataFrame(rows)


def _ns(T: int, n_range, rng) -> dict[str, np.ndarray]:
    return {g: rng.integers(n_range[0], n_range[1], size=T)
            for g in ("A", "B")}


def planted_breach_panels(n_series: int = 12, n_periods: int = 10,
                          tau: float = 0.10, seed: int = 7,
                          n_range=(4000, 8000)):
    """Panels whose latent DP gap crosses ``tau`` at a planted period.

    Group A's rate is ~0.55 and steady; group B's rate declines on a
    straight line so the latent gap crosses tau at a per-series planted
    index in the second half of the window. Returns (frames, breaches)
    where breaches[i] is the first period index whose *latent* gap
    reaches tau.

    The default cell sizes are population-scale (thousands per
    group-period, as in the ACS population trajectories of Paper 11):
    at batch sizes of a few hundred per cell, binomial noise (gap SE
    ~0.02) swamps a per-step latent slope of ~0.016 and *no* forecaster
    can see the trend — that regime is covered separately by
    :func:`tiny_n_panel`, where the honest requirement is wide
    intervals, not skill. Pass a smaller ``n_range`` to generate the
    noisy regime explicitly.
    """
    rng = np.random.default_rng(seed)
    frames, breaches = [], []
    for _ in range(n_series):
        T = n_periods
        a = 0.55 + rng.normal(0, 0.01, size=T)
        a = np.clip(a, 0.05, 0.95)
        gap0 = rng.uniform(0.01, 0.04)
        b_idx = int(rng.integers(T // 2, T - 1))
        gap_end = tau + rng.uniform(0.04, 0.08)
        gaps = np.linspace(gap0, gap_end, T)
        b = np.clip(a - gaps, 0.02, 0.98)
        latent_gap = a - b
        breaches.append(int(np.argmax(latent_gap >= tau)))
        frames.append(_frame({"A": a, "B": b}, _ns(T, n_range, rng), rng))
    return frames, breaches


def never_breach_panels(n_series: int = 12, n_periods: int = 10,
                        tau: float = 0.10, seed: int = 11,
                        n_range=(800, 2000)):
    """Panels that drift *toward* tau but asymptote at 0.8 * tau.

    The near-miss negative: a forecaster that reads any drift as a
    coming breach will burn its false-alarm budget here. Returns
    (frames, [None] * n_series) — no latent breach, ever.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_series):
        T = n_periods
        a = np.clip(0.55 + rng.normal(0, 0.01, size=T), 0.05, 0.95)
        gap0 = rng.uniform(0.01, 0.03)
        gap_inf = 0.8 * tau
        gaps = gap_inf - (gap_inf - gap0) * np.exp(-np.arange(T) / 3.0)
        b = np.clip(a - gaps, 0.02, 0.98)
        frames.append(_frame({"A": a, "B": b}, _ns(T, n_range, rng), rng))
    return frames, [None] * n_series


def seasonal_panels(n_series: int = 8, n_periods: int = 12,
                    tau: float = 0.10, seed: int = 13,
                    n_range=(800, 2000)):
    """Panels whose gap oscillates seasonally around a safe mean with
    zero trend — the second near-miss negative: oscillation is not
    drift, and a trend reader must not call it one."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_series):
        T = n_periods
        a = np.clip(0.55 + rng.normal(0, 0.01, size=T), 0.05, 0.95)
        phase = rng.uniform(0, 2 * np.pi)
        gaps = 0.45 * tau + 0.25 * tau * np.sin(
            2 * np.pi * np.arange(T) / 6.0 + phase)
        b = np.clip(a - gaps, 0.02, 0.98)
        frames.append(_frame({"A": a, "B": b}, _ns(T, n_range, rng), rng))
    return frames, [None] * n_series


def tiny_n_panel(n_periods: int = 8, tau: float = 0.10, seed: int = 17,
                 n_per: int = 25):
    """One panel with very small per-period counts. Its intervals must
    be honestly wide — the fake-precision negative. Returns a frame
    (and a large-n twin with identical latent rates is available via
    :func:`tiny_n_panel` with a bigger ``n_per``)."""
    rng = np.random.default_rng(seed)
    T = n_periods
    a = np.full(T, 0.55)
    gaps = np.linspace(0.02, tau + 0.05, T)
    b = a - gaps
    ns = {g: np.full(T, n_per) for g in ("A", "B")}
    return _frame({"A": a, "B": b}, ns, rng)
