"""The drift-budget certificate (Paper 11's exact layer, productised).

Theorem (the form proved in Paper 11). Let a fairness gap be computed
from per-cell outcome distributions. Between consecutive periods, the
gap can move by at most the sum over cells of the total-variation (TV)
distance between the periods' cell distributions:

    |gap_{t+1} - gap_t| <= eps_t,   eps_t = sum_cells TV(P_t^cell, P_{t+1}^cell)

hence ``gap_{t+h} <= gap_t + eps_t + ... + eps_{t+h-1}`` for every h.
Paper 11 verified the population form of this bound exhaustively:
**0 violations in 22,032 checks** (51 jurisdictions x 9 years x 3 tasks
x 2 model classes), with the score-distribution TV as eps.

FairCast's counts-level version applies the same budget to the
*thresholded decision distributions*: for DP/EO gaps the attaining
functionals are per-cell rates, and the TV between two Bernoulli cell
distributions is ``|p_{t+1} - p_t|`` — so with counts inputs the budget is
exact for the gap it certifies, with no histogram step. Scope: the
counts-level budget certifies *gap* functionals (dp_gap, eo_gap, and a
single selection rate). The disparate-impact *ratio* is not Lipschitz
in this budget (it can move arbitrarily when the denominator rate is
near zero) and is refused, not silently certified. Two modes:

- **population mode** (``noise_z = 0``): inputs are population rates;
  the bound is a theorem and :func:`verify` must report 0 violations.
- **batch mode** (``noise_z > 0``, e.g. 1.96): each step's budget is
  inflated by ``noise_z`` times the standard error of the rate change,
  turning the theorem into a confidence bound for batch-observed data.
  Paper 11's honest warning ships with it: at batch n = 2,000 the
  certified safe horizon was **0 in every cell** of its corpus — the
  bound held everywhere and certified nothing. A certificate that is
  silent is reported as silent, never rounded up to a promise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel
from .errors import FairCastError


def rate_budget(panel: Panel, kinds: tuple[str, ...] = ("sel",),
                noise_z: float = 0.0) -> np.ndarray:
    """Per-step drift budget eps_t (length T-1) from a counts/decisions
    panel: sum over the metric's cells of |rate change|, optionally with
    a z-inflation for sampling noise."""
    if not isinstance(panel, Panel):
        raise FairCastError("rate_budget needs a Panel (counts/decisions)")
    T = len(panel.periods)
    eps = np.zeros(T - 1)
    for (group, kind), cell in panel.cells.items():
        if kind not in kinds:
            continue
        p = cell.rate
        step = np.abs(np.diff(p))
        if noise_z > 0:
            se = np.sqrt(p[:-1] * (1 - p[:-1]) / cell.n[:-1]
                         + p[1:] * (1 - p[1:]) / cell.n[1:])
            step = step + noise_z * se
        eps += step
    return eps


def tv_hist(p_scores: np.ndarray, q_scores: np.ndarray,
            n_bins: int = 50) -> float:
    """Histogram total variation between two score samples on [0, 1]
    (Paper 11's estimator; with the default 50 bins an edge sits at the
    0.5 decision cut, which is what the gap functionals attain on)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    hp, _ = np.histogram(p_scores, bins=edges)
    hq, _ = np.histogram(q_scores, bins=edges)
    if hp.sum() == 0 or hq.sum() == 0:
        return float("nan")
    hp = hp / hp.sum()
    hq = hq / hq.sum()
    return float(0.5 * np.abs(hp - hq).sum())


def score_budget(scores_by_period: list[dict], n_bins: int = 50) -> np.ndarray:
    """Per-step budget from raw score samples: ``scores_by_period[t]``
    maps a cell key to that period's score array. eps_t is the sum over
    cells of histogram TV — Paper 11's population construction."""
    T = len(scores_by_period)
    eps = np.zeros(T - 1)
    cells = list(scores_by_period[0].keys())
    for t in range(T - 1):
        for c in cells:
            eps[t] += tv_hist(scores_by_period[t][c],
                              scores_by_period[t + 1][c], n_bins)
    return eps


def verify(metric_values: np.ndarray, eps: np.ndarray,
           direction: str = "above", tol: float = 1e-9) -> dict:
    """Exhaustively check the certified bound on an observed series:
    for every (t, h), the realised value must lie within the budget.
    Returns {"checks": int, "violations": int}. On population inputs
    (noise_z = 0 budgets) violations must be 0 — anything else is a bug
    or a mis-specified budget, and is reported, not absorbed."""
    m = np.asarray(metric_values, dtype=float)
    eps = np.asarray(eps, dtype=float)
    T = len(m)
    if len(eps) != T - 1:
        raise FairCastError(
            f"eps has length {len(eps)}; expected T-1 = {T - 1}")
    checks = violations = 0
    for t in range(T):
        cum = 0.0
        for h in range(1, T - t):
            cum += eps[t + h - 1]
            checks += 1
            if direction == "above":
                if m[t + h] > m[t] + cum + tol:
                    violations += 1
            else:
                if m[t + h] < m[t] - cum - tol:
                    violations += 1
    return {"checks": checks, "violations": violations}


def certify(metric_values: np.ndarray, eps: np.ndarray, threshold: float,
            direction: str = "above", periods: list | None = None,
            max_horizon: int = 5) -> pd.DataFrame:
    """Certified safe horizon per origin: the largest h such that the
    budget bound stays on the safe side of ``threshold``. A safe horizon
    of 0 is the certificate declining to certify — reported as 0."""
    m = np.asarray(metric_values, dtype=float)
    eps = np.asarray(eps, dtype=float)
    T = len(m)
    if periods is None:
        periods = list(range(T))
    rows = []
    for t in range(T - 1):
        cum = 0.0
        safe = 0
        bounds = []
        for h in range(1, min(max_horizon, T - 1 - t) + 1):
            cum += eps[t + h - 1]
            bound = m[t] + cum if direction == "above" else m[t] - cum
            bounds.append(bound)
            ok = bound < threshold if direction == "above" \
                else bound > threshold
            if ok:
                safe = h
            else:
                break
        rows.append({
            "origin_period": periods[t],
            "value": float(m[t]),
            "certified_safe_horizon": safe,
            "bound_at_safe_horizon": (bounds[safe - 1] if safe else None),
            "bound_at_h1": (bounds[0] if bounds else None),
        })
    return pd.DataFrame(rows)
