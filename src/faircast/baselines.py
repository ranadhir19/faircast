"""Trivial forecasters FairCast must beat to claim any skill.

Paper 11's pre-registered headline was falsified by one of these: a
sophisticated forecaster tied linear extrapolation on median lead time at
matched false-alarm rates. They are therefore not straw men here — they
are first-class engines with the same simulation interface as the
state-space forecaster, and the self-benchmark (:mod:`faircast.benchmark`)
scores FairCast against them and prints NO-SKILL when it does not beat
them. A roadmap for what "beat" means is in :mod:`faircast.benchmark`.

Both baselines work on the observed metric series directly:

- **Persistence** (last value carried forward): future paths are a random
  walk from the last observed value, with step SD = SD of observed first
  differences (which includes sampling noise, so if anything it is
  generous to persistence).
- **Linear extrapolation**: OLS of the metric on time; paths draw the
  coefficients from their sampling distribution and add residual noise.
"""
from __future__ import annotations

import numpy as np


def persistence_paths(values: np.ndarray, horizon: int, n_paths: int,
                      rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    diffs = np.diff(values)
    sd = float(np.std(diffs)) if len(diffs) else 0.0
    steps = rng.standard_normal((n_paths, horizon)) * sd
    return values[-1] + np.cumsum(steps, axis=1)


def linear_paths(values: np.ndarray, horizon: int, n_paths: int,
                 rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    T = len(values)
    t = np.arange(T, dtype=float)
    X = np.column_stack([np.ones(T), t])
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    resid = values - X @ beta
    dof = max(T - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    draws = rng.multivariate_normal(beta, cov, size=n_paths)
    tf = T + np.arange(horizon, dtype=float)
    mean = draws[:, [0]] + draws[:, [1]] * tf[None, :]
    return mean + rng.standard_normal((n_paths, horizon)) * np.sqrt(s2)
