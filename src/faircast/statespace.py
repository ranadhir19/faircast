"""Local-linear-trend state-space model on a transformed scale.

Each cell rate is modelled on the logit scale (rates live in (0, 1) and
their sampling noise is binomial, so the logit scale is where a Gaussian
state-space is honest). A direct metric series is modelled on the identity
scale. The state is (level, damped slope); the observation variance is
*known* from the counts (delta method) rather than fitted, which is what
keeps tiny-n groups honest: with n = 30 the observation variance is large,
the filter knows it, and forecast intervals widen instead of faking
precision.

Hyperparameters (process noises, damping) are chosen by grid search on the
Kalman log-likelihood — deterministic, no optimiser lottery on 4-9 points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_Q_LEVEL = (1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 2e-1)
_Q_SLOPE = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
_PHI = (0.6, 0.8, 0.9, 1.0)


def logit_obs(n: np.ndarray, positives: np.ndarray):
    """Logit-scale observations and delta-method variances from counts.

    Uses the +0.5 smoothing so a 0 or n count stays finite; the variance
    1/(k+.5) + 1/(n-k+.5) is the standard logit variance estimator.
    """
    k = np.asarray(positives, dtype=float)
    nn = np.asarray(n, dtype=float)
    p = (k + 0.5) / (nn + 1.0)
    z = np.log(p / (1.0 - p))
    r = 1.0 / (k + 0.5) + 1.0 / (nn - k + 0.5)
    return z, r


def _filter(z, r, q_level, q_slope, phi):
    """Run the Kalman filter; return (loglik, m_T, P_T)."""
    T = len(z)
    F = np.array([[1.0, 1.0], [0.0, phi]])
    Q = np.diag([q_level, q_slope])
    m = np.array([z[0], np.clip(z[1] - z[0], -1.0, 1.0) if T > 1 else 0.0])
    P = np.diag([1.0, 0.25])
    ll = 0.0
    for t in range(T):
        if t > 0:
            m = F @ m
            P = F @ P @ F.T + Q
        S = P[0, 0] + r[t]
        if S <= 0:
            return -np.inf, m, P
        innov = z[t] - m[0]
        ll += -0.5 * (np.log(2.0 * np.pi * S) + innov * innov / S)
        K = P[:, 0] / S
        m = m + K * innov
        P = P - np.outer(K, P[0, :])
        P = 0.5 * (P + P.T)
    return ll, m, P


@dataclass
class FittedTrend:
    level: float
    slope: float
    m: np.ndarray       # filtered state mean at the last period
    P: np.ndarray       # filtered state covariance at the last period
    q_level: float
    q_slope: float
    phi: float
    loglik: float

    def simulate(self, horizon: int, n_paths: int,
                 rng: np.random.Generator) -> np.ndarray:
        """Latent transformed-scale paths, shape (n_paths, horizon)."""
        F = np.array([[1.0, 1.0], [0.0, self.phi]])
        chol_P = np.linalg.cholesky(self.P + 1e-12 * np.eye(2))
        x = self.m + (chol_P @ rng.standard_normal((2, n_paths))).T
        q = np.array([self.q_level, self.q_slope])
        out = np.empty((n_paths, horizon))
        for h in range(horizon):
            x = x @ F.T + rng.standard_normal((n_paths, 2)) * np.sqrt(q)
            out[:, h] = x[:, 0]
        return out


def fit_trend(z: np.ndarray, r: np.ndarray) -> FittedTrend:
    """Fit the trend model by grid search on the Kalman log-likelihood."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if len(z) < 3:
        raise ValueError("fit_trend needs at least 3 observations")
    if (r <= 0).any():
        raise ValueError("observation variances must be positive")
    best = None
    for ql in _Q_LEVEL:
        for qs in _Q_SLOPE:
            for phi in _PHI:
                ll, m, P = _filter(z, r, ql, qs, phi)
                key = (ll, -(ql + qs), -abs(phi - 0.9))
                if best is None or key > best[0]:
                    best = (key, ll, m, P, ql, qs, phi)
    _, ll, m, P, ql, qs, phi = best
    return FittedTrend(level=float(m[0]), slope=float(m[1]), m=m, P=P,
                       q_level=ql, q_slope=qs, phi=phi, loglik=float(ll))
