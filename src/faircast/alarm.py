"""Alarm policies with a designed false-alarm rate.

An alarm is a sequential decision: each new period, refit and ask whether
the breach probability within the policy horizon justifies acting. The
alarm threshold on that probability is chosen so the *step-level
false-alarm rate* (FAR) — alarms on periods not followed by a breach
within the horizon, divided by such periods — meets a target, estimated
by rolling-origin backtest on the panel's own history. The designed
quantity reported alongside is the in-control average run length,
ARL0 = 1 / FAR.

Paper 11's second headline finding is printed in every design report
because it is the operational truth about this procedure: thresholds
fitted to a 0.10 FAR target on training states fired at a fold-mean
0.293 on held-out states (per-fold range 0.005-1.000). A FAR designed on
one series is a *starting point* for another, never a guarantee —
recalibrate per series, and read the achieved backtest FAR, not the
target, as the honest number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conformal import truncate_panel
from .data import Panel, SeriesPanel
from .errors import FairCastError

TRANSPORT_CAVEAT = (
    "FAR transport caveat (Paper 11, Ghosh 2026): alarm thresholds fitted "
    "to a 0.10 step-FAR target on training jurisdictions fired at a "
    "fold-mean 0.293 on held-out jurisdictions (range 0.005-1.000). The "
    "backtest FAR below is estimated on THIS series' own history; on a "
    "new series, recalibrate before trusting the budget.")


@dataclass
class AlarmDesign:
    threshold_on_probability: float
    target_far: float
    backtest_far: float | None     # None when not estimable
    arl0: float | None             # 1 / backtest_far, periods
    n_negative_origins: int
    n_positive_origins: int
    detection_rate: float | None   # alarms on positive origins
    far_estimable: bool
    caveat: str = TRANSPORT_CAVEAT


def _observed(panel, metric) -> np.ndarray:
    if isinstance(panel, SeriesPanel):
        return np.asarray(panel.values, dtype=float)
    return panel.metric_series(metric)


def design_alarm(make_forecaster, panel, policy_horizon: int = 2,
                 target_far: float = 0.10, min_train: int = 4) -> AlarmDesign:
    """Choose the alarm threshold for a target step-level FAR by backtest.

    ``make_forecaster`` returns an unfitted Forecaster configured for the
    metric/threshold of interest; its horizon must be >= policy_horizon.
    An origin (a fit on the first o periods) is *negative* if the observed
    metric does not breach within the next ``policy_horizon`` periods,
    *positive* if it does.
    """
    if not (0.0 < target_far < 1.0):
        raise FairCastError(f"target_far must lie in (0, 1); got {target_far}")
    proto = make_forecaster()
    if proto.horizon < policy_horizon:
        raise FairCastError(
            f"forecaster horizon {proto.horizon} < policy horizon "
            f"{policy_horizon}")
    metric, tau, direction = proto.metric, proto.threshold, proto.direction
    observed = _observed(panel, metric)
    T = len(observed)

    def breached(idx: int) -> bool:
        v = observed[idx]
        return v >= tau if direction == "above" else v <= tau

    rows = []  # (score, is_positive)
    for origin in range(min_train, T):
        f = make_forecaster().fit(truncate_panel(panel, origin))
        score = float(f.breach_table()["p_breach_by_horizon"]
                      .iloc[policy_horizon - 1])
        window = range(origin, min(origin + policy_horizon, T))
        is_pos = any(breached(i) for i in window)
        rows.append((score, is_pos))
    if not rows:
        raise FairCastError(
            "series too short to design an alarm policy: no backtest "
            f"origins with min_train={min_train}")
    neg = np.array([s for s, p in rows if not p])
    pos = np.array([s for s, p in rows if p])
    if len(neg) == 0:
        raise FairCastError(
            "every backtest origin is followed by a breach within the "
            "policy horizon; a false-alarm rate cannot be designed on a "
            "series that is always breaching. Loosen the threshold or "
            "shorten the policy horizon.")
    cands = np.unique(np.r_[neg, pos, 0.5])
    fars = np.array([float((neg >= c).mean()) for c in cands])
    d = np.abs(fars - target_far)
    best = np.lexsort((fars, d))[0]
    theta = float(cands[best])
    far = float(fars[best])
    estimable = len(neg) >= 5
    return AlarmDesign(
        threshold_on_probability=theta,
        target_far=float(target_far),
        backtest_far=far if estimable else None,
        arl0=(1.0 / far) if (estimable and far > 0) else None,
        n_negative_origins=int(len(neg)),
        n_positive_origins=int(len(pos)),
        detection_rate=float((pos >= theta).mean()) if len(pos) else None,
        far_estimable=estimable)
