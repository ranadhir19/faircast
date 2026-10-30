"""Rolling-origin backtest calibration (split-conformal style).

The state-space model's own intervals are only as good as the model. This
module stress-tests them on the panel's own history: refit at every
feasible origin, forecast forward, and record how far the realised metric
fell outside the nominal band, per horizon. The per-horizon *inflation
factor* is the split-conformal quantile of the needed widenings; applying
it makes the reported band a backtest-calibrated band, and the backtest
coverage (before and after) is reported with it.

Honesty rules: with too few origins the calibration is labelled
*uncalibrated* and the model's own intervals are returned untouched; the
inflation factor is capped at 4 (beyond that the honest statement is
"this series is not forecastable at this horizon", which the coverage
numbers will show).
"""
from __future__ import annotations

import numpy as np

from .data import Panel, SeriesPanel, CellSeries
from .errors import FairCastError

MAX_INFLATION = 4.0
MIN_ORIGINS = 6


def truncate_panel(panel, k: int):
    """The first k periods of a panel (k >= 1)."""
    if isinstance(panel, SeriesPanel):
        return SeriesPanel(panel.periods[:k], panel.values[:k],
                           panel.ses[:k] if panel.ses is not None else None)
    if isinstance(panel, Panel):
        cells = {key: CellSeries(cs.group, cs.kind, cs.n[:k],
                                 cs.positives[:k])
                 for key, cs in panel.cells.items()}
        return Panel(panel.periods[:k], panel.groups, cells,
                     panel.source_kind)
    raise FairCastError(f"cannot truncate {type(panel).__name__}")


def _observed_series(panel, metric) -> np.ndarray:
    if isinstance(panel, SeriesPanel):
        return np.asarray(panel.values, dtype=float)
    return panel.metric_series(metric)


def backtest_inflation(make_forecaster, panel, nominal: float = 0.90,
                       min_train: int = 4) -> dict:
    """Rolling-origin interval audit for a forecaster configuration.

    ``make_forecaster`` is a zero-arg callable returning an unfitted
    Forecaster. Returns per-horizon coverage and inflation factors.
    """
    proto = make_forecaster()
    horizon = proto.horizon
    metric = proto.metric
    observed = _observed_series(panel, metric)
    T = len(observed)
    lo_q = (1.0 - nominal) / 2.0
    factors: dict[int, list[float]] = {}
    covered: dict[int, list[bool]] = {}
    for origin in range(min_train, T):
        sub = truncate_panel(panel, origin)
        f = make_forecaster().fit(sub)
        qs = np.quantile(f.paths_, [lo_q, 0.5, 1.0 - lo_q], axis=0)
        for h in range(1, min(horizon, T - origin) + 1):
            actual = observed[origin + h - 1]
            qlo, med, qhi = qs[0, h - 1], qs[1, h - 1], qs[2, h - 1]
            half = max((qhi - qlo) / 2.0, 1e-9)
            need = abs(actual - med) / half
            factors.setdefault(h, []).append(min(need, MAX_INFLATION))
            covered.setdefault(h, []).append(bool(qlo <= actual <= qhi))
    hs = sorted(factors)
    inflation = np.full(horizon, np.nan)
    cov_before, n_orig = {}, {}
    for h in hs:
        f_arr = np.sort(np.asarray(factors[h]))
        m = len(f_arr)
        n_orig[h] = m
        cov_before[h] = float(np.mean(covered[h]))
        if m >= 3:
            idx = int(np.ceil((m + 1) * nominal)) - 1
            inflation[h - 1] = float(f_arr[min(max(idx, 0), m - 1)])
    total = sum(n_orig.values())
    return {
        "nominal": nominal,
        "inflation": inflation,
        "coverage_before": cov_before,
        "n_origins": n_orig,
        "total_origins": total,
        "estimable": total >= MIN_ORIGINS,
    }


def calibrate(forecaster, panel) -> dict:
    """Attach backtest-calibrated interval inflation to a fitted forecaster.

    Returns the backtest report. If the panel is too short to backtest,
    the forecaster is left uncalibrated and says so.
    """
    from .forecast import Forecaster  # local: avoid an import cycle

    def make():
        return Forecaster(metric=forecaster.metric.name,
                          threshold=forecaster.threshold,
                          direction=forecaster.direction,
                          horizon=forecaster.horizon,
                          n_paths=forecaster.n_paths,
                          seed=forecaster.seed,
                          sampling_noise=forecaster.sampling_noise)

    rep = backtest_inflation(make, panel)
    if not rep["estimable"]:
        forecaster.calibrated = False
        forecaster.calibration_note = (
            f"uncalibrated: only {rep['total_origins']} backtest origins "
            f"available (need {MIN_ORIGINS}); intervals are the model's own")
        forecaster._inflation = None
        return rep
    infl = rep["inflation"]
    infl = np.where(np.isnan(infl), 1.0, infl)
    forecaster._inflation = np.clip(infl, 1.0, MAX_INFLATION)
    forecaster.calibrated = True
    cov = rep["coverage_before"]
    cov_txt = ", ".join(f"h{h}: {cov[h]:.2f} (n={rep['n_origins'][h]})"
                        for h in sorted(cov))
    forecaster.calibration_note = (
        f"conformalised on {rep['total_origins']} rolling origins; "
        f"pre-calibration {int(rep['nominal'] * 100)}% band coverage by "
        f"horizon — {cov_txt}")
    return rep
