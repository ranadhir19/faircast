"""The FairCast forecaster.

Fit on a :class:`~faircast.data.Panel` (grouped counts or decisions) or a
:class:`~faircast.data.SeriesPanel` (an existing metric series). The model
is one local-linear-trend state-space per cell rate (logit scale,
binomial observation noise known from the counts); forecast paths are
simulated from the filtered states and pushed through the metric, so a
breach probability is a statement about the *underlying* rates, not about
a line fitted to the gap.

What this does **not** do: promise lead time. Paper 11 (Ghosh, submitted
to AI & Society, 2026) pre-registered exactly that claim for a forecaster
of this family and falsified it — at matched false-alarm rates it tied
linear extrapolation on median lead. FairCast ships the baselines and a
self-benchmark so the question "does it beat extrapolation *here*?" is
answered on the user's own panel, in the open, every time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel, SeriesPanel
from .errors import FairCastError
from .metrics import get_metric
from .statespace import FittedTrend, fit_trend, logit_obs

_SERIES_R_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def _fit_series_trend(values: np.ndarray,
                      ses: np.ndarray | None) -> FittedTrend:
    if ses is not None:
        return fit_trend(values, ses ** 2)
    base = max(float(np.var(np.diff(values))) / 2.0, 1e-8)
    best = None
    for s in _SERIES_R_SCALES:
        ft = fit_trend(values, np.full(len(values), base * s))
        if best is None or ft.loglik > best.loglik:
            best = ft
    return best


class Forecaster:
    def __init__(self, metric: str = "dp_gap", threshold: float = 0.05,
                 direction: str | None = None, horizon: int = 6,
                 n_paths: int = 4000, seed: int = 12345,
                 sampling_noise: bool = False):
        self.metric = get_metric(metric)
        if not (self.metric.bounds[0] <= threshold <= self.metric.bounds[1]):
            raise FairCastError(
                f"threshold {threshold} is outside the plausible range "
                f"{self.metric.bounds} of metric {metric!r}")
        self.threshold = float(threshold)
        self.direction = direction or self.metric.default_direction
        if self.direction not in ("above", "below"):
            raise FairCastError(
                f"direction must be 'above' or 'below', got {direction!r}")
        if horizon < 1:
            raise FairCastError("horizon must be at least 1")
        self.horizon = int(horizon)
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self.sampling_noise = bool(sampling_noise)
        self._fitted = False
        self.calibrated = False
        self.calibration_note = "not calibrated"
        self._inflation: np.ndarray | None = None

    # ---------------------------------------------------------------- fit
    def fit(self, panel) -> "Forecaster":
        rng = np.random.default_rng(self.seed)
        if isinstance(panel, Panel):
            self.metric.validate_groups(panel.groups)
            self._fit_panel(panel, rng)
        elif isinstance(panel, SeriesPanel):
            self._fit_series(panel, rng)
        else:
            raise FairCastError(
                "fit expects a Panel (load_counts / load_decisions) or a "
                f"SeriesPanel (load_series); got {type(panel).__name__}")
        self._fitted = True
        return self

    def _future_labels(self, periods: list) -> list:
        if all(isinstance(p, (int, float)) and not isinstance(p, bool)
               for p in periods):
            diffs = np.diff(np.asarray(periods, dtype=float))
            step = float(np.median(diffs)) if len(diffs) else 1.0
            last = periods[-1]
            labels = [last + step * h for h in range(1, self.horizon + 1)]
            if all(float(p).is_integer() for p in periods) and \
                    float(step).is_integer():
                labels = [int(x) for x in labels]
            return labels
        return [f"+{h}" for h in range(1, self.horizon + 1)]

    def _fit_panel(self, panel: Panel, rng) -> None:
        self.periods_ = list(panel.periods)
        self.observed_ = panel.metric_series(self.metric)
        self.trends_: dict[tuple[str, str], FittedTrend] = {}
        cell_paths: dict[tuple[str, str], np.ndarray] = {}
        for group in panel.groups:
            for kind in self.metric.kinds:
                cell = panel.cell(group, kind)
                z, r = logit_obs(cell.n, cell.positives)
                trend = fit_trend(z, r)
                self.trends_[(group, kind)] = trend
                latent = 1.0 / (1.0 + np.exp(-trend.simulate(
                    self.horizon, self.n_paths, rng)))
                if self.sampling_noise:
                    n_future = float(np.median(cell.n[-3:]))
                    latent = rng.binomial(int(n_future), latent) / n_future
                cell_paths[(group, kind)] = latent
        paths = np.empty((self.n_paths, self.horizon))
        for h in range(self.horizon):
            for s in range(self.n_paths):
                rates = {k: v[s, h] for k, v in cell_paths.items()}
                paths[s, h] = self.metric.compute(rates, panel.groups)
        self.paths_ = paths
        self.future_periods_ = self._future_labels(self.periods_)

    def _fit_series(self, panel: SeriesPanel, rng) -> None:
        self.periods_ = list(panel.periods)
        self.observed_ = np.asarray(panel.values, dtype=float)
        trend = _fit_series_trend(self.observed_, panel.ses)
        self.trends_ = {("(series)", "value"): trend}
        raw = trend.simulate(self.horizon, self.n_paths, rng)
        lo, hi = self.metric.bounds
        self.paths_ = np.clip(raw, lo, hi)
        self.future_periods_ = self._future_labels(self.periods_)

    # ------------------------------------------------------------ outputs
    def _require_fit(self) -> None:
        if not self._fitted:
            raise FairCastError("call fit(panel) before requesting output")

    def _breach_mask(self) -> np.ndarray:
        if self.direction == "above":
            return self.paths_ >= self.threshold
        return self.paths_ <= self.threshold

    def breach_table(self) -> pd.DataFrame:
        """Per-horizon breach probabilities: ``p_first_breach_in_period``
        is the probability that the *first* breach occurs in that period;
        ``p_breach_by_horizon`` is the cumulative probability of having
        breached at or before it. The first-breach column plus the
        residual ``1 - p_breach_by_horizon[-1]`` (no breach in window)
        sums to 1."""
        self._require_fit()
        mask = self._breach_mask()
        cum = np.maximum.accumulate(mask, axis=1).mean(axis=0)
        first = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)
        first_p = np.array([(first == h).mean()
                            for h in range(self.horizon)])
        return pd.DataFrame({
            "horizon": np.arange(1, self.horizon + 1),
            "period": self.future_periods_,
            "p_first_breach_in_period": first_p,
            "p_breach_by_horizon": cum,
        })

    def breach_time_distribution(self) -> dict:
        """First-breach-period probabilities; ``"none"`` = no breach within
        the horizon. Sums to 1."""
        self._require_fit()
        mask = self._breach_mask()
        first = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)
        out = {}
        for h in range(self.horizon):
            out[self.future_periods_[h]] = float((first == h).mean())
        out["none"] = float((first < 0).mean())
        return out

    def fan(self, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        """Metric quantiles per future period. If conformal calibration has
        been run, the outer (5%/95%) band is widened by the per-horizon
        inflation factor estimated on backtests; inner quantiles are the
        model's own."""
        self._require_fit()
        qs = np.quantile(self.paths_, quantiles, axis=0)
        data = {"horizon": np.arange(1, self.horizon + 1),
                "period": self.future_periods_}
        for i, q in enumerate(quantiles):
            data[f"q{int(round(q * 100)):02d}"] = qs[i]
        df = pd.DataFrame(data)
        if self._inflation is not None and "q05" in df and "q95" in df:
            med = df["q50"]
            df["q05"] = (med - (med - df["q05"]) * self._inflation)
            df["q95"] = (med + (df["q95"] - med) * self._inflation)
            lo, hi = self.metric.bounds
            df["q05"] = df["q05"].clip(lo, hi)
            df["q95"] = df["q95"].clip(lo, hi)
        return df

    def median_breach_horizon(self) -> int | None:
        """First horizon at which cumulative breach probability >= 0.5."""
        cum = self.breach_table()["p_breach_by_horizon"].to_numpy()
        idx = np.argmax(cum >= 0.5) if (cum >= 0.5).any() else None
        return int(idx + 1) if idx is not None else None

    def summary(self) -> dict:
        self._require_fit()
        table = self.breach_table()
        return {
            "metric": self.metric.name,
            "threshold": self.threshold,
            "direction": self.direction,
            "last_period": self.periods_[-1],
            "current_value": float(self.observed_[-1]),
            "horizon": self.horizon,
            "p_breach_by_final_horizon": float(
                table["p_breach_by_horizon"].iloc[-1]),
            "median_breach_horizon": self.median_breach_horizon(),
            "calibrated": bool(self.calibrated),
            "calibration_note": self.calibration_note,
        }
