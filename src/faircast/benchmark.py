"""Self-benchmark: FairCast vs the trivial forecasters, FAR-matched.

The protocol mirrors Paper 11's evaluation, because that protocol is
what falsified Paper 11's own headline:

- Every method issues, at each origin (a fit on the history so far), a
  breach probability for the next ``horizon`` periods.
- Alarm thresholds are chosen **leave-one-series-out**: the threshold
  applied to a series is calibrated on the *other* series' origins to
  hit the target step-level false-alarm rate (ties break toward the
  smaller FAR) — no series grades its own homework.
- A breaching series is *covered* if some origin at or before the
  breach index alarms; *lead* = breach index - first alarm origin + 1
  (periods of warning). Coverage and median lead are reported per
  method per FAR target, with achieved FAR and pooled Brier scores.
- **Skill rule (the paper's falsifier, verbatim in spirit):** FairCast
  demonstrates skill at a FAR target only if its median lead *strictly
  exceeds both* trivial forecasters' at matched FAR. Anything else —
  a tie with extrapolation included — prints **NO-SKILL**. On Paper
  11's own corpus the honest verdict was NO-SKILL at every target;
  expect this harness to say so whenever it is true.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import linear_paths, persistence_paths
from .conformal import truncate_panel
from .data import Panel, SeriesPanel
from .errors import FairCastError
from .forecast import Forecaster
from .metrics import get_metric

METHODS = ("faircast", "persistence", "linear")


@dataclass
class BenchmarkResult:
    table: pd.DataFrame
    verdicts: list[str] = field(default_factory=list)
    brier: dict = field(default_factory=dict)
    n_series: int = 0
    n_breaching: int = 0
    n_already_breached: int = 0

    def to_markdown(self) -> str:
        lines = [f"Series: {self.n_series} ({self.n_breaching} breaching "
                 f"inside the monitored window, {self.n_already_breached} "
                 f"already in breach before the first origin and excluded "
                 f"from coverage).",
                 "",
                 "| method | FAR target | achieved FAR | coverage | "
                 "median lead | Brier |",
                 "|---|---|---|---|---|---|"]
        for _, r in self.table.iterrows():
            lines.append(
                f"| {r.method} | {r.far_target:.2f} | "
                f"{r.achieved_far:.3f} | {r.coverage:.3f} | "
                f"{r.median_lead:.1f} | {self.brier.get(r.method, float('nan')):.3f} |")
        lines.append("")
        lines.extend(f"- {v}" for v in self.verdicts)
        return "\n".join(lines)


def _breach_index(values: np.ndarray, tau: float, direction: str):
    hit = values >= tau if direction == "above" else values <= tau
    return int(np.argmax(hit)) if hit.any() else None


def _paths_breach_prob(paths: np.ndarray, tau: float,
                       direction: str) -> float:
    mask = paths >= tau if direction == "above" else paths <= tau
    return float(np.maximum.accumulate(mask, axis=1)[:, -1].mean())


def benchmark_panels(panels: list, metric: str = "dp_gap",
                     threshold: float = 0.05, direction: str | None = None,
                     horizon: int = 2,
                     far_targets=(0.05, 0.10, 0.20),
                     min_train: int = 4, seed: int = 12345,
                     n_paths: int = 2000) -> BenchmarkResult:
    """Run the FAR-matched benchmark over a list of single-series panels
    (Panel or SeriesPanel objects, one monitored series each)."""
    if not panels:
        raise FairCastError("benchmark needs at least one panel")
    met = get_metric(metric)
    direction = direction or met.default_direction

    def observed_of(p) -> np.ndarray:
        if isinstance(p, SeriesPanel):
            return np.asarray(p.values, dtype=float)
        return p.metric_series(met)

    # ---- pass 1: scores at every origin for every method
    records = []  # dicts: series, origin, method, score, label
    breaches = []
    for si, p in enumerate(panels):
        obs = observed_of(p)
        T = len(obs)
        breaches.append(_breach_index(obs, threshold, direction))
        for origin in range(min_train, T):
            hist = obs[:origin]
            label = int(any(
                (obs[i] >= threshold) if direction == "above"
                else (obs[i] <= threshold)
                for i in range(origin, min(origin + horizon, T))))
            sub = truncate_panel(p, origin)
            fc = Forecaster(metric=metric, threshold=threshold,
                            direction=direction, horizon=horizon,
                            n_paths=n_paths, seed=seed).fit(sub)
            scores = {
                "faircast": float(fc.breach_table()
                                  ["p_breach_by_horizon"].iloc[-1]),
            }
            rng = np.random.default_rng(seed + 1000 * si + origin)
            for name, fn in (("persistence", persistence_paths),
                             ("linear", linear_paths)):
                paths = fn(hist, horizon, n_paths, rng)
                lo, hi = met.bounds
                scores[name] = _paths_breach_prob(
                    np.clip(paths, lo, hi), threshold, direction)
            for m, s in scores.items():
                records.append({"series": si, "origin": origin,
                                "method": m, "score": s, "label": label})
    R = pd.DataFrame(records)
    if R.empty:
        raise FairCastError(
            "no backtest origins: every series is shorter than "
            f"min_train + 1 = {min_train + 1} periods")

    brier = {m: float(((g.score - g.label) ** 2).mean())
             for m, g in R.groupby("method")}

    # ---- pass 2: LOSO thresholds, trajectory outcomes
    rows = []
    verdicts = []
    # A series that first breached *before* the first backtest origin was
    # already in breach when monitoring started: it cannot be "covered"
    # by any forecaster, so it is excluded from coverage denominators
    # and reported separately rather than silently penalising everyone.
    n_already = sum(b is not None and b < min_train for b in breaches)
    n_breaching = sum(b is not None and b >= min_train for b in breaches)
    for target in far_targets:
        med_lead = {}
        for m in METHODS:
            thetas = {}
            for si in range(len(panels)):
                others = R[(R.method == m) & (R.series != si)
                           & (R.label == 0)]
                own = R[(R.method == m) & (R.series == si)]
                if len(others) == 0:
                    thetas[si] = 0.5
                    continue
                cands = np.unique(np.r_[others.score.to_numpy(), 0.5])
                fars = np.array([
                    float((others.score.to_numpy() >= c).mean())
                    for c in cands])
                d = np.abs(fars - target)
                thetas[si] = float(cands[np.lexsort((fars, d))[0]])
            alarmed_neg = alarmed_all = 0
            leads = []
            for si, b in enumerate(breaches):
                g = R[(R.method == m) & (R.series == si)]
                alarms = g[g.score >= thetas[si]]
                neg = g[g.label == 0]
                alarmed_neg += int((neg.score >= thetas[si]).sum())
                alarmed_all += len(neg)
                if b is None or b < min_train:
                    continue
                timely = alarms[alarms.origin <= b]
                if len(timely):
                    leads.append(b - int(timely.origin.min()) + 1)
            cov = (len(leads) / n_breaching) if n_breaching else 0.0
            med = float(np.median(leads)) if leads else 0.0
            med_lead[m] = med
            rows.append({
                "method": m, "far_target": target,
                "achieved_far": (alarmed_neg / alarmed_all)
                if alarmed_all else float("nan"),
                "coverage": cov, "median_lead": med,
                "n_breach": n_breaching})
        if n_breaching == 0:
            verdicts.append(
                f"FAR target {target:.2f}: no series in the panel "
                "breaches; forecasting skill is not assessable.")
        elif med_lead["faircast"] > med_lead["persistence"] and \
                med_lead["faircast"] > med_lead["linear"]:
            verdicts.append(
                f"FAR target {target:.2f}: SKILL — faircast median lead "
                f"{med_lead['faircast']:.1f} strictly exceeds persistence "
                f"({med_lead['persistence']:.1f}) and linear "
                f"({med_lead['linear']:.1f}) at matched FAR.")
        else:
            verdicts.append(
                f"FAR target {target:.2f}: NO-SKILL — faircast median "
                f"lead {med_lead['faircast']:.1f} does not exceed both "
                f"trivial forecasters (persistence "
                f"{med_lead['persistence']:.1f}, linear "
                f"{med_lead['linear']:.1f}) at matched FAR.")
    return BenchmarkResult(table=pd.DataFrame(rows), verdicts=verdicts,
                           brier=brier, n_series=len(panels),
                           n_breaching=n_breaching,
                           n_already_breached=n_already)
