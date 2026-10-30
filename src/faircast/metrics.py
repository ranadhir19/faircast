"""Fairness metrics as functions of per-cell outcome rates.

A *cell* is one (group, kind) rate series, e.g. the selection rate of group
"A" (kind ``"sel"``) or its true-positive rate (kind ``"tpr"``). A metric is
a pure function of one period's cell rates. Forecasting models the cells and
propagates uncertainty *through* the metric, instead of fitting a model to
the ratio/gap directly — a gap is a non-smooth function of noisy rates, and
modelling it as if it were a clean series is how confident nonsense happens.

Breach semantics: a metric breaches when it crosses ``threshold`` in its
breach direction — ``"above"`` for gaps (breach at ``metric >= threshold``),
``"below"`` for the disparate-impact ratio (breach at
``metric <= threshold``, e.g. the four-fifths rule at 0.8).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .errors import FairCastError


@dataclass(frozen=True)
class Metric:
    name: str
    kinds: tuple[str, ...]          # cell kinds the metric consumes
    bounds: tuple[float, float]     # plausible range of the metric value
    default_direction: str          # "above" | "below"

    def compute(self, rates: dict[tuple[str, str], float],
                groups: list[str]) -> float:
        return _COMPUTE[self.name](rates, groups)

    def validate_groups(self, groups: list[str]) -> None:
        if self.name == "selection_rate":
            if len(groups) < 1:
                raise FairCastError("selection_rate needs at least one group")
            return
        if len(groups) < 2:
            raise FairCastError(
                f"metric {self.name!r} compares groups and needs at least 2; "
                f"got {len(groups)} ({groups}). A single-group series has no "
                "fairness gap to forecast.")


def _sel(rates, g):
    return rates[(g, "sel")]


def _dp_gap(rates, groups):
    vals = [_sel(rates, g) for g in groups]
    return max(abs(a - b) for a, b in combinations(vals, 2))


def _di_ratio(rates, groups):
    vals = [_sel(rates, g) for g in groups]
    hi = max(vals)
    if hi <= 0.0:
        return 1.0  # no selections anywhere: no disparity in selection
    return min(vals) / hi


def _eo_gap(rates, groups):
    worst = 0.0
    for kind in ("tpr", "fpr"):
        vals = [rates[(g, kind)] for g in groups]
        worst = max(worst, max(abs(a - b) for a, b in combinations(vals, 2)))
    return worst


def _selection_rate(rates, groups):
    return _sel(rates, groups[0])


_COMPUTE = {
    "dp_gap": _dp_gap,
    "di_ratio": _di_ratio,
    "eo_gap": _eo_gap,
    "selection_rate": _selection_rate,
}

METRICS: dict[str, Metric] = {
    "dp_gap": Metric("dp_gap", ("sel",), (0.0, 1.0), "above"),
    "di_ratio": Metric("di_ratio", ("sel",), (0.0, 1.0), "below"),
    "eo_gap": Metric("eo_gap", ("tpr", "fpr"), (0.0, 1.0), "above"),
    "selection_rate": Metric("selection_rate", ("sel",), (0.0, 1.0), "above"),
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise FairCastError(
            f"unknown metric {name!r}; available: {sorted(METRICS)}") from None
