"""Input loading and validation.

FairCast accepts three input shapes:

1. **Tidy counts** — one row per (period, group) with columns
   ``period, group, n, positives``: the number of decisions and the number
   of positive outcomes for that group in that period. This is the primary
   contract; selection rates are ``positives / n``.
2. **Decisions** — one row per decision with ``period, group, y, y_pred``;
   aggregated internally into selection / true-positive / false-positive
   rate cells (for the EO gap).
3. **Metric series** — one row per period with ``period, value`` (and an
   optional ``se`` column): an already-computed fairness metric over time,
   e.g. a gap exported from an existing audit pipeline.

Validation is strict and specific. A group missing a period, ``n = 0``, 
``positives > n``, duplicated periods, NaNs, or fewer than
:data:`MIN_PERIODS` periods each raise :class:`FairCastError` naming the
offending rows. Nothing is dropped, filled, or repaired silently — a
fairness series with a hole in it is a finding, not an inconvenience.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import FairCastError

MIN_PERIODS = 4


@dataclass
class CellSeries:
    """One (group, kind) rate series aligned to the panel's periods."""

    group: str
    kind: str
    n: np.ndarray          # denominators per period (float, > 0)
    positives: np.ndarray  # numerators per period

    @property
    def rate(self) -> np.ndarray:
        return self.positives / self.n


@dataclass
class Panel:
    """Validated grouped outcome data over time."""

    periods: list
    groups: list[str]
    cells: dict[tuple[str, str], CellSeries]
    source_kind: str  # "counts" | "decisions"

    def cell(self, group: str, kind: str) -> CellSeries:
        try:
            return self.cells[(group, kind)]
        except KeyError:
            raise FairCastError(
                f"panel has no cell (group={group!r}, kind={kind!r}); "
                f"available kinds: {sorted({k for _, k in self.cells})}. "
                "An EO-gap forecast needs a decisions input (with y and "
                "y_pred); a counts input only carries selection rates."
            ) from None

    def metric_series(self, metric) -> np.ndarray:
        """Observed metric value per period, from the cell rates."""
        metric.validate_groups(self.groups)
        vals = []
        for i in range(len(self.periods)):
            rates = {key: cs.rate[i] for key, cs in self.cells.items()}
            vals.append(metric.compute(rates, self.groups))
        return np.asarray(vals, dtype=float)


@dataclass
class SeriesPanel:
    """An already-computed metric series over time."""

    periods: list
    values: np.ndarray
    ses: np.ndarray | None  # per-period standard errors, if supplied

    def metric_series(self, metric) -> np.ndarray:
        return self.values


def _read(source) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        try:
            return pd.read_csv(source)
        except Exception as exc:  # file missing / unparsable
            raise FairCastError(f"could not read CSV {source!r}: {exc}") from exc
    if isinstance(source, pd.DataFrame):
        return source.copy()
    raise FairCastError(
        f"expected a CSV path or a pandas DataFrame, got {type(source).__name__}")


def _require_columns(df: pd.DataFrame, cols: list[str], shape: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise FairCastError(
            f"{shape} input is missing column(s) {missing}; "
            f"found columns {list(df.columns)}")


def _check_periods(periods: list, who: str) -> None:
    if len(periods) != len(set(periods)):
        raise FairCastError(f"{who}: duplicate periods in {periods}")
    if len(periods) < MIN_PERIODS:
        raise FairCastError(
            f"{who}: only {len(periods)} period(s) {periods}; at least "
            f"{MIN_PERIODS} are required to fit a trend and propagate "
            "uncertainty honestly. Fewer points can describe a metric, "
            "they cannot forecast one.")


def _as_count(x, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise FairCastError(f"{what}: value {x!r} is not a number") from None
    if not np.isfinite(v):
        raise FairCastError(f"{what}: value is NaN or infinite")
    return v


def load_counts(source, period_col="period", group_col="group",
                n_col="n", positives_col="positives") -> Panel:
    """Load tidy counts (period, group, n, positives) into a Panel."""
    df = _read(source)
    _require_columns(df, [period_col, group_col, n_col, positives_col],
                     "counts")
    df = df[[period_col, group_col, n_col, positives_col]]
    if df.isna().any().any():
        bad = df[df.isna().any(axis=1)].head(3).to_dict("records")
        raise FairCastError(f"counts input contains missing values: {bad}")
    dup = df.duplicated(subset=[period_col, group_col])
    if dup.any():
        rows = df[dup].head(3).to_dict("records")
        raise FairCastError(
            f"counts input has duplicate (period, group) rows: {rows}. "
            "Aggregate them before loading; FairCast will not guess how.")

    periods = sorted(df[period_col].unique().tolist())
    groups = sorted(df[group_col].astype(str).unique().tolist())
    _check_periods(periods, "counts input")
    cells: dict[tuple[str, str], CellSeries] = {}
    for g in groups:
        sub = df[df[group_col].astype(str) == g].set_index(period_col)
        missing = [p for p in periods if p not in sub.index]
        if missing:
            raise FairCastError(
                f"group {g!r} is missing period(s) {missing}; every group "
                "must be observed in every period. A gap in a fairness "
                "series must be explained by the caller, not filled in.")
        ns, pos = [], []
        for p in periods:
            n = _as_count(sub.loc[p, n_col], f"n for group {g!r}, period {p}")
            k = _as_count(sub.loc[p, positives_col],
                          f"positives for group {g!r}, period {p}")
            if n <= 0:
                raise FairCastError(
                    f"group {g!r}, period {p}: n = {n:g}; the rate is "
                    "undefined for an empty group-period. Remove the "
                    "period or explain the absence — it is not dropped "
                    "silently.")
            if k < 0 or k > n:
                raise FairCastError(
                    f"group {g!r}, period {p}: positives = {k:g} outside "
                    f"[0, n = {n:g}]; the counts are inconsistent.")
            ns.append(n)
            pos.append(k)
        cells[(g, "sel")] = CellSeries(g, "sel", np.array(ns), np.array(pos))
    return Panel(periods, groups, cells, "counts")


def load_decisions(source, period_col="period", group_col="group",
                   y_col="y", yhat_col="y_pred") -> Panel:
    """Load individual decisions (period, group, y, y_pred) into a Panel
    with selection, TPR and FPR cells per group."""
    df = _read(source)
    _require_columns(df, [period_col, group_col, y_col, yhat_col], "decisions")
    df = df[[period_col, group_col, y_col, yhat_col]]
    if df.isna().any().any():
        raise FairCastError("decisions input contains missing values")
    for col in (y_col, yhat_col):
        vals = set(df[col].unique().tolist())
        if not vals.issubset({0, 1, 0.0, 1.0, True, False}):
            raise FairCastError(
                f"decisions column {col!r} must be binary 0/1; "
                f"found values {sorted(map(str, vals))[:6]}")
    df[group_col] = df[group_col].astype(str)
    periods = sorted(df[period_col].unique().tolist())
    groups = sorted(df[group_col].unique().tolist())
    _check_periods(periods, "decisions input")

    cells: dict[tuple[str, str], CellSeries] = {}
    for g in groups:
        sub_g = df[df[group_col] == g]
        present = set(sub_g[period_col].unique().tolist())
        missing = [p for p in periods if p not in present]
        if missing:
            raise FairCastError(
                f"group {g!r} has no decisions in period(s) {missing}; "
                "every group must be observed in every period.")
        store = {k: ([], []) for k in ("sel", "tpr", "fpr")}
        for p in periods:
            sub = sub_g[sub_g[period_col] == p]
            y = sub[y_col].to_numpy(dtype=float)
            yh = sub[yhat_col].to_numpy(dtype=float)
            store["sel"][0].append(len(sub))
            store["sel"][1].append(float(yh.sum()))
            for kind, mask in (("tpr", y == 1), ("fpr", y == 0)):
                n = int(mask.sum())
                if n == 0:
                    raise FairCastError(
                        f"group {g!r}, period {p}: no rows with "
                        f"y = {1 if kind == 'tpr' else 0}, so the "
                        f"{kind.upper()} is undefined. The EO gap cannot "
                        "be computed for this panel as given.")
                store[kind][0].append(n)
                store[kind][1].append(float(yh[mask].sum()))
        for kind, (ns, pos) in store.items():
            cells[(g, kind)] = CellSeries(
                g, kind, np.array(ns, dtype=float), np.array(pos, dtype=float))
    return Panel(periods, groups, cells, "decisions")


def load_series(source, period_col="period", value_col="value",
                se_col="se") -> SeriesPanel:
    """Load an already-computed metric series (period, value[, se])."""
    df = _read(source)
    _require_columns(df, [period_col, value_col], "series")
    if df.duplicated(subset=[period_col]).any():
        raise FairCastError("series input has duplicate periods")
    df = df.sort_values(period_col)
    periods = df[period_col].tolist()
    _check_periods(periods, "series input")
    values = df[value_col].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise FairCastError("series input contains NaN or infinite values")
    ses = None
    if se_col in df.columns and df[se_col].notna().all():
        ses = df[se_col].to_numpy(dtype=float)
        if (ses <= 0).any():
            raise FairCastError("series standard errors must be positive")
    return SeriesPanel(periods, values, ses)


def bundled_acs_path() -> Path:
    """Path of the bundled ACS population-gap slice (folktables / Paper 11).

    18 trajectories (3 ACS tasks x 6 states), years 2014-2019, population
    EO and DP gaps of a frozen 2014-trained logistic-regression model, as
    computed by the Paper 11 pipeline from the folktables benchmark
    (Ding et al.), which is derived from the American Community Survey
    published by the US Census Bureau.
    """
    return Path(str(resources.files("faircast").joinpath(
        "data", "acs_population_gaps.csv")))
