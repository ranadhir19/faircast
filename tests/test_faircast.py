"""FairCast test suite.

Groups: (1) input validation — every in-domain-but-invalid input must
raise FairCastError with a specific message, never be silently
repaired; (2) metric correctness on hand-computed cases; (3) forecast
behaviour on planted ground truth; (4) near-miss negatives (seasonal
oscillation is not trend, tiny-n widens intervals, never-breach panels
stay inside the alarm budget); (5) certificate algebra; (6) benchmark
honesty (unforecastable jumps yield NO-SKILL); (7) CLI smoke.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import faircast as fc
from faircast import FairCastError
from faircast.certificate import tv_hist
from faircast.conformal import calibrate, truncate_panel
from faircast.data import SeriesPanel
from faircast.synthetic import (never_breach_panels, planted_breach_panels,
                                seasonal_panels, tiny_n_panel)


def counts_df(periods=6, groups=("A", "B"), n=1000, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(periods):
        for g in groups:
            p = 0.5 if g == "A" else 0.48
            rows.append({"period": t, "group": g, "n": n,
                         "positives": int(rng.binomial(n, p))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- validation
def test_counts_missing_column():
    df = counts_df().drop(columns=["positives"])
    with pytest.raises(FairCastError, match="missing column"):
        fc.load_counts(df)


def test_counts_nan_rejected():
    df = counts_df()
    df.loc[0, "positives"] = np.nan
    with pytest.raises(FairCastError, match="missing values"):
        fc.load_counts(df)


def test_counts_duplicate_cell_rejected():
    df = pd.concat([counts_df(), counts_df().iloc[[0]]],
                   ignore_index=True)
    with pytest.raises(FairCastError, match="duplicate"):
        fc.load_counts(df)


def test_counts_group_missing_period_rejected():
    df = counts_df()
    df = df[~((df.group == "B") & (df.period == 3))]
    with pytest.raises(FairCastError, match="missing period"):
        fc.load_counts(df)


def test_counts_nonpositive_n_rejected():
    df = counts_df()
    df.loc[0, "n"] = 0
    with pytest.raises(FairCastError, match="undefined"):
        fc.load_counts(df)


def test_counts_positives_above_n_rejected():
    df = counts_df()
    df.loc[0, "positives"] = df.loc[0, "n"] + 1
    with pytest.raises(FairCastError, match="outside"):
        fc.load_counts(df)


def test_counts_too_few_periods_rejected():
    with pytest.raises(FairCastError, match="at least"):
        fc.load_counts(counts_df(periods=3))


def decisions_df():
    rows = []
    rng = np.random.default_rng(5)
    for t in range(5):
        for g in ("A", "B"):
            for y in (0, 1):
                for _ in range(40):
                    rows.append({"period": t, "group": g, "y": y,
                                 "y_pred": int(rng.random() < 0.7)})
    return pd.DataFrame(rows)


def test_decisions_nonbinary_rejected():
    df = decisions_df()
    df.loc[0, "y"] = 2
    with pytest.raises(FairCastError, match="binary"):
        fc.load_decisions(df)


def test_decisions_eo_needs_both_classes():
    df = decisions_df()
    df = df[~((df.group == "B") & (df.period == 2) & (df.y == 1))]
    with pytest.raises(FairCastError, match="no rows with"):
        fc.load_decisions(df)


def test_series_duplicate_period_rejected():
    df = pd.DataFrame({"period": [0, 1, 1, 2, 3],
                       "value": [0.01, 0.02, 0.03, 0.02, 0.01]})
    with pytest.raises(FairCastError, match="duplicate"):
        fc.load_series(df)


def test_series_nonpositive_se_rejected():
    df = pd.DataFrame({"period": range(5), "value": [0.01] * 5,
                       "se": [0.01, 0.01, 0.0, 0.01, 0.01]})
    with pytest.raises(FairCastError, match="standard errors"):
        fc.load_series(df, se_col="se")


def test_unknown_metric_rejected():
    with pytest.raises(FairCastError, match="unknown metric"):
        fc.Forecaster(metric="accuracy_parity")


def test_output_before_fit_rejected():
    with pytest.raises(FairCastError, match="call fit"):
        fc.Forecaster().breach_table()


# ------------------------------------------------------------------- metrics
def _panel_from_rates():
    rows = []
    for t in range(4):
        rows.append({"period": t, "group": "A", "n": 200,
                     "positives": 120})   # rate 0.60
        rows.append({"period": t, "group": "B", "n": 200,
                     "positives": 90})    # rate 0.45
    return fc.load_counts(pd.DataFrame(rows))


def test_metric_values_hand_computed():
    panel = _panel_from_rates()
    dp = panel.metric_series(fc.METRICS["dp_gap"])
    di = panel.metric_series(fc.METRICS["di_ratio"])
    assert np.allclose(dp, 0.15)
    assert np.allclose(di, 0.45 / 0.60)


def test_comparison_metric_needs_two_groups():
    df = counts_df(groups=("A",))
    panel = fc.load_counts(df)
    with pytest.raises(FairCastError, match="at least 2"):
        panel.metric_series(fc.METRICS["dp_gap"])
    sr = panel.metric_series(fc.METRICS["selection_rate"])
    assert len(sr) == 6


def test_di_ratio_breach_direction_is_below():
    f = fc.Forecaster(metric="di_ratio", threshold=0.8)
    assert f.direction == "below"


# ------------------------------------------------------------------ forecast
def test_planted_breach_is_forecast():
    frames, breaches = planted_breach_panels(n_series=1, seed=7)
    panel = fc.load_counts(frames[0])
    origin = breaches[0] - 2
    f = fc.Forecaster(metric="dp_gap", threshold=0.10, horizon=4,
                      n_paths=2000).fit(truncate_panel(panel, origin))
    tab = f.breach_table()
    # cumulative P(breach) at the horizon reaching the planted breach
    h_breach = breaches[0] - origin + 1
    p_at = float(tab.p_breach_by_horizon.iloc[h_breach - 1])
    assert p_at >= 0.40, f"P(breach by true breach period) = {p_at}"
    dist = f.breach_time_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert f.median_breach_horizon() is not None


def test_fan_quantiles_ordered_and_bounded():
    frames, _ = planted_breach_panels(n_series=1, seed=7)
    panel = fc.load_counts(frames[0])
    f = fc.Forecaster(metric="dp_gap", threshold=0.10, horizon=3,
                      n_paths=1500).fit(panel)
    fan = f.fan()
    assert (fan.q05 <= fan.q25).all() and (fan.q25 <= fan.q50).all()
    assert (fan.q50 <= fan.q75).all() and (fan.q75 <= fan.q95).all()
    assert fan.q05.min() >= 0.0 and fan.q95.max() <= 1.0


def test_sampling_noise_option_valid():
    frames, _ = planted_breach_panels(n_series=1, seed=7)
    panel = fc.load_counts(frames[0])
    f = fc.Forecaster(metric="dp_gap", threshold=0.10, horizon=3,
                      n_paths=1500, sampling_noise=True).fit(panel)
    tab = f.breach_table()
    assert tab.p_breach_by_horizon.between(0, 1).all()


def test_di_ratio_forecast_breach():
    frames, _ = planted_breach_panels(n_series=1, seed=7)
    panel = fc.load_counts(frames[0])
    f = fc.Forecaster(metric="di_ratio", threshold=0.8, horizon=4,
                      n_paths=2000).fit(truncate_panel(panel, 7))
    p = float(f.breach_table().p_breach_by_horizon.iloc[-1])
    assert p > 0.5, f"di_ratio breach probability {p}"


# ------------------------------------------------------------- near-miss set
def test_seasonal_oscillation_is_not_a_trend():
    """The latent gap oscillates in [0.02, 0.07] around 0.045 with zero
    trend (tau = 0.10). 'Not read as trend' is asserted on the central
    forecast, which is what a trend-reader gets wrong: the median path
    must stay well clear of the threshold, the immediate breach
    probability must stay small, and even the cumulative probability —
    which grows mechanically as the fan widens — must not call a
    breach more likely than not."""
    frames, _ = seasonal_panels(n_series=1, seed=13)
    panel = fc.load_counts(frames[0])
    f = fc.Forecaster(metric="dp_gap", threshold=0.10, horizon=6,
                      n_paths=2000).fit(panel)
    tab = f.breach_table()
    fan = f.fan()
    assert float(fan.q50.iloc[-1]) < 0.075, fan.q50.iloc[-1]
    assert float(tab.p_first_breach_in_period.iloc[0]) < 0.10
    p_cum = float(tab.p_breach_by_horizon.iloc[-1])
    assert p_cum < 0.50, f"seasonal panel called breach-likely: {p_cum}"


def test_tiny_n_intervals_are_wide():
    small = fc.load_counts(tiny_n_panel(n_per=25))
    big = fc.load_counts(tiny_n_panel(n_per=20000))
    kw = dict(metric="dp_gap", threshold=0.10, horizon=3, n_paths=2000)
    fs = fc.Forecaster(**kw).fit(truncate_panel(small, 5))
    fb = fc.Forecaster(**kw).fit(truncate_panel(big, 5))
    w_small = float((fs.fan().q95 - fs.fan().q05).iloc[1])
    w_big = float((fb.fan().q95 - fb.fan().q05).iloc[1])
    assert w_small > 1.8 * w_big, (w_small, w_big)


def test_never_breach_alarm_within_budget():
    frames, _ = never_breach_panels(n_series=1, seed=11)
    panel = fc.load_counts(frames[0])
    design = fc.design_alarm(
        lambda: fc.Forecaster(metric="dp_gap", threshold=0.10, horizon=2,
                              n_paths=2000),
        panel, policy_horizon=2, target_far=0.10)
    assert design.far_estimable
    assert design.backtest_far <= 0.20, design.backtest_far
    assert design.arl0 is not None and design.arl0 >= 5.0


def test_alarm_design_refuses_always_breaching_panel():
    # A series above the threshold at every period: every backtest
    # origin is followed by a breach, so no FAR can be designed.
    panel = SeriesPanel(periods=list(range(8)),
                        values=np.full(8, 0.20), ses=None)
    with pytest.raises(FairCastError, match="always breaching"):
        fc.design_alarm(
            lambda: fc.Forecaster(metric="dp_gap", threshold=0.10,
                                  horizon=2, n_paths=1500),
            panel, policy_horizon=2, target_far=0.10)


def test_calibration_runs_and_notes_itself():
    frames, _ = planted_breach_panels(n_series=1, n_periods=12, seed=7)
    panel = fc.load_counts(frames[0])
    f = fc.Forecaster(metric="dp_gap", threshold=0.10, horizon=2,
                      n_paths=1500).fit(panel)
    calibrate(f, panel)
    assert "calibrat" in f.calibration_note.lower()


# --------------------------------------------------------------- certificate
def test_certificate_zero_violations_on_random_exact_panels():
    rng = np.random.default_rng(21)
    for _ in range(12):
        rows = []
        base = {g: rng.uniform(0.2, 0.7) for g in ("A", "B")}
        for t in range(6):
            for g in ("A", "B"):
                p = float(np.clip(base[g] + rng.normal(0, 0.03), 0.05, 0.95))
                n = int(rng.integers(3000, 6000))
                rows.append({"period": t, "group": g, "n": n,
                             "positives": int(rng.binomial(n, p))})
        panel = fc.load_counts(pd.DataFrame(rows))
        metric = fc.METRICS["dp_gap"]  # counts panels carry sel cells
        values = panel.metric_series(metric)
        eps = fc.rate_budget(panel, kinds=metric.kinds)
        v = fc.verify(values, eps, direction="above")
        assert v["violations"] == 0, v


def test_verify_detects_a_shrunken_budget():
    frames, _ = planted_breach_panels(n_series=1, seed=7)
    panel = fc.load_counts(frames[0])
    metric = fc.METRICS["dp_gap"]
    values = panel.metric_series(metric)
    eps = fc.rate_budget(panel, kinds=metric.kinds)
    v = fc.verify(values, eps * 0.05)
    assert v["violations"] > 0  # verify() is not vacuous


def test_certify_safe_horizon_shrinks_to_zero():
    frames, _ = planted_breach_panels(n_series=1, seed=7)
    panel = fc.load_counts(frames[0])
    metric = fc.METRICS["dp_gap"]
    values = panel.metric_series(metric)
    eps = fc.rate_budget(panel, kinds=metric.kinds)
    table = fc.certify(values, eps, 0.10, periods=panel.periods)
    assert int(table.certified_safe_horizon.iloc[0]) >= 1
    assert int(table.certified_safe_horizon.iloc[-1]) == 0


def test_tv_hist_properties():
    rng = np.random.default_rng(0)
    a = rng.random(5000)
    b = rng.random(5000)
    assert tv_hist(a, a) == pytest.approx(0.0, abs=1e-12)
    low = rng.uniform(0.0, 0.4, 4000)
    high = rng.uniform(0.6, 1.0, 4000)
    assert tv_hist(low, high) > 0.95
    assert 0.0 <= tv_hist(a, b) <= 1.0


# ----------------------------------------------------------------- benchmark
def _jump_panel(seed, jump=True):
    """Flat series whose last period jumps over the threshold: a level
    shift with no precursor — unforecastable by construction."""
    rng = np.random.default_rng(seed)
    vals = 0.030 - 0.0015 * np.arange(8) + rng.normal(0, 0.0005, 8)
    if jump:
        vals[-1] = 0.09
    return SeriesPanel(periods=list(range(8)), values=vals, ses=None)


def test_benchmark_no_skill_on_unforecastable_jumps():
    panels = [_jump_panel(s, jump=(s % 2 == 0)) for s in range(6)]
    res = fc.benchmark_panels(panels, metric="dp_gap", threshold=0.05,
                              horizon=2, n_paths=1200)
    assert res.n_breaching == 3
    assert all("NO-SKILL" in v or "not assessable" in v
               for v in res.verdicts), res.verdicts
    assert set(res.table.method) == {"faircast", "persistence", "linear"}


def test_benchmark_rejects_empty_panel_list():
    with pytest.raises(FairCastError, match="at least one"):
        fc.benchmark_panels([])


# ----------------------------------------------------------------------- CLI
def test_cli_forecast_and_certify(tmp_path):
    from faircast.cli import main
    frames, _ = planted_breach_panels(n_series=1, seed=7)
    csv = tmp_path / "counts.csv"
    frames[0].to_csv(csv, index=False)
    assert main(["forecast", str(csv), "--metric", "dp_gap",
                 "--threshold", "0.10", "--horizon", "3"]) == 0
    assert main(["certify", str(csv), "--metric", "dp_gap",
                 "--threshold", "0.10"]) == 0
    # di_ratio cannot be certified — named refusal, exit code 2
    assert main(["certify", str(csv), "--metric", "di_ratio",
                 "--threshold", "0.8"]) == 2


def test_cli_benchmark_on_bundled_slice(tmp_path):
    from faircast.cli import main
    from faircast.data import bundled_acs_path
    out = tmp_path / "bench.md"
    assert main(["benchmark", str(bundled_acs_path()),
                 "--series-cols", "task,state", "--period-col", "year",
                 "--value-col", "eo_gap", "--metric", "eo_gap",
                 "--threshold", "0.05", "--out", str(out)]) == 0
    assert "NO-SKILL" in out.read_text() or "SKILL" in out.read_text()
