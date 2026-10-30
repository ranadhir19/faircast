"""Worked example on synthetic counts: the full pipeline end to end.

A planted-breach counts panel (known ground truth) exercises the parts
of FairCast the ACS slice cannot: counts loading, an alarm policy
designed to a target false-alarm rate, and the drift-budget
certificate — which on these observed counts must verify with exactly
0 violations, because for gap metrics the budget bound is algebra, not
an estimate.

Run:  python examples/synthetic_counts_example.py
Writes examples/synthetic_report.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faircast as fc  # noqa: E402
from faircast.conformal import truncate_panel  # noqa: E402
from faircast.report import write_report  # noqa: E402
from faircast.synthetic import (never_breach_panels,  # noqa: E402
                                planted_breach_panels)

TAU = 0.10


def main() -> None:
    frames, breaches = planted_breach_panels(n_series=4, seed=7)
    panel = fc.load_counts(frames[0])
    print(f"planted latent breach period: {breaches[0]}")

    # forecast from two periods before the planted breach
    origin = breaches[0] - 2
    f = fc.Forecaster(metric="dp_gap", threshold=TAU, horizon=4).fit(
        truncate_panel(panel, origin))
    print(f.breach_table().to_string(index=False))

    # Alarm policy designed on a panel that drifts toward the threshold
    # but never breaches — the setting a false-alarm budget exists for.
    # (On the planted panel above, every backtest origin is followed by
    # a breach, so design_alarm refuses to quote a FAR there; that
    # refusal is itself part of the demo if you point it at frames[0].)
    calm_frames, _ = never_breach_panels(n_series=1, seed=11)
    calm = fc.load_counts(calm_frames[0])
    alarm = fc.design_alarm(
        lambda: fc.Forecaster(metric="dp_gap", threshold=TAU, horizon=2),
        calm, policy_horizon=2, target_far=0.10)
    print(f"alarm: theta={alarm.threshold_on_probability:.3f} "
          f"target FAR={alarm.target_far:.2f} backtest FAR="
          f"{alarm.backtest_far} ARL0={alarm.arl0} "
          f"estimable={alarm.far_estimable}")

    # certificate: the observed gap path must respect the drift budget
    values = panel.metric_series(fc.METRICS["dp_gap"])
    eps = fc.rate_budget(panel, kinds=("sel",))
    v = fc.verify(values, eps)
    print(f"certificate: {v['checks']} checks, {v['violations']} "
          f"violations (must be 0)")
    print(fc.certify(values, eps, TAU, periods=panel.periods)
          .to_string(index=False))

    write_report(f, Path(__file__).resolve().parent / "synthetic_report.md",
                 alarm=alarm)
    print("report written to examples/synthetic_report.md")


if __name__ == "__main__":
    main()
