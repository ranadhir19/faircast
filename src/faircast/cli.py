"""Command line: ``faircast forecast | benchmark | certify``.

Examples
--------
Forecast breach risk for a counts file::

    faircast forecast counts.csv --format counts --metric dp_gap \
        --threshold 0.05 --horizon 6 --target-far 0.10 \
        --out report.md --fan fan.png

Self-benchmark on a long table of metric series (one row per
series/period)::

    faircast benchmark acs.csv --series-cols task,state \
        --value-col eo_gap --metric eo_gap --threshold 0.05

Certify a series with the drift-budget certificate::

    faircast certify counts.csv --metric dp_gap --threshold 0.05
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .alarm import design_alarm
from .benchmark import benchmark_panels
from .certificate import certify, rate_budget, verify
from .conformal import calibrate
from .data import load_counts, load_decisions, load_series
from .errors import FairCastError
from .forecast import Forecaster
from .metrics import get_metric
from .report import fan_chart, write_report


def _load_panel(args):
    if args.format == "counts":
        return load_counts(args.input)
    if args.format == "decisions":
        return load_decisions(args.input)
    return load_series(args.input, value_col=args.value_col)


def _make(args):
    return Forecaster(metric=args.metric, threshold=args.threshold,
                      direction=args.direction, horizon=args.horizon,
                      seed=args.seed)


def cmd_forecast(args) -> int:
    panel = _load_panel(args)
    fc = _make(args).fit(panel)
    if args.calibrate:
        calibrate(fc, panel)
    alarm = None
    if args.target_far is not None:
        proto = _make(args)
        alarm = design_alarm(lambda: proto.__class__(
            metric=args.metric, threshold=args.threshold,
            direction=args.direction, horizon=args.horizon, seed=args.seed),
            panel, policy_horizon=args.policy_horizon,
            target_far=args.target_far)
    s = fc.summary()
    print(f"metric={s['metric']} threshold={s['threshold']:g} "
          f"current={s['current_value']:.4f} "
          f"P(breach within {s['horizon']})="
          f"{s['p_breach_by_final_horizon']:.3f}")
    print(fc.breach_table().to_string(index=False))
    print(f"intervals: {s['calibration_note']}")
    fan_png = None
    if args.fan:
        fan_chart(fc, args.fan)
        fan_png = args.fan
    if args.out:
        write_report(fc, args.out, alarm=alarm, fan_png=fan_png)
        print(f"report written to {args.out}")
    if alarm is not None:
        print(f"alarm: theta={alarm.threshold_on_probability:.3f} "
              f"target FAR={alarm.target_far:.2f} backtest FAR="
              f"{alarm.backtest_far} ARL0={alarm.arl0}")
    return 0


def cmd_benchmark(args) -> int:
    df = pd.read_csv(args.input)
    series_cols = [c.strip() for c in args.series_cols.split(",")]
    panels = []
    for _, sub in df.groupby(series_cols):
        sub = sub.sort_values(args.period_col)
        from .data import SeriesPanel
        panels.append(SeriesPanel(
            periods=sub[args.period_col].tolist(),
            values=sub[args.value_col].to_numpy(dtype=float), ses=None))
    res = benchmark_panels(panels, metric=args.metric,
                           threshold=args.threshold,
                           direction=args.direction, horizon=args.horizon,
                           seed=args.seed)
    md = res.to_markdown()
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md + "\n")
    return 0


def cmd_certify(args) -> int:
    panel = load_counts(args.input)
    metric = get_metric(args.metric)
    if metric.name == "di_ratio":
        raise FairCastError(
            "the drift-budget certificate does not apply to di_ratio: a "
            "ratio is not Lipschitz in the budget when the denominator "
            "rate is near zero. Certify dp_gap instead (same input).")
    values = panel.metric_series(metric)
    eps = rate_budget(panel, kinds=metric.kinds, noise_z=args.noise_z)
    v = verify(values, eps, direction=args.direction
               or metric.default_direction)
    print(f"certificate verification on observed history: "
          f"{v['checks']} checks, {v['violations']} violations "
          f"(noise_z={args.noise_z:g})")
    table = certify(values, eps, args.threshold,
                    direction=args.direction or metric.default_direction,
                    periods=panel.periods)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="faircast", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("forecast", help="forecast breach risk")
    pf.add_argument("input")
    pf.add_argument("--format", choices=["counts", "decisions", "series"],
                    default="counts")
    pf.add_argument("--value-col", default="value")
    pf.add_argument("--metric", default="dp_gap")
    pf.add_argument("--threshold", type=float, default=0.05)
    pf.add_argument("--direction", choices=["above", "below"], default=None)
    pf.add_argument("--horizon", type=int, default=6)
    pf.add_argument("--seed", type=int, default=12345)
    pf.add_argument("--calibrate", action="store_true")
    pf.add_argument("--target-far", type=float, default=None)
    pf.add_argument("--policy-horizon", type=int, default=2)
    pf.add_argument("--out", default=None)
    pf.add_argument("--fan", default=None)
    pf.set_defaults(fn=cmd_forecast)

    pb = sub.add_parser("benchmark", help="FAR-matched self-benchmark")
    pb.add_argument("input")
    pb.add_argument("--series-cols", required=True)
    pb.add_argument("--period-col", default="year")
    pb.add_argument("--value-col", required=True)
    pb.add_argument("--metric", default="eo_gap")
    pb.add_argument("--threshold", type=float, default=0.05)
    pb.add_argument("--direction", choices=["above", "below"], default=None)
    pb.add_argument("--horizon", type=int, default=2)
    pb.add_argument("--seed", type=int, default=12345)
    pb.add_argument("--out", default=None)
    pb.set_defaults(fn=cmd_benchmark)

    pc = sub.add_parser("certify", help="drift-budget certificate")
    pc.add_argument("input")
    pc.add_argument("--metric", default="dp_gap")
    pc.add_argument("--threshold", type=float, default=0.05)
    pc.add_argument("--direction", choices=["above", "below"], default=None)
    pc.add_argument("--noise-z", type=float, default=0.0)
    pc.add_argument("--out", default=None)
    pc.set_defaults(fn=cmd_certify)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except FairCastError as exc:
        print(f"faircast: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
