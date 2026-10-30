"""Human-readable outputs: fan charts and Markdown forecast reports.

The report template carries a fixed honesty box. It is not decorative:
Paper 11 falsified its own headline claim under a pre-registered
protocol, and a tool that productises that work does not get to present
a breach probability without stating what the number is and is not.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .alarm import AlarmDesign  # noqa: E402

HONESTY_BOX = """\
> **What this forecast is, and is not.** The breach probabilities come
> from a per-cell state-space model of the underlying group rates,
> propagated through the metric. They are *not* a promised lead time:
> in Paper 11 (Ghosh, submitted to AI & Society, 2026), a forecaster of
> this family was pre-registered against persistence and linear
> extrapolation at matched false-alarm rates and **did not beat
> extrapolation on median lead** — the claim was reported falsified.
> Run `faircast benchmark` on your own panels; FairCast prints
> NO-SKILL whenever it does not beat both trivial forecasters. Alarm
> budgets likewise do not transport for free: thresholds fitted to a
> 0.10 false-alarm target on one jurisdiction fired at 0.293 on others
> in Paper 11's corpus, so recalibrate per series and read the
> *achieved* backtest rate, not the target.
"""


def fan_chart(forecaster, path) -> Path:
    """Observed metric + forecast fan (50% and 90% bands) + threshold."""
    fan = forecaster.fan()
    obs_x = list(range(len(forecaster.periods_)))
    fx = [len(forecaster.periods_) - 1 + h for h in fan["horizon"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(obs_x, forecaster.observed_, "o-", color="black", lw=1.6,
            label="observed")
    ax.fill_between(fx, fan["q05"], fan["q95"], color="#4C72B0",
                    alpha=0.18, label="90% band")
    ax.fill_between(fx, fan["q25"], fan["q75"], color="#4C72B0",
                    alpha=0.35, label="50% band")
    ax.plot(fx, fan["q50"], "-", color="#4C72B0", lw=1.6,
            label="forecast median")
    ax.axhline(forecaster.threshold, color="#C44E52", lw=1.4,
               ls="--", label=f"threshold {forecaster.threshold:g}")
    ax.set_xlabel("period index")
    ax.set_ylabel(forecaster.metric.name)
    ax.set_title("FairCast forecast fan")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(forecaster, path, alarm: AlarmDesign | None = None,
                 benchmark_md: str | None = None,
                 fan_png: str | None = None) -> Path:
    s = forecaster.summary()
    table = forecaster.breach_table()
    dist = forecaster.breach_time_distribution()
    lines = [
        "# FairCast forecast report",
        "",
        f"- Metric: **{s['metric']}**, threshold {s['threshold']:g} "
        f"(breach direction: {s['direction']})",
        f"- Last observed period: {s['last_period']} — current value "
        f"{s['current_value']:.4f}",
        f"- P(breach within {s['horizon']} periods): "
        f"**{s['p_breach_by_final_horizon']:.3f}**",
        f"- Median breach horizon: "
        f"{s['median_breach_horizon'] if s['median_breach_horizon'] else 'beyond horizon'}",
        f"- Intervals: {s['calibration_note']}",
        "",
        HONESTY_BOX,
        "",
        "## Breach probability by horizon",
        "",
        "| period | P(first breach in period) | P(breach by period) |",
        "|---|---|---|",
    ]
    for period, p_first, p_by in zip(
            table.period, table.p_first_breach_in_period,
            table.p_breach_by_horizon):
        lines.append(f"| {period} | {p_first:.3f} | {p_by:.3f} |")
    lines += ["", "## First-breach distribution", ""]
    for k, v in dist.items():
        lines.append(f"- {k}: {v:.3f}")
    if alarm is not None:
        lines += [
            "", "## Alarm policy (designed on this series' backtest)", "",
            f"- Alarm when P(breach within policy horizon) >= "
            f"{alarm.threshold_on_probability:.3f}",
            f"- Target step FAR: {alarm.target_far:.2f}; backtest FAR: "
            f"{alarm.backtest_far if alarm.backtest_far is not None else 'not estimable'}"
            f" (negative origins: {alarm.n_negative_origins}, positive: "
            f"{alarm.n_positive_origins})",
            f"- Designed ARL0: "
            f"{alarm.arl0 if alarm.arl0 is not None else 'not estimable'} periods",
            f"- Detection rate on positive origins: "
            f"{alarm.detection_rate}",
            "",
            f"> {alarm.caveat}",
        ]
    if benchmark_md:
        lines += ["", "## Self-benchmark (FAR-matched)", "", benchmark_md]
    if fan_png:
        lines += ["", f"![forecast fan]({fan_png})"]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path
