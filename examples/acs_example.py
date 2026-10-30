"""Worked example on public data: the bundled ACS slice.

Data: population EO/DP gaps for a frozen logistic model on three
folktables ACS tasks in six states, 2014-2019 — a reduced slice of the
51-jurisdiction corpus of Paper 11 (Ghosh, submitted to AI & Society,
2026), derived from the American Community Survey (US Census Bureau)
via folktables. See faircast.data.bundled_acs_path for provenance.

Two single-series forecasts are shown, chosen because together they
are the honest picture:

* ACSIncome, CA — the EO gap rose steadily (0.044, 0.047, 0.049) and
  first crossed 0.05 in 2017. Fitted only through 2016, does the tool
  see it coming?
* ACSEmployment, CA — the EO gap sat at ~0.013 for five years and then
  jumped to 0.054 in 2019. Fitted through 2018, the tool cannot see a
  level shift that has no precursor in the series — and says so with a
  low breach probability that the realised value then contradicts.
  Forecasting has failure modes; this example keeps one in view.

Then the FAR-matched self-benchmark runs over all 18 trajectories and
prints its verdicts, whatever they are.

Run:  python examples/acs_example.py
Writes examples/acs_report.md and examples/acs_income_ca_fan.png, and
prints a SUMMARY_JSON line (the numbers validate.py checks against
examples/acs_expected.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faircast as fc  # noqa: E402
from faircast.conformal import truncate_panel  # noqa: E402
from faircast.data import SeriesPanel, bundled_acs_path  # noqa: E402
from faircast.report import fan_chart, write_report  # noqa: E402

TAU = 0.05


def series_panel(df: pd.DataFrame, task: str, state: str) -> SeriesPanel:
    sub = df[(df.task == task) & (df.state == state)].sort_values("year")
    return SeriesPanel(periods=sub.year.tolist(),
                       values=sub.eo_gap.to_numpy(dtype=float), ses=None)


def main() -> dict:
    df = pd.read_csv(bundled_acs_path())

    # --- demo A: a foreseen breach (fit 2014-2016, breach realised 2017)
    ca_income = series_panel(df, "ACSIncome", "CA")
    hist_a = truncate_panel(ca_income, 3)
    fa = fc.Forecaster(metric="eo_gap", threshold=TAU, horizon=3).fit(hist_a)
    tab_a = fa.breach_table()
    realised_a = float(ca_income.values[3])  # 2017

    # --- demo B: an unforeseen level shift (fit 2014-2018, jump in 2019)
    ca_emp = series_panel(df, "ACSEmployment", "CA")
    hist_b = truncate_panel(ca_emp, 5)
    fb = fc.Forecaster(metric="eo_gap", threshold=TAU, horizon=2).fit(hist_b)
    tab_b = fb.breach_table()
    realised_b = float(ca_emp.values[5])  # 2019

    # --- self-benchmark over all 18 trajectories
    panels = [series_panel(df, t, s)
              for (t, s) in sorted(df.groupby(["task", "state"]).groups)]
    bench = fc.benchmark_panels(panels, metric="eo_gap", threshold=TAU,
                                horizon=2)

    out_dir = Path(__file__).resolve().parent
    fan_chart(fa, out_dir / "acs_income_ca_fan.png")
    write_report(fa, out_dir / "acs_report.md",
                 benchmark_md=bench.to_markdown(),
                 fan_png="acs_income_ca_fan.png")

    summary = {
        "demo_a": {
            "series": "ACSIncome CA eo_gap, fit 2014-2016",
            "p_breach_by_horizon": [round(float(v), 4) for v in
                                    tab_a.p_breach_by_horizon],
            "realised_2017": realised_a,
            "realised_breach": bool(realised_a >= TAU),
        },
        "demo_b": {
            "series": "ACSEmployment CA eo_gap, fit 2014-2018",
            "p_breach_by_horizon": [round(float(v), 4) for v in
                                    tab_b.p_breach_by_horizon],
            "realised_2019": realised_b,
            "realised_breach": bool(realised_b >= TAU),
        },
        "benchmark": {
            "n_series": bench.n_series,
            "n_breaching": bench.n_breaching,
            "n_already_breached": bench.n_already_breached,
            "rows": [{k: (round(float(r[k]), 4) if k != "method" else r[k])
                      for k in ("method", "far_target", "achieved_far",
                                "coverage", "median_lead")}
                     for _, r in bench.table.iterrows()],
            "brier": {k: round(float(v), 4)
                      for k, v in bench.brier.items()},
            "verdicts": bench.verdicts,
        },
    }
    print(json.dumps(summary, indent=2))
    print("SUMMARY_JSON " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
