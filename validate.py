#!/usr/bin/env python3
"""FairCast validation harness.

Stages:
  1. import/compile — the package imports and byte-compiles clean.
  2. test suite — the full pytest suite (validation refusals, metric
     correctness, planted-truth behaviour, near-miss negatives,
     certificate algebra, benchmark honesty, CLI smoke).
  3. planted skill — on planted-breach panels with known breach dates,
     the state-space forecaster must beat *persistence* on median
     breach-date error (the bar set in the tool's design plan). The
     linear-extrapolation comparison is printed either way: on the
     clean planted trends used here, extrapolation is at least as good
     — the same result Paper 11 reported — and this stage does not
     hide it. Failure prints NO-SKILL and fails the stage.
  4. near-miss negatives — compact independent re-checks: invalid
     inputs refuse, never-breach panels stay inside the designed alarm
     budget, seasonal oscillation is not read as a trend, tiny-n
     intervals widen.
  5. reproduction — (a) the bundled ACS worked example reproduces its
     frozen expected numbers; (b) OPTIONAL external stage: if Paper
     11's surviving results tree is present on this machine, the
     score-TV drift-budget certificate is re-verified against the
     paper's population scores (540 checks, 0 violations expected).
     Absent tree -> SKIP, not fail.
  6. README consistency — the README's quoted example numbers match
     examples/acs_expected.json exactly as formatted there.

Exit code 0 iff every non-skipped stage passes.
"""
from __future__ import annotations

import compileall
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("PYTHONPATH", str(SRC))

RESULTS: list[tuple[str, bool | None, str]] = []


def stage(name):
    def deco(fn):
        try:
            note = fn()
            RESULTS.append((name, True, note or ""))
            print(f"[PASS] {name} {note or ''}")
        except Exception as exc:  # noqa: BLE001 — harness reports, not hides
            RESULTS.append((name, False, str(exc)))
            print(f"[FAIL] {name}: {exc}")
        return fn
    return deco


def skip(name, why):
    RESULTS.append((name, None, why))
    print(f"[SKIP] {name}: {why}")


# ---------------------------------------------------------------- stage 1
@stage("1 import/compile")
def _s1():
    ok = compileall.compile_dir(str(SRC), quiet=1)
    assert ok, "byte-compile failed"
    import faircast
    return f"(faircast {faircast.__version__})"


# ---------------------------------------------------------------- stage 2
@stage("2 test suite (pytest)")
def _s2():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)})
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    assert r.returncode == 0, f"pytest failed:\n{r.stdout[-3000:]}"
    return f"({tail})"


# ---------------------------------------------------------------- stage 3
@stage("3 planted skill vs persistence (breach-date error)")
def _s3():
    import numpy as np

    import faircast as fc
    from faircast.baselines import linear_paths, persistence_paths
    from faircast.conformal import truncate_panel
    from faircast.synthetic import planted_breach_panels

    tau, horizon = 0.10, 4
    frames, breaches = planted_breach_panels(n_series=12, seed=7)
    errs = {"faircast": [], "persistence": [], "linear": []}
    for fr, b in zip(frames, breaches):
        panel = fc.load_counts(fr)
        origin = b - 1
        true_step = b - origin + 1
        sub = truncate_panel(panel, origin)
        f = fc.Forecaster(metric="dp_gap", threshold=tau, horizon=horizon,
                          n_paths=2000).fit(sub)
        dist = f.breach_time_distribution()
        steps = np.arange(1, horizon + 1)
        probs = np.array([dist[p] for p in f.future_periods_])
        e = float(probs @ steps + dist["none"] * (horizon + 1))
        errs["faircast"].append(abs(e - true_step))
        gaps = sub.metric_series(fc.METRICS["dp_gap"])
        rng = np.random.default_rng(99)
        for name, fn in (("persistence", persistence_paths),
                         ("linear", linear_paths)):
            paths = np.clip(fn(gaps, horizon, 2000, rng), 0, 1)
            mask = paths >= tau
            first = np.where(mask.any(axis=1),
                             mask.argmax(axis=1) + 1, horizon + 1)
            errs[name].append(abs(float(first.mean()) - true_step))
    med = {k: float(np.median(v)) for k, v in errs.items()}
    print(f"    median |breach-date error| (periods): "
          f"faircast {med['faircast']:.3f}, persistence "
          f"{med['persistence']:.3f}, linear {med['linear']:.3f}")
    if not med["faircast"] < med["persistence"]:
        raise AssertionError(
            "NO-SKILL: faircast does not beat persistence on planted "
            "breach-date error; the skill claim fails with it.")
    note = (f"(faircast {med['faircast']:.3f} < persistence "
            f"{med['persistence']:.3f}; linear {med['linear']:.3f} — "
            f"extrapolation comparison printed, not hidden)")
    return note


# ---------------------------------------------------------------- stage 4
@stage("4 near-miss negatives")
def _s4():
    import numpy as np
    import pandas as pd

    import faircast as fc
    from faircast import FairCastError
    from faircast.conformal import truncate_panel
    from faircast.synthetic import (never_breach_panels, seasonal_panels,
                                    tiny_n_panel)

    # (a) invalid inputs refuse
    bad = pd.DataFrame({"period": [0, 1, 2, 3], "group": ["A"] * 4,
                        "n": [100] * 4, "positives": [200] * 4})
    try:
        fc.load_counts(bad)
        raise AssertionError("positives > n was not refused")
    except FairCastError:
        pass
    # (b) never-breach alarm budget
    frames, _ = never_breach_panels(n_series=1, seed=11)
    panel = fc.load_counts(frames[0])
    design = fc.design_alarm(
        lambda: fc.Forecaster(metric="dp_gap", threshold=0.10, horizon=2,
                              n_paths=2000),
        panel, policy_horizon=2, target_far=0.10)
    assert design.far_estimable and design.backtest_far <= 0.20, design
    # (c) seasonal median stays clear of the threshold
    frames, _ = seasonal_panels(n_series=1, seed=13)
    f = fc.Forecaster(metric="dp_gap", threshold=0.10, horizon=6,
                      n_paths=2000).fit(fc.load_counts(frames[0]))
    assert float(f.fan().q50.iloc[-1]) < 0.075
    # (d) tiny-n intervals widen
    kw = dict(metric="dp_gap", threshold=0.10, horizon=3, n_paths=2000)
    fs = fc.Forecaster(**kw).fit(truncate_panel(
        fc.load_counts(tiny_n_panel(n_per=25)), 5))
    fb = fc.Forecaster(**kw).fit(truncate_panel(
        fc.load_counts(tiny_n_panel(n_per=20000)), 5))
    w_s = float((fs.fan().q95 - fs.fan().q05).iloc[1])
    w_b = float((fb.fan().q95 - fb.fan().q05).iloc[1])
    assert w_s > 1.8 * w_b, (w_s, w_b)
    return "(refusals, FAR budget, seasonal, tiny-n all hold)"


# ---------------------------------------------------------------- stage 5
@stage("5a reproduction: bundled ACS example")
def _s5a():
    r = subprocess.run(
        [sys.executable, "examples/acs_example.py"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)})
    assert r.returncode == 0, f"example failed:\n{r.stderr[-2000:]}"
    got = None
    for line in r.stdout.splitlines():
        if line.startswith("SUMMARY_JSON "):
            got = json.loads(line[len("SUMMARY_JSON "):])
    assert got is not None, "example printed no SUMMARY_JSON"
    exp = json.loads((ROOT / "examples" / "acs_expected.json")
                     .read_text())

    def close(a, b, tol=0.02):
        if isinstance(a, float) or isinstance(b, float):
            return abs(float(a) - float(b)) <= tol
        if isinstance(a, list):
            return len(a) == len(b) and all(
                close(x, y, tol) for x, y in zip(a, b))
        if isinstance(a, dict):
            return all(close(a[k], b[k], tol) for k in b)
        return a == b

    assert close(got["demo_a"], exp["demo_a"]), "demo A numbers drifted"
    assert close(got["demo_b"], exp["demo_b"]), "demo B numbers drifted"
    gb, eb = got["benchmark"], exp["benchmark"]
    assert gb["n_series"] == eb["n_series"]
    assert gb["n_breaching"] == eb["n_breaching"]
    assert gb["verdicts"] == eb["verdicts"], "benchmark verdicts drifted"
    assert close(gb["rows"], eb["rows"]), "benchmark table drifted"
    assert close(gb["brier"], eb["brier"]), "brier scores drifted"
    return "(demo A/B, benchmark table, Brier, verdicts reproduce)"


PAPER11_RESULTS = (Path.home() / "workspace" / "goals" /
                   "10-journal-papers-submitted-to-good-journals" /
                   "hidden_files" / "paper11_work" / "results")


def _s5b():
    import numpy as np
    import pandas as pd

    import faircast as fc

    pop = pd.read_csv(PAPER11_RESULTS / "population_metrics.csv")
    total_checks = total_violations = 0
    for (task, state), ser in pop.groupby(["task", "state"]):
        ser = ser.sort_values("year")
        years = ser.year.tolist()
        raw = {}
        for y in years:
            d = np.load(PAPER11_RESULTS / "scores" /
                        f"{task}_{state}_{y}.npz")
            raw[y] = (d["scores"], d["y"], d["a"])
        for metric_name, col, kind in (
                ("dp_gap", "dp_gap", "dp"), ("eo_gap", "eo_gap", "eo")):
            by_period = []
            for y in years:
                s, yy, a = raw[y]
                if kind == "dp":
                    cells = {(f"g{av}",): s[a == av] for av in (0, 1)}
                else:
                    cells = {(f"g{av}y{yv}",): s[(a == av) & (yy == yv)]
                             for av in (0, 1) for yv in (0, 1)}
                by_period.append(cells)
            eps = fc.score_budget(by_period)
            v = fc.verify(ser[col].to_numpy(dtype=float), eps)
            total_checks += v["checks"]
            total_violations += v["violations"]
    assert total_checks == 540, f"expected 540 checks, got {total_checks}"
    assert total_violations == 0, \
        f"score-TV certificate violated {total_violations} times"
    return "(Paper 11 population scores: 540 checks, 0 violations)"


if (PAPER11_RESULTS / "scores").is_dir():
    stage("5b reproduction: Paper 11 score-TV certificate")(_s5b)
else:
    skip("5b reproduction: Paper 11 score-TV certificate",
         "paper results tree not present on this machine")


# ---------------------------------------------------------------- stage 6
@stage("6 README/results consistency")
def _s6():
    exp = json.loads((ROOT / "examples" / "acs_expected.json").read_text())
    readme = (ROOT / "README.md").read_text()
    required = [f"{p:.3f}" for p in exp["demo_a"]["p_breach_by_horizon"]]
    required += [f"{exp['benchmark']['brier'][m]:.3f}"
                 for m in ("faircast", "persistence", "linear")]
    required += ["NO-SKILL", "22,032"]
    missing = [s for s in required if s not in readme]
    assert not missing, f"README is missing measured strings: {missing}"
    return "(README quotes the frozen example numbers)"


def main() -> int:
    failed = [n for n, ok, _ in RESULTS if ok is False]
    skipped = [n for n, ok, _ in RESULTS if ok is None]
    print()
    print(f"OVERALL: {'FAIL' if failed else 'PASS'}"
          f" ({sum(1 for _, ok, _ in RESULTS if ok)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped)")
    if failed:
        print("failed stages:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
