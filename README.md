# FairCast — fairness-breach forecasting under natural drift

FairCast answers a question that static fairness audits cannot: *given
how a fairness metric has moved so far, what is the probability it
breaches its threshold in the next few periods — and how much is that
number worth?*

It productises the machinery of Paper 11 — Ghosh, *The Limits of
Fairness-Breach Forecasting Under Natural Drift: An Exact Drift-Budget
Certificate, a Calibration-Transport Bound, and a Cross-State Transport
Failure* (submitted to *AI & Society*, 2026) — including that paper's
central honest result, which is built into the tool rather than buried:
**the forecasting claim was pre-registered and falsified.** FairCast's
value proposition is calibrated breach-risk estimation, an exactly
verified drift-budget certificate, and accounting that prints
**NO-SKILL** whenever the forecaster has not beaten the trivial
baselines. Read [What this tool does not
establish](#what-this-tool-does-not-establish) before quoting any
number it produces.

## Install

```bash
pip install faircast
```

Python ≥ 3.10; dependencies are numpy, scipy, pandas, matplotlib.

## Inputs

Three tidy shapes, all strictly validated — invalid inputs raise
`FairCastError` with a specific message. Nothing is silently dropped,
filled, aggregated, or repaired: in fairness monitoring, a gap in the
data is a fact the caller must explain, not a nuisance to interpolate
over.

| shape | columns | use |
|---|---|---|
| counts | `period, group, n, positives` | selection-rate metrics (`dp_gap`, `di_ratio`, `selection_rate`) |
| decisions | `period, group, y, y_pred` | adds TPR/FPR cells, enabling `eo_gap` |
| series | `period, value[, se]` | an already-computed metric trajectory |

A fairness history needs at least 4 periods to forecast; fewer can
describe a metric, they cannot forecast one.

## Quickstart

```bash
# breach-probability table + report + fan chart
faircast forecast counts.csv --format counts --metric dp_gap \
    --threshold 0.05 --horizon 6 --target-far 0.10 \
    --out report.md --fan fan.png

# FAR-matched self-benchmark against persistence and linear extrapolation
faircast benchmark series.csv --series-cols task,state \
    --value-col eo_gap --metric eo_gap --threshold 0.05

# drift-budget certificate for a counts panel
faircast certify counts.csv --metric dp_gap --threshold 0.05
```

```python
import faircast as fc

panel = fc.load_counts("counts.csv")                    # or load_decisions / load_series
f = fc.Forecaster(metric="dp_gap", threshold=0.05, horizon=6).fit(panel)
f.breach_table()          # P(first breach in period), P(breach by period)
f.fan()                   # metric quantiles per future period
fc.calibrate(f, panel)    # conformal-widened 90% band from backtests

alarm = fc.design_alarm(lambda: fc.Forecaster(metric="dp_gap", threshold=0.05),
                        panel, target_far=0.10)   # designed FAR + achieved backtest FAR

bench = fc.benchmark_panels(panels, metric="eo_gap", threshold=0.05)
print(bench.to_markdown())                        # prints SKILL or NO-SKILL, per FAR target
```

## How it works

- **Forecaster.** Each group cell's rate is modelled on the logit
  scale by a local-linear-trend state-space (Kalman) model —
  observation variance from the cell counts, hyperparameters by grid
  search on the Kalman log-likelihood, slope damping. Simulated
  latent paths are pushed through the metric exactly, so breach
  probabilities are statements about the *gap*, not about a
  linearisation of it. A binomial sampling-noise layer is available
  (`sampling_noise=True`); the default forecasts the latent gap.
- **Conformal calibration.** Rolling-origin backtests measure how
  much the nominal 90% fan must widen to cover realised values, per
  horizon (split-conformal inflation, capped). Too few origins → the
  forecaster stays uncalibrated and says so.
- **Alarm policy.** The alarm threshold on breach probability is
  chosen to meet a target step-level false-alarm rate (FAR) on the
  series' own backtest, and the *achieved* backtest FAR and ARL0 =
  1/FAR are reported next to the target. If every backtest origin is
  followed by a breach, no FAR can be designed and the call refuses.
- **Certificate.** For gap functionals, the metric can move in one
  step by at most the sum over cells of the cells' distribution
  movement (total variation). FairCast computes this drift budget
  from counts (exact, by triangle inequality) or from raw score
  samples (histogram TV, Paper 11's construction), verifies it
  exhaustively over a series' history, and reports the *certified
  safe horizon* per origin. The disparate-impact ratio is refused:
  a ratio is not Lipschitz in this budget.
- **Self-benchmark.** Rolling origins, leave-one-series-out
  FAR-matched alarm thresholds, coverage / median lead / Brier for
  FairCast vs persistence vs linear extrapolation. Skill is claimed
  only under Paper 11's falsifier rule: median lead must *strictly*
  beat both trivial forecasters at matched FAR.

## Worked example: ACS population gaps (public data)

`examples/acs_example.py` runs on a bundled public slice —
`faircast/data/acs_population_gaps.csv`: population EO and DP gaps of
a frozen 2014-trained logistic model on three folktables ACS tasks
(ACSIncome, ACSEmployment, ACSPublicCoverage) in six states,
2014–2019, derived from the American Community Survey (US Census
Bureau). It is a reduced slice (18 of the paper's trajectories) of
Paper 11's 51-jurisdiction corpus. All numbers below were measured
by running the shipped code on that slice (frozen in
`examples/acs_expected.json`, re-verified by `validate.py`).

**A foreseen breach.** ACSIncome, California: the EO gap rose
0.044 → 0.047 → 0.049 over 2014–2016 and first crossed 0.05 in 2017
(realised 0.0556). Fitted only through 2016, FairCast put the
cumulative breach probability at **0.565** by 2017, 0.703 by 2018,
0.762 by 2019.

**An unforeseen breach.** ACSEmployment, California: the EO gap sat
at ≈0.013 for five years and jumped to 0.0542 in 2019. Fitted through
2018, FairCast put the breach probability at 0.000 — the level shift
had no precursor in the series, and the tool did not invent one.
Both outcomes are kept in the example on purpose.

**Self-benchmark over the 18 trajectories** (EO gap, τ = 0.05,
horizon 2; 5 series breach inside the monitored window — all of them
2019 employment level shifts — and 6 were already in breach before
the first origin and are excluded from coverage):

| method | FAR target | achieved FAR | coverage | median lead | Brier |
|---|---|---|---|---|---|
| faircast | 0.05 | 0.143 | 0.000 | 0.0 | 0.319 |
| persistence | 0.05 | 0.071 | 0.000 | 0.0 | 0.304 |
| linear | 0.05 | 0.071 | 0.000 | 0.0 | 0.327 |
| faircast | 0.10 | 0.143 | 0.000 | 0.0 | 0.319 |
| persistence | 0.10 | 0.071 | 0.000 | 0.0 | 0.304 |
| linear | 0.10 | 0.071 | 0.000 | 0.0 | 0.327 |
| faircast | 0.20 | 0.143 | 0.000 | 0.0 | 0.319 |
| persistence | 0.20 | 0.214 | 0.000 | 0.0 | 0.304 |
| linear | 0.20 | 0.214 | 0.000 | 0.0 | 0.327 |

Verdict at every FAR target: **NO-SKILL** — no method covered any of
the five in-window breaches, because all five were level shifts.
That is the Paper 11 finding in miniature, reproduced by the tool on
public data: on this corpus there was no lead time to be had, and
FairCast says so instead of rounding a near-miss up to a win.

**Planted-truth check** (validate.py stage 3; synthetic panels whose
latent breach dates are known, population-scale cells): median
breach-date error is 0.494 periods for FairCast vs 2.363 for
persistence — the state-space model does beat the naive forecaster
where a trend exists to be read. Linear extrapolation scores 0.344:
on clean linear trends it dates breaches slightly *better* than
FairCast. Both comparisons are printed by the harness; the README
quotes both.

**Certificate reproduction.** Paper 11 verified its population
drift-budget certificate exhaustively: **0 violations in 22,032
checks** (51 jurisdictions × 9 years × 3 tasks × 2 model classes).
FairCast's shipped `score_budget`/`verify` code re-verifies the
paper's surviving baseline corpus (18 trajectories × 2 metrics):
540 checks, 0 violations (validate.py stage 5b, run where the
paper's results tree is present).

## What this tool does not establish

- **No lead-time win.** Paper 11's pre-registered forecaster did not
  beat linear extrapolation on median lead at any FAR target
  (0.05/0.10/0.20), in either model class, and its transport-robust
  successor (TRA-Q) repaired calibration and coverage but still not
  lead time. FairCast inherits that result, not a marketing version
  of it. Where FairCast earns its keep is calibrated probabilities,
  honest intervals, designed alarm budgets, and the certificate.
- **Alarm budgets do not transport.** Thresholds fitted to a 0.10
  FAR target on Paper 11's training jurisdictions fired at a
  fold-mean 0.293 on held-out jurisdictions (per-fold range
  0.005–1.000). A FAR designed on one series is a starting point for
  another, never a guarantee — recalibrate per series and read the
  achieved backtest FAR, not the target.
- **The batch certificate is usually silent.** The population bound
  is exact; estimated from batch samples of n = 2,000 it held
  everywhere in Paper 11's corpus and certified a safe horizon of
  **0 in every cell**. A certificate that cannot certify is reported
  as silent (`certified_safe_horizon = 0`), never rounded up.
- **Level shifts are unforecastable from the series alone.** A gap
  that jumps after years of flatness (demo B above; all five
  in-window ACS breaches) carries no precursor for any forecaster
  reading only that series. FairCast will report a low probability
  and be wrong; the benchmark exists to keep that priced in.
- **Cells are forecast independently.** Cross-cell correlation
  (e.g. a shock moving all groups at once) is not modelled, so joint
  extremes are understated.
- **Cumulative probabilities inflate with horizon.** As the fan
  widens, P(breach by horizon h) grows partly from uncertainty, not
  only from drift. Read it with the first-breach distribution and
  the median path.
- **The bundled slice is a slice.** 18 trajectories from the
  paper's baseline design (6 states, one model class); the paper's
  full corpus is 51 jurisdictions × 2 model classes, and its
  headline numbers — not this slice's — are the ones to cite about
  the method.

## Validation

`python validate.py` runs: import/compile; the full pytest suite
(33 tests: validation refusals, hand-computed metrics, planted-truth
behaviour, near-miss negatives, certificate algebra including a
shrunken-budget control proving `verify` is not vacuous, NO-SKILL on
unforecastable jumps, CLI smoke); the planted-skill stage (fails,
printing NO-SKILL, if FairCast stops beating persistence on
breach-date error); near-miss re-checks; the ACS reproduction stage;
the optional Paper 11 certificate re-verification; and a
README/results consistency check.

## Provenance and citation

FairCast productises: Ranadhir Ghosh, *The Limits of Fairness-Breach
Forecasting Under Natural Drift: An Exact Drift-Budget Certificate,
a Calibration-Transport Bound, and a Cross-State Transport Failure*,
submitted to *AI & Society* (Springer) on 2026-10-29, submission ID
246414a7-2092-4505-b9ec-6d8ae3b38ac7. Data: folktables benchmark,
derived from the American Community Survey, US Census Bureau.

The project's own success bar (90 days from first release): ≥3,000
downloads/month and ≥200 GitHub stars; below half at day 60, the
engine folds back into the paper's artifact repository.

## License

MIT © 2026 Ranadhir Ghosh

