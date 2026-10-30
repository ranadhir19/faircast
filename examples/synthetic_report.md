# FairCast forecast report

- Metric: **dp_gap**, threshold 0.1 (breach direction: above)
- Last observed period: 3 — current value 0.0555
- P(breach within 4 periods): **0.705**
- Median breach horizon: 3
- Intervals: not calibrated

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


## Breach probability by horizon

| period | P(breach in period) | P(breach by period) |
|---|---|---|
| 4.0 | 0.034 | 0.034 |
| 5.0 | 0.355 | 0.355 |
| 6.0 | 0.593 | 0.594 |
| 7.0 | 0.703 | 0.705 |

## First-breach distribution

- 4: 0.034
- 5: 0.321
- 6: 0.239
- 7: 0.111
- none: 0.295

## Alarm policy (designed on this series' backtest)

- Alarm when P(breach within policy horizon) >= 0.310
- Target step FAR: 0.10; backtest FAR: 0.16666666666666666 (negative origins: 6, positive: 0)
- Designed ARL0: 6.0 periods
- Detection rate on positive origins: None

> FAR transport caveat (Paper 11, Ghosh 2026): alarm thresholds fitted to a 0.10 step-FAR target on training jurisdictions fired at a fold-mean 0.293 on held-out jurisdictions (range 0.005-1.000). The backtest FAR below is estimated on THIS series' own history; on a new series, recalibrate before trusting the budget.
