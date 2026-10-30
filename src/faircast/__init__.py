"""FairCast — fairness-breach forecasting under natural drift.

Given grouped outcome data over time (or an existing fairness-metric
series), FairCast forecasts breach risk: calibrated breach probabilities
by horizon, a fan chart, an alarm policy with a designed false-alarm
rate, a drift-budget certificate, and a self-benchmark against the
trivial forecasters that prints NO-SKILL when FairCast has not beaten
them. It productises Paper 11 (Ghosh, "The Limits of Fairness-Breach
Forecasting Under Natural Drift", submitted to AI & Society, 2026) —
including that paper's central honest result: its pre-registered
forecaster did *not* beat linear extrapolation on lead time, so this
tool sells calibration, verification and honest accounting, not
promised lead.
"""

from .alarm import AlarmDesign, design_alarm
from .benchmark import BenchmarkResult, benchmark_panels
from .certificate import certify, rate_budget, score_budget, verify
from .conformal import calibrate
from .data import Panel, SeriesPanel, load_counts, load_decisions, load_series
from .errors import FairCastError
from .forecast import Forecaster
from .metrics import METRICS

__version__ = "0.1.0"

__all__ = [
    "Forecaster",
    "Panel",
    "SeriesPanel",
    "load_counts",
    "load_decisions",
    "load_series",
    "FairCastError",
    "METRICS",
    "benchmark_panels",
    "BenchmarkResult",
    "design_alarm",
    "AlarmDesign",
    "calibrate",
    "rate_budget",
    "score_budget",
    "certify",
    "verify",
    "__version__",
]
