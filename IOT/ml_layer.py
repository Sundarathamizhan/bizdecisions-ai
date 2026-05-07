"""
ml_layer.py — Machine Learning Pipeline
════════════════════════════════════════
Three ML algorithms run continuously on the sensor stream:

  1. Z-Score Anomaly Detection
     ──────────────────────────
     Uses a rolling window to compute μ (mean) and σ (std deviation).
     A reading is flagged anomalous when |z| = |(x − μ) / σ| > threshold.
     Window: 20 samples. Threshold: 2.5σ (≈99th percentile of normal dist).

  2. Isolation Forest (pure-Python approximation)
     ───────────────────────────────────────────
     Random axis-parallel splits isolate anomalies faster (shorter avg path).
     We approximate with 50 random trees, path-length scoring, and compare
     against expected avg path length E(n) = 2×H(n-1) − (2(n-1)/n)
     where H is the harmonic number. Anomaly score ∈ [0,1]; > 0.6 = alert.

  3. Linear Regression Forecasting
     ────────────────────────────────
     Ordinary least-squares regression over the rolling window.
     Gives slope (trend direction) and predicts value N steps ahead.
     Slope interpretation: >0.05 = rising, <-0.05 = falling, else stable.

  4. Composite Health Score
     ────────────────────────
     Starts at 100, penalizes based on anomaly density across all sensors.
     anomaly_rate = anomalies_in_last_30 / 30 per sensor.
     Score = 100 − Σ(anomaly_rate_i × 25) clamped to [0, 100].

References:
  • Liu et al. (2008) "Isolation Forest" ICDM
  • Montgomery & Runger "Applied Statistics and Probability for Engineers"
"""

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from iot_layer import SensorReading


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLResult:
    """
    ML output for one sensor at one tick.
    """
    sensor_id:      str
    tick:           int
    value:          float
    unit:           str
    # Z-Score
    z_score:        Optional[float]   = None
    z_anomaly:      bool              = False
    mean:           Optional[float]   = None
    std:            Optional[float]   = None
    # Isolation Forest
    if_score:       Optional[float]   = None
    if_anomaly:     bool              = False
    # Linear Regression
    slope:          Optional[float]   = None
    trend:          str               = "unknown"
    forecast_5:     Optional[float]   = None   # predicted value in 5 steps
    # Combined
    anomaly:        bool              = False   # Z OR IF triggered
    severity:       str               = "ok"   # ok / warning / critical


@dataclass
class HistoryRecord:
    """Used for backtest replay and export."""
    tick:    int
    results: Dict[str, MLResult]


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Statistics Helper
# ─────────────────────────────────────────────────────────────────────────────

class RollingWindow:
    """
    Maintains a fixed-size circular buffer and computes μ / σ in O(1)
    using Welford's online algorithm for numerical stability.
    """

    def __init__(self, maxlen: int = 20):
        self._buf:    deque  = deque(maxlen=maxlen)
        self._sum:    float  = 0.0
        self._sum_sq: float  = 0.0

    def push(self, x: float):
        if len(self._buf) == self._buf.maxlen:
            old = self._buf[0]
            self._sum    -= old
            self._sum_sq -= old * old
        self._buf.append(x)
        self._sum    += x
        self._sum_sq += x * x

    def mean(self) -> Optional[float]:
        n = len(self._buf)
        return self._sum / n if n > 0 else None

    def std(self) -> Optional[float]:
        n = len(self._buf)
        if n < 2:
            return None
        variance = (self._sum_sq - self._sum ** 2 / n) / n
        return math.sqrt(max(0.0, variance))

    def values(self) -> List[float]:
        return list(self._buf)

    def __len__(self):
        return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# Isolation Forest (Pure Python Approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _harmonic(n: int) -> float:
    """Approximate harmonic number H(n) = Σ 1/k for k=1..n."""
    return math.log(n) + 0.5772156649   # Euler-Mascheroni constant

def _avg_path_length(n: int) -> float:
    """Expected average path length in a random BST of size n."""
    if n <= 1:
        return 1.0
    return 2.0 * _harmonic(n - 1) - 2.0 * (n - 1) / n

def _isolation_path(value: float, samples: List[float], depth: int = 0, max_depth: int = 12) -> float:
    """
    Simulate one isolation tree split recursively.
    Returns the path length to isolate `value` from `samples`.
    """
    n = len(samples)
    if n <= 1 or depth >= max_depth:
        return depth + _avg_path_length(n)

    lo, hi = min(samples), max(samples)
    if hi == lo:
        return depth + _avg_path_length(n)

    split = random.uniform(lo, hi)
    left  = [x for x in samples if x <= split]
    right = [x for x in samples if x >  split]

    if value <= split:
        return _isolation_path(value, left, depth + 1, max_depth)
    else:
        return _isolation_path(value, right, depth + 1, max_depth)


class IsolationForest:
    """
    Lightweight single-variable Isolation Forest.
    Builds `n_trees` random isolation paths and averages them.
    Anomaly score = 2^(−avg_path / E(n))
    Score > 0.60 → anomaly (approximately top 5% most isolated).
    """

    def __init__(self, n_trees: int = 50, threshold: float = 0.60):
        self.n_trees   = n_trees
        self.threshold = threshold

    def score(self, value: float, samples: List[float]) -> float:
        n = len(samples)
        if n < 8:
            return 0.0
        avg_path = sum(
            _isolation_path(value, samples) for _ in range(self.n_trees)
        ) / self.n_trees
        c = _avg_path_length(n)
        return 2 ** (-avg_path / c)

    def is_anomaly(self, value: float, samples: List[float]) -> Tuple[float, bool]:
        s = self.score(value, samples)
        return s, s > self.threshold


# ─────────────────────────────────────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    slope:      float
    intercept:  float
    r_squared:  float
    trend:      str        # "rising" | "falling" | "stable"

    def predict(self, steps_ahead: int, current_n: int) -> float:
        return self.intercept + self.slope * (current_n - 1 + steps_ahead)


def linear_regression(values: List[float]) -> Optional[RegressionResult]:
    """
    Ordinary Least Squares regression y = a + bx.
    Returns slope, intercept, R², and trend label.
    Requires ≥ 4 data points.
    """
    n = len(values)
    if n < 4:
        return None

    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_yy = sum((y - y_mean) ** 2 for y in values)
    ss_xy = sum((xs[i] - x_mean) * (values[i] - y_mean) for i in range(n))

    if ss_xx < 1e-12:
        return None

    slope     = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_sq      = (ss_xy ** 2 / (ss_xx * ss_yy)) if ss_yy > 1e-12 else 1.0

    if   slope >  0.05:  trend = "rising"
    elif slope < -0.05:  trend = "falling"
    else:                trend = "stable"

    return RegressionResult(slope, intercept, r_sq, trend)


# ─────────────────────────────────────────────────────────────────────────────
# ML Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MLPipeline:
    """
    Processes sensor readings tick-by-tick through all three ML layers.

    Usage
    -----
        pipeline = MLPipeline(window=20, zscore_threshold=2.5)
        results  = pipeline.process(readings, tick)   # Dict[str, MLResult]
        score    = pipeline.compute_health_score()    # int [0..100]
        summary  = pipeline.get_summary()             # Dict for AI layer
    """

    def __init__(self, window: int = 20, zscore_threshold: float = 2.5):
        self.window_size      = window
        self.zscore_threshold = zscore_threshold
        self.windows:   Dict[str, RollingWindow]    = defaultdict(lambda: RollingWindow(window))
        self.iso_forest = IsolationForest(n_trees=50, threshold=0.60)
        self.results_log: List[Dict[str, MLResult]] = []   # full history
        self.anomaly_log: Dict[str, deque]          = defaultdict(lambda: deque(maxlen=30))
        self.history:     List[HistoryRecord]       = []
        self.SENSOR_DECIMALS = {"temp":1,"humidity":1,"soil":1,"co2":0,"light":0,"ph":2}

    # ── Core Process ─────────────────────────────────────────────────────

    def process(self, readings: Dict[str, SensorReading], tick: int) -> Dict[str, MLResult]:
        """
        Run full ML pipeline on one tick's readings.
        Returns MLResult per sensor.
        """
        tick_results: Dict[str, MLResult] = {}

        for sid, reading in readings.items():
            v   = reading.value
            dec = self.SENSOR_DECIMALS.get(sid, 1)
            win = self.windows[sid]

            # ── Z-Score Detection ────────────────────────────────────────
            win.push(v)
            mu  = win.mean()
            sig = win.std()
            z_anomaly = False
            z_score   = None

            if mu is not None and sig is not None and sig > 1e-6 and len(win) >= 10:
                z_score   = (v - mu) / sig
                z_anomaly = abs(z_score) > self.zscore_threshold

            # ── Isolation Forest ─────────────────────────────────────────
            samples   = win.values()
            if_score, if_anomaly = self.iso_forest.is_anomaly(v, samples)

            # ── Linear Regression ────────────────────────────────────────
            reg         = linear_regression(samples)
            slope       = reg.slope          if reg else None
            trend       = reg.trend          if reg else "unknown"
            forecast_5  = round(reg.predict(5, len(samples)), dec) if reg else None

            # [FIX] Clamp forecast to physical bounds
            if forecast_5 is not None:
                bounds = {
                    "temp": (-20, 60), "humidity": (0, 100), "soil": (0, 100),
                    "light": (0, 120000), "co2": (0, 5000), "ph": (0, 14)
                }
                b_min, b_max = bounds.get(sid, (0, 1e6))
                forecast_5 = max(b_min, min(b_max, forecast_5))


            # ── Combined Anomaly Flag ────────────────────────────────────
            anomaly  = z_anomaly or if_anomaly
            severity = "ok"
            if anomaly:
                severity = "critical" if (z_score and abs(z_score) > 3.5) else "warning"

            result = MLResult(
                sensor_id   = sid,
                tick        = tick,
                value       = v,
                unit        = reading.unit,
                z_score     = round(z_score, 2) if z_score is not None else None,
                z_anomaly   = z_anomaly,
                mean        = round(mu, dec)  if mu  is not None else None,
                std         = round(sig, dec) if sig is not None else None,
                if_score    = round(if_score, 3),
                if_anomaly  = if_anomaly,
                slope       = round(slope, 4) if slope is not None else None,
                trend       = trend,
                forecast_5  = forecast_5,
                anomaly     = anomaly,
                severity    = severity,
            )
            tick_results[sid] = result
            self.anomaly_log[sid].append(1 if anomaly else 0)

        self.results_log.append(tick_results)
        self.history.append(HistoryRecord(tick, tick_results))
        return tick_results

    # ── Health Score ──────────────────────────────────────────────────────

    def compute_health_score(self) -> int:
        """
        Composite health score [0..100].
        Penalizes each sensor proportionally to its recent anomaly rate.
        penalty_i = anomaly_rate_i × 25
        score     = 100 − Σ penalty_i   clamped to [0, 100]
        """
        penalty = 0.0
        for sid, log in self.anomaly_log.items():
            if len(log) == 0:
                continue
            rate     = sum(log) / len(log)
            penalty += rate * 25.0
        return max(0, min(100, round(100 - penalty)))

    # ── Summary for AI Layer ──────────────────────────────────────────────

    def get_summary(self) -> Dict:
        """
        Builds a structured summary of the latest ML state for the AI advisor.
        """
        if not self.results_log:
            return {}
        latest    = self.results_log[-1]
        summary   = {}
        for sid, r in latest.items():
            anomaly_count = sum(self.anomaly_log[sid])
            summary[sid] = {
                "name":          sid,
                "current_value": r.value,
                "unit":          r.unit,
                "mean":          r.mean,
                "std":           r.std,
                "z_score":       r.z_score,
                "anomaly":       r.anomaly,
                "severity":      r.severity,
                "trend":         r.trend,
                "slope":         r.slope,
                "forecast_5":    r.forecast_5,
                "if_score":      r.if_score,
                "recent_anomalies": anomaly_count,
            }
        return summary

    # ── Evaluation Metrics ────────────────────────────────────────────────

    def evaluate(self, ground_truth: Dict[str, List[bool]]) -> Dict:
        """
        Compare ML anomaly flags against ground truth labels.
        Returns precision, recall, F1 per sensor.
        Used by the backtesting module.
        """
        metrics = {}
        for sid in ground_truth:
            predicted = [
                row[sid].anomaly
                for row in self.results_log
                if sid in row
            ]
            actual = ground_truth[sid]
            n = min(len(predicted), len(actual))
            if n == 0:
                continue

            tp = sum(p and a for p, a in zip(predicted[:n], actual[:n]))
            fp = sum(p and not a for p, a in zip(predicted[:n], actual[:n]))
            fn = sum(not p and a for p, a in zip(predicted[:n], actual[:n]))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            metrics[sid] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 3),
                "recall":    round(recall, 3),
                "f1":        round(f1, 3),
            }
        return metrics
