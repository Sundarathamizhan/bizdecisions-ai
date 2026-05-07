"""
ml_advanced.py — Advanced ML Extensions for NeuralFarm
═══════════════════════════════════════════════════════
Adds on top of the baseline MLPipeline (kept unchanged for comparison):

  1. DynamicThresholdDetector  — adaptive Z-score threshold
  2. HoltWintersForecaster     — level+trend exponential smoothing
  3. MinimalRNN                — pure-Python Elman RNN, online BPTT
  4. PearsonCorrelationTracker — ML-based sensor coupling alerts
  5. ExplainabilityEngine      — human-readable "reason" per decision
  6. AdvancedMLPipeline        — orchestrates all of the above

Run comparison:
  python main.py --mode compare --ticks 200
"""

import math
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from iot_layer import SensorReading
from ml_layer  import MLPipeline, MLResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _variance(vals) -> float:
    lst = list(vals)
    n = len(lst)
    if n < 2:
        return 0.0
    mean = sum(lst) / n
    return sum((x - mean) ** 2 for x in lst) / n


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dynamic Threshold Detector
# ─────────────────────────────────────────────────────────────────────────────

class DynamicThresholdDetector:
    """
    Adaptive Z-score threshold based on short-window vs long-window variance.
      • Noisy right now  → higher threshold (fewer false positives)
      • Stable right now → lower  threshold (catch subtle drift earlier)
    Threshold is clamped to [1.5σ, 4.0σ].
    """

    def __init__(self, base: float = 2.5, short: int = 20, long: int = 100):
        self.base   = base
        self._short = deque(maxlen=short)
        self._long  = deque(maxlen=long)

    def push(self, x: float):
        self._short.append(x)
        self._long.append(x)

    def threshold(self) -> float:
        if len(self._short) < 5 or len(self._long) < 20:
            return self.base
        sv = _variance(self._short)
        lv = _variance(self._long)
        if lv < 1e-9:
            return self.base
        ratio = sv / lv                          # >1 noisier, <1 calmer
        t = self.base + (ratio - 1.0) * 0.5
        return round(max(1.5, min(4.0, t)), 3)

    def is_anomaly(self, z: Optional[float]) -> bool:
        return z is not None and abs(z) > self.threshold()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Holt-Winters Forecaster (Double Exponential Smoothing)
# ─────────────────────────────────────────────────────────────────────────────

class HoltWintersForecaster:
    """
    Holt's method: level l_t and trend b_t updated each tick.
      l_t = α·y + (1-α)·(l_{t-1}+b_{t-1})
      b_t = β·(l_t-l_{t-1}) + (1-β)·b_{t-1}
      ŷ_{t+k} = l_t + k·b_t
    Better than OLS on trending/non-stationary greenhouse data.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta  = beta
        self._l: Optional[float] = None
        self._b: float = 0.0

    def update(self, y: float):
        if self._l is None:
            self._l, self._b = y, 0.0
        else:
            l_prev  = self._l
            self._l = self.alpha * y + (1 - self.alpha) * (self._l + self._b)
            self._b = self.beta * (self._l - l_prev) + (1 - self.beta) * self._b

    def predict(self, steps: int = 5) -> Optional[float]:
        return None if self._l is None else self._l + steps * self._b

    @property
    def trend_dir(self) -> str:
        if self._b >  0.05: return "rising"
        if self._b < -0.05: return "falling"
        return "stable"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Minimal RNN (Elman Network, Pure Python, Online BPTT)
# ─────────────────────────────────────────────────────────────────────────────

class MinimalRNN:
    """
    Elman RNN — scalar input/output, one hidden unit.
      h_t = tanh(w_rh·h_{t-1} + w_rx·x_t + b_h)
      y_t = w_o·h_t + b_o
    Trained online with 1-step SGD backprop.
    Input/output normalised to [0,1] by AdvancedMLPipeline.
    """

    def __init__(self, lr: float = 0.01):
        rng = random.Random(0)
        self.w_rh = rng.gauss(0, 0.1)
        self.w_rx = rng.gauss(0, 0.1)
        self.b_h  = 0.0
        self.w_o  = rng.gauss(0, 0.1)
        self.b_o  = 0.0
        self.h    = 0.0
        self.lr   = lr
        self._px  = 0.0   # previous input
        self._ph  = 0.0   # previous hidden

    @staticmethod
    def _tanh(x: float) -> float:
        if x > 20:  return  1.0
        if x < -20: return -1.0
        return math.tanh(x)

    def forward(self, x: float) -> float:
        self._px, self._ph = x, self.h
        h_pre  = self.w_rh * self.h + self.w_rx * x + self.b_h
        self.h = self._tanh(h_pre)
        return self.w_o * self.h + self.b_o

    def _backward(self, pred: float, target: float):
        err    = pred - target
        dh     = err * self.w_o
        dh_pre = dh * (1 - self.h * self.h)
        self.w_o  -= self.lr * err * self.h
        self.b_o  -= self.lr * err
        self.w_rh -= self.lr * dh_pre * self._ph
        self.w_rx -= self.lr * dh_pre * self._px
        self.b_h  -= self.lr * dh_pre

    def train_step(self, x_t: float, y_target: float) -> float:
        pred = self.forward(x_t)
        self._backward(pred, y_target)
        return pred

    def predict_ahead(self, x_start: float, steps: int = 5) -> float:
        """Autoregressive rollout — does NOT modify hidden state."""
        h, x = self.h, x_start
        for _ in range(steps):
            h_pre = self.w_rh * h + self.w_rx * x + self.b_h
            h     = self._tanh(h_pre)
            x     = self.w_o * h + self.b_o
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pearson Correlation Tracker
# ─────────────────────────────────────────────────────────────────────────────

class PearsonCorrelationTracker:
    """
    Rolling Pearson r for two sensor channels. Fires alert when measured
    correlation diverges from the expected physical relationship.
    e.g. (temp,humidity) expected r ≈ -0.8; large deviation → fault signal.
    """

    def __init__(self, window: int = 50,
                 expected_corr: Optional[float] = None,
                 alert_drop: float = 0.4):
        self._xs = deque(maxlen=window)
        self._ys = deque(maxlen=window)
        self.expected_corr = expected_corr
        self.alert_drop    = alert_drop

    def push(self, x: float, y: float):
        self._xs.append(x)
        self._ys.append(y)

    def correlation(self) -> Optional[float]:
        n = len(self._xs)
        if n < 10:
            return None
        xs, ys = list(self._xs), list(self._ys)
        mx, my = sum(xs)/n, sum(ys)/n
        num  = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
        den  = (math.sqrt(sum((x-mx)**2 for x in xs)) *
                math.sqrt(sum((y-my)**2 for y in ys)))
        return round(num/den, 3) if den > 1e-9 else None

    def is_decorrelated(self) -> bool:
        if self.expected_corr is None:
            return False
        r = self.correlation()
        return r is not None and abs(r - self.expected_corr) > self.alert_drop


# ─────────────────────────────────────────────────────────────────────────────
# 5. Explainability Engine
# ─────────────────────────────────────────────────────────────────────────────

class ExplainabilityEngine:
    """Generates a cited, human-readable reason string for each ML decision."""

    @staticmethod
    def explain(base_r: MLResult,
                dynamic_threshold: Optional[float] = None) -> str:
        reasons = []

        if base_r.z_score is not None and base_r.z_anomaly:
            th  = dynamic_threshold or 2.5
            lbl = "dynamic" if dynamic_threshold else "fixed"
            sev = "CRITICAL" if abs(base_r.z_score) > 3.5 else "WARNING"
            reasons.append(
                f"Z={base_r.z_score:+.2f}σ > {lbl} threshold {th:.2f}σ [{sev}]"
            )

        if base_r.if_score is not None and base_r.if_anomaly:
            reasons.append(
                f"IF-score={base_r.if_score:.3f} > 0.60 (short isolation path)"
            )

        if base_r.trend in ("rising", "falling") and base_r.slope is not None:
            reasons.append(
                f"Sustained {'upward' if base_r.trend=='rising' else 'downward'} "
                f"drift (slope={base_r.slope:+.4f}/tick)"
            )

        if (base_r.forecast_5 is not None and base_r.mean is not None
                and base_r.std and base_r.std > 1e-6):
            delta = base_r.forecast_5 - base_r.mean
            if abs(delta) > base_r.std * 1.5:
                reasons.append(
                    f"OLS 5-step forecast diverges {delta:+.2f}{base_r.unit} from mean"
                )

        return " | ".join(reasons) if reasons else "All indicators within normal bounds."


# ─────────────────────────────────────────────────────────────────────────────
# 6. Advanced ML Pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Physical ranges for RNN normalisation
_SENSOR_RANGES = {
    "temp":     ( 5.0,  50.0),
    "humidity": (10.0,  99.0),
    "soil":     ( 5.0,  95.0),
    "co2":      (250.0, 900.0),
    "light":    (100.0, 10000.0),
    "ph":       ( 3.5,   9.0),
}
_DECIMALS = {"temp":1, "humidity":1, "soil":1, "co2":0, "light":0, "ph":2}


class AdvancedMLPipeline:
    """
    Wraps baseline MLPipeline (unchanged) and adds advanced components
    per-sensor. Returns both baseline and advanced results every tick
    so performance can be compared in --mode compare.

    Usage
    -----
        pipeline = AdvancedMLPipeline()
        adv = pipeline.process(readings, tick)   # Dict[str, dict]
        score = pipeline.compute_health_score()   # from baseline
        comparison = pipeline.compare_with_baseline(ground_truth)
    """

    def __init__(self, window: int = 20):
        # Baseline (kept intact for fair comparison)
        self.baseline   = MLPipeline(window=window, zscore_threshold=2.5)

        # Advanced components (one per sensor, created lazily)
        self._dyn:  Dict[str, DynamicThresholdDetector] = {}
        self._hw:   Dict[str, HoltWintersForecaster]    = {}
        self._rnn:  Dict[str, MinimalRNN]               = {}
        self._prev: Dict[str, float]                    = {}  # for RNN training

        # Correlation trackers (cross-sensor pairs)
        self._corr = {
            ("temp", "humidity"): PearsonCorrelationTracker(expected_corr=-0.8),
            ("light", "temp"):    PearsonCorrelationTracker(expected_corr= 0.6),
            ("co2",  "temp"):     PearsonCorrelationTracker(expected_corr= 0.3),
        }

        self._adv_log: List[Dict] = []   # full advanced history

    # ── Core Process ──────────────────────────────────────────────────────

    def process(self, readings: Dict[str, SensorReading],
                tick: int) -> Dict[str, dict]:
        """
        Run baseline + advanced ML on one tick's readings.
        Returns dict[sensor_id → advanced_result_dict].
        """
        # 1. Baseline (unchanged)
        base_results = self.baseline.process(readings, tick)

        # 2. Update correlation trackers
        vals = {sid: r.value for sid, r in readings.items()}
        for (s1, s2), tracker in self._corr.items():
            if s1 in vals and s2 in vals:
                tracker.push(vals[s1], vals[s2])

        # 3. Per-sensor advanced processing
        adv: Dict[str, dict] = {}
        for sid, base_r in base_results.items():
            v   = base_r.value
            dec = _DECIMALS.get(sid, 1)
            lo, hi = _SENSOR_RANGES.get(sid, (0.0, 1.0))
            vn  = (v - lo) / (hi - lo) if hi > lo else 0.0   # normalised

            # — Dynamic threshold —
            if sid not in self._dyn:
                self._dyn[sid] = DynamicThresholdDetector()
            self._dyn[sid].push(v)
            dyn_th   = self._dyn[sid].threshold()
            dyn_anom = self._dyn[sid].is_anomaly(base_r.z_score) or base_r.if_anomaly

            # — Holt-Winters —
            if sid not in self._hw:
                self._hw[sid] = HoltWintersForecaster()
            self._hw[sid].update(v)
            hw_f5_raw = self._hw[sid].predict(5)
            # [FIX] Clamp forecast to physical bounds
            if hw_f5_raw is not None:
                hw_f5_raw = max(lo, min(hi, hw_f5_raw))
            hw_f5 = round(hw_f5_raw, dec) if hw_f5_raw is not None else None

            # — RNN (train on prev→curr, then predict ahead) —
            if sid not in self._rnn:
                self._rnn[sid] = MinimalRNN()
            if sid in self._prev:
                self._rnn[sid].train_step(self._prev[sid], vn)
            else:
                self._rnn[sid].forward(vn)  # warm up hidden state
            rnn_raw  = self._rnn[sid].predict_ahead(vn, steps=5)
            # [FIX] Clamp prediction before denormalizing
            rnn_raw = max(0.0, min(1.0, rnn_raw))
            rnn_f5   = round(rnn_raw * (hi - lo) + lo, dec)
            self._prev[sid] = vn

            # — Correlation alert —
            corr_alert: Optional[str] = None
            for (s1, s2), tracker in self._corr.items():
                if sid in (s1, s2) and tracker.is_decorrelated():
                    other = s2 if sid == s1 else s1
                    r_val = tracker.correlation()
                    corr_alert = (
                        f"Correlation {sid}↔{other} broke "
                        f"(r={r_val:.2f}, expected≈{tracker.expected_corr:.1f})"
                    )
                    break

            # — Explainability —
            reason = ExplainabilityEngine.explain(base_r, dynamic_threshold=dyn_th)

            # — Per-sensor confidence (0-100) —
            confidence = self._sensor_confidence(sid, base_r, dyn_anom)

            adv[sid] = {
                "base":              base_r,          # original MLResult
                "dynamic_threshold": dyn_th,
                "dynamic_anomaly":   dyn_anom,
                "holt_forecast_5":   hw_f5,
                "rnn_forecast_5":    rnn_f5,
                "reason":            reason,
                "confidence":        confidence,
                "correlation_alert": corr_alert,
            }

        self._adv_log.append({"tick": tick, "results": adv})
        return adv

    # ── Delegated helpers ─────────────────────────────────────────────────

    def compute_health_score(self) -> int:
        return self.baseline.compute_health_score()

    def get_summary(self) -> Dict:
        return self.baseline.get_summary()

    def evaluate(self, ground_truth: Dict[str, List[bool]]) -> Dict[str, dict]:
        """Allows AdvancedMLPipeline to be used drop-in where MLPipeline is expected."""
        return self.compare_with_baseline(ground_truth)["advanced"]

    # ── Confidence ────────────────────────────────────────────────────────

    def _sensor_confidence(self, sid: str, base_r: MLResult,
                            dyn_anom: bool) -> int:
        conf = 100
        if base_r.anomaly:                       conf -= 30
        if dyn_anom and not base_r.anomaly:      conf -= 10
        if base_r.severity == "critical":        conf -= 20
        if base_r.trend in ("rising", "falling") and base_r.anomaly:
                                                 conf -= 10
        recent = sum(self.baseline.anomaly_log.get(sid, []))
        conf -= min(15, recent // 2)
        return max(0, min(100, conf))

    # ── Comparison ────────────────────────────────────────────────────────

    def compare_with_baseline(self,
                               ground_truth: Dict[str, List[bool]]) -> Dict:
        """
        Evaluate both baseline and advanced anomaly flags against ground truth.
        Returns {'baseline': metrics, 'advanced': metrics} per sensor.
        """
        baseline_metrics = self.baseline.evaluate(ground_truth)

        # Build advanced predicted flags (dynamic_anomaly per tick)
        adv_predicted: Dict[str, List[bool]] = defaultdict(list)
        for tick_data in self._adv_log:
            for sid, r in tick_data["results"].items():
                adv_predicted[sid].append(r["dynamic_anomaly"])

        adv_metrics: Dict[str, dict] = {}
        for sid, predicted in adv_predicted.items():
            actual = ground_truth.get(sid, [])
            n = min(len(predicted), len(actual))
            if n == 0:
                continue
            tp = sum(p and a for p, a in zip(predicted[:n], actual[:n]))
            fp = sum(p and not a for p, a in zip(predicted[:n], actual[:n]))
            fn = sum(not p and a for p, a in zip(predicted[:n], actual[:n]))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
            adv_metrics[sid] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 3),
                "recall":    round(rec,  3),
                "f1":        round(f1,   3),
            }

        return {"baseline": baseline_metrics, "advanced": adv_metrics}
