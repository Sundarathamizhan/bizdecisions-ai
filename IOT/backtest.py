"""
backtest.py — ML Evaluation with Ground Truth Labels
═════════════════════════════════════════════════════
Fault types
───────────
  spike        — single tick at 95% max_val
  step         — sustained +4σ offset for 3–8 ticks
  drift        — linearly growing offset 0→5σ
  multi_sensor — spike injected into 3 random sensors simultaneously
  cascade      — temperature spike → forces humidity crash (2nd sensor)

Metrics
───────
  Precision, Recall, F1     (per sensor)
  Confusion matrix          (TP/TN/FP/FN, Accuracy, Specificity)
  ROC curve + AUC           (using |Z-score| as decision score)
"""

import math
import random
from typing import Dict, List, Optional

from iot_layer import SensorNetwork
from ml_layer  import MLPipeline


# ── ANSI helpers (no external deps) ─────────────────────────────────────────

_ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
}

def _c(color: str, text: str) -> str:
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


# ─────────────────────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Runs the full NeuralFarm pipeline over a synthetic sensor stream
    with structured fault injections and measures detection performance.

    Parameters
    ----------
    ticks : int
        Total simulation length in ticks (default: 200, as in the paper).
    seed  : int
        Random seed for reproducible fault placement (default: 42).

    Attributes
    ----------
    ground_truth  : Dict[str, List[bool]]
        Per-sensor boolean label list; True = anomalous tick.
    fault_windows : List[tuple]
        Each entry: (start_tick, end_tick, sensor_id, fault_type)
    metrics       : Dict[str, Dict]
        Precision / Recall / F1 per sensor after run().
    """

    def __init__(self, ticks: int = 200, seed: int = 42):
        self.ticks        = ticks
        self.seed         = seed
        self.network      = SensorNetwork()
        self.pipeline     = MLPipeline(window=20, zscore_threshold=2.5)
        self.ground_truth: Dict[str, List[bool]] = {
            sid: [False] * ticks for sid in self.network.sensors
        }
        self.fault_windows: List[tuple] = []
        self.metrics: Dict[str, Dict]   = {}
        random.seed(seed)

    # ── Fault Planning ────────────────────────────────────────────────────

    def _plan_faults(self):
        """
        Schedule 4 fault windows at random positions.
        Fault types: spike, step, drift, multi_sensor, cascade.
        """
        sensor_ids  = list(self.network.sensors.keys())
        fault_types = ["spike", "step", "drift", "multi_sensor", "cascade"]

        for _ in range(4):
            start = random.randint(25, self.ticks - 30)
            dur   = random.randint(3, 8)
            end   = min(start + dur, self.ticks - 1)
            sid   = random.choice(sensor_ids)
            ftype = random.choice(fault_types)

            self.fault_windows.append((start, end, sid, ftype))

            # Mark ground truth with ±1 tolerance
            for t in range(max(0, start - 1), min(end + 2, self.ticks)):
                self.ground_truth[sid][t] = True
                # cascade also affects humidity
                if ftype == "cascade" and sid == "temp" and "humidity" in self.ground_truth:
                    self.ground_truth["humidity"][t] = True
                # multi_sensor: affect 2 additional random sensors
                if ftype == "multi_sensor":
                    for extra in random.sample(
                        [s for s in sensor_ids if s != sid], min(2, len(sensor_ids)-1)
                    ):
                        if 0 <= t < self.ticks:
                            self.ground_truth[extra][t] = True

    # ── Simulation Loop ───────────────────────────────────────────────────

    def run(self):
        """
        Simulate all ticks, inject faults per plan, process through ML pipeline.
        Call print_results() afterwards to display the evaluation table.
        """
        self._plan_faults()
        print(f"\n[BT]  Starting {self.ticks}-tick backtest (seed={self.seed})...")
        print(f"[BT]  Injecting {len(self.fault_windows)} fault windows...\n")

        for tick in range(1, self.ticks + 1):
            readings = self.network.read_all(tick)

            # Inject faults according to plan
            for start, end, sid, ftype in self.fault_windows:
                if start <= tick <= end and sid in readings:
                    r      = readings[sid]
                    sensor = self.network.sensors[sid]

                    if ftype == "spike":
                        readings[sid].value = round(
                            sensor.cfg.max_val * 0.95, sensor.cfg.decimals)

                    elif ftype == "step":
                        readings[sid].value = round(
                            min(r.value + sensor.cfg.noise_amplitude * 4,
                                sensor.cfg.max_val), sensor.cfg.decimals)

                    elif ftype == "drift":
                        frac = (tick - start) / max(end - start, 1)
                        readings[sid].value = round(
                            min(r.value + frac * sensor.cfg.noise_amplitude * 5,
                                sensor.cfg.max_val), sensor.cfg.decimals)

                    elif ftype == "multi_sensor":
                        readings[sid].value = round(
                            sensor.cfg.max_val * 0.95, sensor.cfg.decimals)
                        others = [s for s in readings if s != sid]
                        for extra in random.sample(others, min(2, len(others))):
                            es = self.network.sensors[extra]
                            readings[extra].value = round(
                                es.cfg.max_val * 0.92, es.cfg.decimals)
                            readings[extra].is_injected_anomaly = True

                    elif ftype == "cascade":
                        readings[sid].value = round(
                            min(r.value + sensor.cfg.noise_amplitude * 5,
                                sensor.cfg.max_val), sensor.cfg.decimals)
                        if sid == "temp" and "humidity" in readings:
                            hs = self.network.sensors["humidity"]
                            readings["humidity"].value = round(
                                max(readings["humidity"].value - 25.0,
                                    hs.cfg.min_val), hs.cfg.decimals)
                            readings["humidity"].is_injected_anomaly = True

                    readings[sid].is_injected_anomaly = True

            self.pipeline.process(readings, tick)

        self.metrics = self.pipeline.evaluate(self.ground_truth)

    # ── Results Display ───────────────────────────────────────────────────

    def print_results(self):
        """Print the full backtest evaluation table to stdout."""
        if not self.metrics:
            print("[BT] No metrics — did you call run() first?")
            return

        print()
        print(_c("bold", "═" * 68))
        print(_c("bold", "  NeuralFarm  ·  Backtest Evaluation Results"))
        print(_c("bold", "═" * 68))
        print(f"  Simulation ticks : {_c('cyan', str(self.ticks))}")
        print(f"  Random seed      : {_c('cyan', str(self.seed))}")
        print(f"  Fault windows    : {_c('cyan', str(len(self.fault_windows)))}")
        print()

        header = (f"  {'SENSOR':<20} {'TP':>4} {'FP':>4} {'FN':>4}"
                  f" {'PRECISION':>10} {'RECALL':>8} {'F1':>8}")
        print(_c("bold", header))
        print("  " + "─" * 64)

        f1_scores = []
        for sid, m in self.metrics.items():
            f1   = m["f1"];  prec = m["precision"];  rec = m["recall"]
            f1_col = "green" if f1 >= 0.7 else ("yellow" if f1 >= 0.4 else "red")
            print(f"  {sid:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}"
                  f" {prec:>10.3f} {rec:>8.3f}"
                  f" {_c(f1_col, f'{f1:.3f}'):>8}")
            f1_scores.append(f1)

        avg_f1  = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_col = "green" if avg_f1 >= 0.7 else ("yellow" if avg_f1 >= 0.4 else "red")
        print("  " + "─" * 64)
        print(f"  {'Macro avg F1':<20} {'':>4} {'':>4} {'':>4}"
              f" {'':>10} {'':>8} {_c(avg_col, f'{avg_f1:.3f}'):>8}")

        print()
        print(_c("bold", "  Injected Fault Windows"))
        print("  " + "─" * 64)
        for i, (start, end, sid, ftype) in enumerate(self.fault_windows, 1):
            fc = {"spike":"red","step":"yellow","drift":"cyan",
                  "multi_sensor":"green","cascade":"cyan"}.get(ftype, "reset")
            print(f"  [{i}] ticks {start:>3}–{end:<3}  "
                  f"sensor={sid:<12}  type={_c(fc, ftype)}")
        print()
        print(_c("bold", "═" * 68))
        print()

    # ── ROC / AUC ─────────────────────────────────────────────────────────

    def compute_roc_auc(self) -> Dict[str, dict]:
        """Compute ROC curve + AUC per sensor using |Z-score| as score."""
        roc = {}
        for sid in self.ground_truth:
            scores  = [
                abs(row[sid].z_score) if (sid in row and row[sid].z_score is not None) else 0.0
                for row in self.pipeline.results_log
            ]
            actuals = self.ground_truth[sid]
            n       = min(len(scores), len(actuals))
            if n == 0: continue
            n_pos = sum(actuals[:n]);  n_neg = n - n_pos
            if n_pos == 0 or n_neg == 0: continue
            pairs   = sorted(zip(scores[:n], actuals[:n]), key=lambda x: -x[0])
            fpr_pts = [0.0];  tpr_pts = [0.0];  tp = fp = 0
            for score, label in pairs:
                if label: tp += 1
                else:     fp += 1
                tpr_pts.append(tp / n_pos);  fpr_pts.append(fp / n_neg)
            fpr_pts.append(1.0);  tpr_pts.append(1.0)
            auc = sum(
                (fpr_pts[i+1]-fpr_pts[i]) * (tpr_pts[i+1]+tpr_pts[i]) / 2
                for i in range(len(fpr_pts)-1)
            )
            roc[sid] = {"fpr": fpr_pts, "tpr": tpr_pts, "auc": round(auc, 4)}
        return roc

    # ── Confusion Matrix ──────────────────────────────────────────────────

    def confusion_matrix(self) -> Dict[str, dict]:
        """Full confusion matrix: TP/TN/FP/FN, Accuracy, Specificity."""
        cm = {}
        for sid, m in self.metrics.items():
            tp, fp, fn = m["tp"], m["fp"], m["fn"]
            tn  = max(0, self.ticks - tp - fp - fn)
            cm[sid] = {
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "accuracy":    round((tp+tn) / max(self.ticks, 1), 3),
                "specificity": round(tn / max(tn+fp, 1), 3),
            }
        return cm

    def print_advanced_metrics(self):
        """Print ROC AUC + confusion matrix (call after run())."""
        if not self.metrics:
            print("[BT] Run run() first."); return
        roc = self.compute_roc_auc();  cm = self.confusion_matrix()
        print()
        print(_c("bold", "═" * 68))
        print(_c("bold", "  Advanced Metrics: AUC + Confusion Matrix"))
        print(_c("bold", "═" * 68))
        print(_c("bold", f"  {'SENSOR':<15} {'AUC':>6} {'ACC':>7} {'SPEC':>7}"
                          f" {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}"))
        print("  " + "─" * 58)
        aucs = []
        for sid in cm:
            auc_v   = roc.get(sid, {}).get("auc", 0.0);  aucs.append(auc_v)
            m       = cm[sid]
            auc_col = "green" if auc_v >= 0.8 else ("yellow" if auc_v >= 0.6 else "red")
            print(f"  {sid:<15} {_c(auc_col, f'{auc_v:.3f}'):>6}"
                  f" {m['accuracy']:>7.3f} {m['specificity']:>7.3f}"
                  f" {m['TP']:>5} {m['TN']:>5} {m['FP']:>5} {m['FN']:>5}")
        avg_auc = sum(aucs) / len(aucs) if aucs else 0
        print("  " + "─" * 58)
        print(f"  {'Macro avg AUC':<15} {avg_auc:>6.3f}")
        print(_c("bold", "═" * 68))
        print()


    # ── Results Display ───────────────────────────────────────────────────

    def print_results(self):
        """Print the full backtest evaluation table to stdout."""
        if not self.metrics:
            print("[BT] No metrics — did you call run() first?")
            return

        print()
        print(_c("bold", "═" * 68))
        print(_c("bold", "  NeuralFarm  ·  Backtest Evaluation Results"))
        print(_c("bold", "═" * 68))
        print(f"  Simulation ticks : {_c('cyan', str(self.ticks))}")
        print(f"  Random seed      : {_c('cyan', str(self.seed))}")
        print(f"  Fault windows    : {_c('cyan', str(len(self.fault_windows)))}")
        print()

        # Per-sensor table
        header = (
            f"  {'SENSOR':<20} {'TP':>4} {'FP':>4} {'FN':>4}"
            f" {'PRECISION':>10} {'RECALL':>8} {'F1':>8}"
        )
        print(_c("bold", header))
        print("  " + "─" * 64)

        f1_scores = []
        for sid, m in self.metrics.items():
            f1   = m["f1"]
            prec = m["precision"]
            rec  = m["recall"]
            f1_col = "green" if f1 >= 0.7 else ("yellow" if f1 >= 0.4 else "red")
            print(
                f"  {sid:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}"
                f" {prec:>10.3f} {rec:>8.3f}"
                f" {_c(f1_col, f'{f1:.3f}'):>8}"
            )
            f1_scores.append(f1)

        # Macro average
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        print("  " + "─" * 64)
        avg_col = "green" if avg_f1 >= 0.7 else ("yellow" if avg_f1 >= 0.4 else "red")
        print(
            f"  {'Macro avg F1':<20} {'':>4} {'':>4} {'':>4}"
            f" {'':>10} {'':>8}"
            f" {_c(avg_col, f'{avg_f1:.3f}'):>8}"
        )

        # Fault window log
        print()
        print(_c("bold", "  Injected Fault Windows"))
        print("  " + "─" * 64)
        for i, (start, end, sid, ftype) in enumerate(self.fault_windows, 1):
            ftype_col = {
                "spike": "red",
                "step":  "yellow",
                "drift": "cyan",
            }.get(ftype, "reset")
            print(
                f"  [{i}] ticks {start:>3}–{end:<3}  "
                f"sensor={sid:<12}  "
                f"type={_c(ftype_col, ftype)}"
            )

        print()
        print(_c("bold", "═" * 68))
        print()
