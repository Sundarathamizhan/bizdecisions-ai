"""
dashboard.py — Terminal Dashboard
══════════════════════════════════
Rich text report printed to stdout.
No external dependencies — pure Python f-strings and ANSI codes.
"""

from typing import Dict, List
from ml_layer import MLResult, HistoryRecord


ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "dim":    "\033[2m",
}

def c(color: str, text: str) -> str:
    return f"{ANSI[color]}{text}{ANSI['reset']}"


class Dashboard:
    """Collects results and prints a structured terminal report."""

    def __init__(self):
        self._records: List[HistoryRecord] = []

    def update(self, tick: int, readings: dict, results: Dict[str, MLResult]):
        from ml_layer import HistoryRecord
        self._records.append(HistoryRecord(tick, results))

    def update_bulk(self, records: List[HistoryRecord]):
        self._records = records

    def print_report(self):
        if not self._records:
            print("No data to report.")
            return

        total_ticks    = len(self._records)
        all_results    = [r for rec in self._records for r in rec.results.values()]
        total_anomalies= sum(1 for r in all_results if r.anomaly)
        critical_count = sum(1 for r in all_results if r.severity == "critical")

        print()
        print(c("bold", "═" * 68))
        print(c("bold", "  NeuralFarm  ·  Simulation Report"))
        print(c("bold", "═" * 68))
        print(f"  Ticks processed : {c('cyan', str(total_ticks))}")
        print(f"  Total readings  : {c('cyan', str(len(all_results)))}")
        print(f"  Anomalies caught: {c('yellow', str(total_anomalies))}")
        print(f"  Critical alerts : {c('red', str(critical_count))}")
        print()

        # Per-sensor summary
        sensor_ids = list(self._records[-1].results.keys())
        print(c("bold", f"  {'SENSOR':<20} {'LATEST':>8} {'MEAN':>8} {'STD':>6} {'TREND':>8} {'FORECAST':>10} {'ANOMALIES':>10}"))
        print("  " + "─" * 66)

        for sid in sensor_ids:
            # Gather all results for this sensor
            s_results = [rec.results[sid] for rec in self._records if sid in rec.results]
            if not s_results:
                continue
            latest    = s_results[-1]
            n_anomaly = sum(1 for r in s_results if r.anomaly)
            trend_sym = {"rising": "↑", "falling": "↓", "stable": "→"}.get(latest.trend, "?")
            trend_col = {"rising": "yellow", "falling": "red", "stable": "green"}.get(latest.trend, "dim")

            anom_str  = c("red", str(n_anomaly)) if n_anomaly > 0 else c("green", "0")
            lat_str   = f"{latest.value}{latest.unit}"
            mean_str  = f"{latest.mean}{latest.unit}" if latest.mean else "—"
            std_str   = str(latest.std) if latest.std else "—"
            fore_str  = f"{latest.forecast_5}{latest.unit}" if latest.forecast_5 else "—"

            print(
                f"  {sid:<20} {lat_str:>8} {mean_str:>8} {std_str:>6} "
                f"{c(trend_col, trend_sym + ' ' + latest.trend):>8}  "
                f"{fore_str:>10} {anom_str:>10}"
            )

        # Anomaly log
        anomaly_records = [
            (rec.tick, sid, rec.results[sid])
            for rec in self._records
            for sid in rec.results
            if rec.results[sid].anomaly
        ]
        if anomaly_records:
            print()
            print(c("bold", "  ANOMALY LOG (last 10)"))
            print("  " + "─" * 66)
            for tick, sid, r in anomaly_records[-10:]:
                sev_col = "red" if r.severity == "critical" else "yellow"
                print(
                    f"  Tick {tick:>4}  {sid:<15} value={r.value}{r.unit:<5}  "
                    f"z={r.z_score:>5}σ  IF={r.if_score:.2f}  "
                    f"{c(sev_col, r.severity.upper())}"
                )

        print()
        print(c("bold", "═" * 68))
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────────────────────

"""
backtest.py — ML Evaluation with Ground Truth Labels
═════════════════════════════════════════════════════
Generates synthetic dataset with KNOWN anomaly windows,
then measures detection accuracy: precision / recall / F1.
"""

import random
from iot_layer import SensorNetwork
from ml_layer  import MLPipeline


class Backtester:
    """
    Injects structured fault windows into the sensor network
    and evaluates whether the ML pipeline correctly detects them.

    Fault types injected
    ─────────────────────
      • Step fault   : sustained offset (e.g. irrigation failure)
      • Spike fault  : single large outlier (sensor glitch)
      • Drift fault  : slowly accumulating offset (sensor calibration drift)
    """

    def __init__(self, ticks: int = 200, seed: int = 42):
        self.ticks   = ticks
        self.seed    = seed
        self.network = SensorNetwork()
        self.pipeline= MLPipeline(window=20)
        self.ground_truth: Dict[str, List[bool]] = {
            sid: [False] * ticks for sid in self.network.sensors
        }
        self.fault_windows = []
        self.metrics = {}
        random.seed(seed)

    def _plan_faults(self):
        """Plan 3–5 fault windows at random ticks."""
        sensor_ids = list(self.network.sensors.keys())
        for _ in range(4):
            start  = random.randint(25, self.ticks - 30)
            dur    = random.randint(3, 8)
            sid    = random.choice(sensor_ids)
            ftype  = random.choice(["spike", "step", "drift"])
            self.fault_windows.append((start, start + dur, sid, ftype))
            for t in range(start, min(start + dur, self.ticks)):
                self.ground_truth[sid][t] = True

    def run(self):
        """Simulate all ticks and collect results."""
        self._plan_faults()

        for tick in range(1, self.ticks + 1):
            readings = self.network.read_all(tick)

            # Inject faults according to plan
            for start, end, sid, ftype in self.fault_windows:
                if start <= tick <= end and sid in readings:
                    r = readings[sid]
                    sensor = self.network.sensors[sid]
                    if ftype == "spike":
                        readings[sid].value = sensor.cfg.max_val * 0.95
                    elif ftype == "step":
                        readings[sid].value = min(
                            r.value + sensor.cfg.noise_amplitude * 4,
                            sensor.cfg.max_val
                        )
                    elif ftype == "drift":
                        frac = (tick - start) / max(end - start, 1)
                        readings[sid].value = min(
                            r.value + frac * sensor.cfg.noise_amplitude * 5,
                            sensor.cfg.max_val
                        )
                    readings[sid].is_injected_anomaly = True

            self.pipeline.process(readings, tick)

        self.metrics = self.pipeline.evaluate(self.ground_truth)

    def print_results(self):
        print()
        print(c("bold", "═" * 68))
        print(c("bold", "  NeuralFarm Backtest Results"))
        print(c("bold", "═" * 68))
        print(f"  Ticks: {self.ticks}   Fault windows: {len(self.fault_windows)}")
        print(f"  Random seed: {self.seed}")
        print()
        print(c("bold", f"  {'SENSOR':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'PRECISION':>10} {'RECALL':>8} {'F1':>6}"))
        print("  " + "─" * 60)

        f1_scores = []
        for sid, m in self.metrics.items():
            f1_col = "green" if m["f1"] >= 0.7 else "yellow" if m["f1"] >= 0.4 else "red"
            print(
                f"  {sid:<20} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
                f"{m['precision']:>10.3f} {m['recall']:>8.3f} "
                f"{c(f1_col, f'{m[chr(102)+chr(49)]:.3f}'):>6}"
            )
            f1_scores.append(m["f1"])

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        print("  " + "─" * 60)
        print(f"  {'Macro avg F1':<20} {avg_f1:>52.3f}")
        print()

        print("  Fault windows injected:")
        for start, end, sid, ftype in self.fault_windows:
            print(f"    ticks {start}–{end}  sensor={sid}  type={ftype}")
        print()
        print(c("bold", "═" * 68))


# ─────────────────────────────────────────────────────────────────────────────
# Data Exporter
# ─────────────────────────────────────────────────────────────────────────────

"""
data_export.py — CSV and JSON export of sensor + ML data.
"""

import csv
import json
import time
from typing import List, Dict


class DataExporter:
    """
    Exports simulation results to CSV (flat sensor log)
    and JSON (structured ML report with metadata).
    """

    def __init__(self, prefix: str = "neuralfarm_output"):
        self.prefix    = prefix
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def to_csv(self, records: List[Dict]) -> str:
        path = f"{self.prefix}_{self.timestamp}.csv"
        fieldnames = [
            "tick", "sensor_id", "value", "unit",
            "mean", "std", "z_score", "z_anomaly",
            "if_score", "if_anomaly", "anomaly", "severity",
            "trend", "slope", "forecast_5",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                tick = record["tick"]
                for sid, r in record["ml"].items():
                    writer.writerow({
                        "tick":        tick,
                        "sensor_id":   sid,
                        "value":       r.value,
                        "unit":        r.unit,
                        "mean":        r.mean,
                        "std":         r.std,
                        "z_score":     r.z_score,
                        "z_anomaly":   r.z_anomaly,
                        "if_score":    r.if_score,
                        "if_anomaly":  r.if_anomaly,
                        "anomaly":     r.anomaly,
                        "severity":    r.severity,
                        "trend":       r.trend,
                        "slope":       r.slope,
                        "forecast_5":  r.forecast_5,
                    })
        return path

    def to_json(self, records: List[Dict], ml_summary: Dict) -> str:
        path = f"{self.prefix}_{self.timestamp}.json"
        report = {
            "metadata": {
                "project":   "NeuralFarm",
                "version":   "1.0",
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ticks":     len(records),
            },
            "ml_summary":  ml_summary,
            "time_series": [
                {
                    "tick": rec["tick"],
                    "sensors": {
                        sid: {
                            "value":     r.value,
                            "anomaly":   r.anomaly,
                            "severity":  r.severity,
                            "trend":     r.trend,
                            "forecast":  r.forecast_5,
                        }
                        for sid, r in rec["ml"].items()
                    }
                }
                for rec in records
            ],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return path
