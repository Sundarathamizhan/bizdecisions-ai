"""
data_export.py — CSV and JSON export of NeuralFarm simulation data
═══════════════════════════════════════════════════════════════════
Exports the full sensor + ML results log to disk in two formats:

  1. CSV  — flat, row-per-sensor-per-tick table (easy to open in Excel/pandas)
  2. JSON — structured nested report with metadata and ML summary

Both filenames include a timestamp so multiple exports don't overwrite each other.

Usage
─────
  python main.py --mode export --ticks 100 --out my_run
  # Produces: my_run_20260401_130000.csv
  #           my_run_20260401_130000.json
"""

import csv
import json
import os
import time
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# DataExporter
# ─────────────────────────────────────────────────────────────────────────────

class DataExporter:
    """
    Serialises NeuralFarm simulation results to disk.

    Parameters
    ----------
    prefix : str
        Filename prefix for all output files (default: "neuralfarm_output").

    Methods
    -------
    to_csv(records)           → str  (path to written CSV file)
    to_json(records, summary) → str  (path to written JSON file)
    """

    # CSV column order
    CSV_FIELDS = [
        "tick",
        "sensor_id",
        "sensor_name",
        "value",
        "unit",
        "mean",
        "std",
        "z_score",
        "z_anomaly",
        "if_score",
        "if_anomaly",
        "anomaly",
        "severity",
        "trend",
        "slope",
        "forecast_5",
    ]

    def __init__(self, prefix: str = "neuralfarm_output"):
        self.prefix    = prefix
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    # ── CSV Export ────────────────────────────────────────────────────────

    def to_csv(self, records: List[Dict]) -> str:
        """
        Write a flat CSV with one row per (tick × sensor).

        Parameters
        ----------
        records : list of dicts with keys: 'tick', 'readings', 'ml'
                  As built by run_export() in main.py.

        Returns
        -------
        str — absolute path to the written file.
        """
        path = f"{self.prefix}_{self.timestamp}.csv"

        # Map sensor_id → display name from first available reading
        sensor_names: Dict[str, str] = {}
        if records:
            first_readings = records[0].get("readings", {})
            for sid, reading in first_readings.items():
                sensor_names[sid] = getattr(reading, "sensor_id", sid)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()

            for record in records:
                tick = record["tick"]
                ml   = record.get("ml", {})

                for sid, r in ml.items():
                    writer.writerow({
                        "tick":        tick,
                        "sensor_id":   sid,
                        "sensor_name": sensor_names.get(sid, sid),
                        "value":       r.value,
                        "unit":        r.unit,
                        "mean":        r.mean        if r.mean        is not None else "",
                        "std":         r.std         if r.std         is not None else "",
                        "z_score":     r.z_score     if r.z_score     is not None else "",
                        "z_anomaly":   int(r.z_anomaly),
                        "if_score":    r.if_score    if r.if_score    is not None else "",
                        "if_anomaly":  int(r.if_anomaly),
                        "anomaly":     int(r.anomaly),
                        "severity":    r.severity,
                        "trend":       r.trend,
                        "slope":       r.slope       if r.slope       is not None else "",
                        "forecast_5":  r.forecast_5  if r.forecast_5  is not None else "",
                    })

        return os.path.abspath(path)

    # ── JSON Export ───────────────────────────────────────────────────────

    def to_json(self, records: List[Dict], ml_summary: Dict) -> str:
        """
        Write a structured JSON report with metadata, ML summary,
        and full time-series data.

        Parameters
        ----------
        records    : same list used by to_csv()
        ml_summary : output of MLPipeline.get_summary()

        Returns
        -------
        str — absolute path to the written file.
        """
        path = f"{self.prefix}_{self.timestamp}.json"

        # Build time-series array
        time_series = []
        for record in records:
            tick = record["tick"]
            ml   = record.get("ml", {})

            time_series.append({
                "tick": tick,
                "sensors": {
                    sid: {
                        "value":      r.value,
                        "unit":       r.unit,
                        "mean":       r.mean,
                        "std":        r.std,
                        "z_score":    r.z_score,
                        "z_anomaly":  r.z_anomaly,
                        "if_score":   r.if_score,
                        "if_anomaly": r.if_anomaly,
                        "anomaly":    r.anomaly,
                        "severity":   r.severity,
                        "trend":      r.trend,
                        "slope":      r.slope,
                        "forecast_5": r.forecast_5,
                    }
                    for sid, r in ml.items()
                },
            })

        # Aggregate stats across all ticks
        all_anomalies = sum(
            1
            for record in records
            for r in record.get("ml", {}).values()
            if r.anomaly
        )
        total_readings = sum(len(record.get("ml", {})) for record in records)

        report = {
            "metadata": {
                "project":        "NeuralFarm",
                "version":        "1.0",
                "description":    "IoT–ML–AI Smart Greenhouse Simulation",
                "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ticks":          len(records),
                "sensors":        6,
                "total_readings": total_readings,
                "total_anomalies": all_anomalies,
                "anomaly_rate":   round(all_anomalies / total_readings, 4)
                                  if total_readings > 0 else 0,
            },
            "ml_summary":  ml_summary,
            "time_series": time_series,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return os.path.abspath(path)

    # ── Summary Print ─────────────────────────────────────────────────────

    def print_export_summary(self, csv_path: str, json_path: str,
                             n_ticks: int, n_anomalies: int):
        """Print a short export summary to stdout."""
        print()
        print("┌─ NeuralFarm Export Complete " + "─" * 38)
        print(f"│  CSV  → {csv_path}")
        print(f"│  JSON → {json_path}")
        print(f"│  Ticks exported : {n_ticks}")
        print(f"│  Anomalies logged: {n_anomalies}")
        print("└" + "─" * 66)
        print()
