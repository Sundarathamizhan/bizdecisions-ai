"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NeuralFarm — IoT + AI + ML Smart Greenhouse System             ║
║                         Full Python Implementation                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE:
  Layer 1 — IoT     : Simulated sensor streams (6 channels, real physics)
  Layer 2 — ML      : Anomaly detection, forecasting, health scoring
  Layer 3 — AI      : Claude API for natural language crop diagnostics

Run modes:
  python main.py --mode demo        # one-shot demo with report
  python main.py --mode live        # continuous live monitoring
  python main.py --mode backtest    # run ML on historical data
  python main.py --mode export      # save CSV + JSON report
  python main.py --mode advanced    # demo with ALL advanced ML features
  python main.py --mode compare     # baseline vs advanced F1 comparison
"""

import argparse
import sys
from iot_layer    import SensorNetwork
from ml_layer     import MLPipeline
from ai_layer     import AIAdvisor
from dashboard    import Dashboard
from data_export  import DataExporter
from database     import NeuralFarmDB
from digital_twin import DigitalTwin


def parse_args():
    p = argparse.ArgumentParser(description="NeuralFarm Smart Greenhouse System")
    p.add_argument("--mode",
                   choices=["demo", "live", "backtest", "export", "advanced", "compare"],
                   default="demo", help="Run mode (default: demo)")
    p.add_argument("--ticks",    type=int,   default=60,
                   help="Number of sensor ticks to simulate (default: 60)")
    p.add_argument("--interval", type=float, default=0.5,
                   help="Seconds between ticks for live mode (default: 0.5)")
    p.add_argument("--ai",       action="store_true",
                   help="Enable Claude AI diagnostic (requires ANTHROPIC_API_KEY)")
    p.add_argument("--out",      default="neuralfarm_output",
                   help="Output file prefix for export mode")
    return p.parse_args()


def run_demo(args):
    """
    Demo mode: simulate N ticks → run full ML pipeline → optional AI report.
    Prints a structured report to stdout.
    """
    print("\n" + "═" * 68)
    print("  NeuralFarm  ·  Smart Greenhouse Intelligence  ·  Demo Mode")
    print("═" * 68)

    # ── 1. Boot IoT sensor network ──────────────────────────────────────
    network = SensorNetwork()
    print(f"\n[IoT]  Booting {len(network.sensors)} sensor channels...")
    for s in network.sensors.values():
        print(f"       ├─ {s.name:<20} range [{s.min_val}, {s.max_val}] {s.unit}")

    # ── 2. Simulate ticks ───────────────────────────────────────────────
    ml        = MLPipeline(window=20, zscore_threshold=2.5)
    dashboard = Dashboard()
    db        = NeuralFarmDB("neuralfarm.db")
    twin      = DigitalTwin("tomato")

    print(f"\n[SIM]  Streaming {args.ticks} sensor ticks...\n")
    twin_metrics = {}
    for tick in range(1, args.ticks + 1):
        readings     = network.read_all(tick)
        results      = ml.process(readings, tick)
        dashboard.update(tick, readings, results)
        db.insert_telemetry(tick, results)
        twin_metrics = twin.update(readings)

    # ── 3. Final report ─────────────────────────────────────────────────
    dashboard.print_report()

    # FIX: DigitalTwin.update() returns 'stresses' (list), not 'stress_level'
    print("\n[TWIN] Final Crop Status:")
    stresses   = twin_metrics.get("stresses", [])
    stress_str = stresses[0] if stresses else "None"
    print(f"       Yield: {twin_metrics.get('yield_pct', 0.0):.1f}%  |  "
          f"Stress: {stress_str}")
    print(f"       Days elapsed: {twin_metrics.get('days_elapsed', 0):.1f}  |  "
          f"GDD: {twin_metrics.get('total_gdd', 0.0):.2f}")

    # ── 4. Optional AI analysis ─────────────────────────────────────────
    if args.ai:
        advisor = AIAdvisor()
        summary = ml.get_summary()
        score   = ml.compute_health_score()
        print("\n[AI]   Requesting Claude diagnostic...\n")

        # diagnose() returns DiagnosticReport (not a plain string)
        report = advisor.diagnose(summary, score)
        print(f"  Confidence : {report.confidence}/100")

        # format_actions expects Dict[str, List[dict]] — use advisor.decisions
        if report.actions:
            actions_by_sensor = advisor.decisions.evaluate(summary)
            formatted = advisor.decisions.format_actions(actions_by_sensor)
            print(f"  Auto-Actions:\n{formatted}")

        print("┌─ Claude AI Crop Diagnostic " + "─" * 40)
        for line in report.raw_insight.split("\n"):
            print(f"│  {line}")
        print("└" + "─" * 68)


def run_live(args):
    """
    Live mode: continuous streaming loop with real-time terminal updates.
    Press Ctrl+C to stop.
    """
    import time
    network = SensorNetwork()
    ml      = MLPipeline(window=20, zscore_threshold=2.5)
    advisor = AIAdvisor() if args.ai else None
    db      = NeuralFarmDB("neuralfarm.db")
    twin    = DigitalTwin("tomato")

    print("\n[LIVE] NeuralFarm monitoring active. Press Ctrl+C to stop.\n")
    tick = 0
    try:
        while True:
            tick += 1
            readings     = network.read_all(tick)
            results      = ml.process(readings, tick)
            db.insert_telemetry(tick, results)
            twin_metrics = twin.update(readings)

            score  = ml.compute_health_score()

            # FIX: MLPipeline returns MLResult objects → use attributes, not dict keys
            alerts = sum(1 for r in results.values() if r.anomaly)
            line   = f"Tick {tick:>4}  Health={score:>3}/100  Alerts={alerts}"
            for sid, r in results.items():
                a    = "!" if r.anomaly else " "
                line += f"  {sid[:4]}={r.value}{a}"

            # FIX: stresses is a list, not a scalar key 'stress_level'
            stresses = twin_metrics.get("stresses", [])
            line    += f"  | Yield={twin_metrics.get('yield_pct', 0.0):.1f}%"
            if stresses:
                line += f"  ⚠ {stresses[0]}"
            print(line)

            # Every 30 ticks, optionally ping AI
            if args.ai and advisor and tick % 30 == 0:
                summary = ml.get_summary()
                # FIX: diagnose() returns DiagnosticReport, not a plain string
                report = advisor.diagnose(summary, score)
                print("\n  [AI DIGEST] " + report.raw_insight.split("\n")[0] + "\n")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[LIVE] Stopped. Printing final summary...\n")
        d = Dashboard()
        d.update_bulk(ml.history)
        d.print_report()


def run_backtest(args):
    """
    Backtest mode: inject synthetic historical dataset with known anomaly
    windows, then evaluate ML detection accuracy (precision / recall / F1).
    """
    from backtest import Backtester
    bt = Backtester(ticks=args.ticks)
    bt.run()
    bt.print_results()
    bt.print_advanced_metrics()


def run_export(args):
    """
    Export mode: simulate → save CSV sensor log + JSON ML report.
    """
    network = SensorNetwork()
    ml      = MLPipeline(window=20, zscore_threshold=2.5)
    records = []

    for tick in range(1, args.ticks + 1):
        readings = network.read_all(tick)
        results  = ml.process(readings, tick)
        records.append({"tick": tick, "readings": readings, "ml": results})

    exporter  = DataExporter(prefix=args.out)
    csv_path  = exporter.to_csv(records)
    json_path = exporter.to_json(records, ml.get_summary())

    n_anomalies = sum(
        1 for rec in records
        for r in rec["ml"].values() if r.anomaly
    )
    exporter.print_export_summary(csv_path, json_path, args.ticks, n_anomalies)


# ─────────────────────────────────────────────────────────────────────────────
# Advanced mode
# ─────────────────────────────────────────────────────────────────────────────

def run_advanced(args):
    """
    Advanced mode: runs AdvancedMLPipeline (Dynamic Threshold + Holt-Winters
    + RNN + Correlation ML + Explainability) with optional AI advisory.
    """
    from ml_advanced import AdvancedMLPipeline
    from ai_layer    import AIAdvisor, AutoDecisionEngine

    print("\n" + "═" * 68)
    print("  NeuralFarm  ·  ADVANCED MODE  (Dynamic + HW + RNN + Correlation)")
    print("═" * 68)

    network  = SensorNetwork()
    pipeline = AdvancedMLPipeline(window=20)
    advisor  = AIAdvisor() if args.ai else None
    engine   = AutoDecisionEngine()
    twin     = DigitalTwin("tomato")
    db       = NeuralFarmDB("neuralfarm.db")

    print(f"\n[ADV] Streaming {args.ticks} ticks...\n")
    for tick in range(1, args.ticks + 1):
        readings     = network.read_all(tick)
        adv          = pipeline.process(readings, tick)
        twin_metrics = twin.update(readings)
        db.insert_telemetry(tick, adv)

        if tick == args.ticks:              # print final tick detail
            score = pipeline.compute_health_score()
            print(f"\n{'='*68}")
            print(f"  Final tick={tick}   Health={score}/100")
            print(f"  {'SENSOR':<12} {'VALUE':>8} {'DYN_THR':>8} "
                  f"{'HW_F5':>8} {'RNN_F5':>8} {'CONF':>6} {'ANOM':>6}")
            print("  " + "-" * 64)
            for sid, r in adv.items():
                base = r["base"]
                print(
                    f"  {sid:<12} {base.value:>8} "
                    f"{r['dynamic_threshold']:>8.2f} "
                    f"{str(r['holt_forecast_5']):>8} "
                    f"{str(r['rnn_forecast_5']):>8} "
                    f"{r['confidence']:>6}% "
                    f"{'YES' if r['dynamic_anomaly'] else 'no':>6}"
                )
                if r["reason"] != "All indicators within normal bounds.":
                    print(f"    └─ REASON: {r['reason']}")
                if r["correlation_alert"]:
                    print(f"    └─ CORR  : {r['correlation_alert']}")
            print()

            # Digital twin summary
            stresses = twin_metrics.get("stresses", [])
            print(f"  [TWIN] Yield: {twin_metrics.get('yield_pct', 0.0):.2f}%  |  "
                  f"GDD: {twin_metrics.get('total_gdd', 0.0):.2f}  |  "
                  f"Stress events: {len(stresses)}")
            print()

            # Auto-decision engine
            summary = pipeline.get_summary()
            actions = engine.evaluate(summary)
            print("  AUTO-DECISIONS:")
            print(engine.format_actions(actions) or "  No automated actions triggered.")

            # Optional AI report
            if args.ai and advisor:
                print("\n[AI] Requesting Claude diagnostic...")
                report = advisor.diagnose(summary, score)
                print(f"\n  Confidence : {report.confidence}/100")
                print("┌─ Claude AI Diagnostic " + "─" * 46)
                for line in report.raw_insight.split("\n"):
                    print(f"│  {line}")
                print("└" + "─" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# Compare mode
# ─────────────────────────────────────────────────────────────────────────────

def run_compare(args):
    """
    Compare mode: injects identical faults into both Baseline and Advanced
    pipelines and prints a side-by-side F1 comparison table.
    Same random seed ensures identical fault positions.
    """
    import random
    from backtest    import Backtester
    from ml_advanced import AdvancedMLPipeline

    print("\n" + "═" * 68)
    print("  NeuralFarm  ·  ML COMPARISON  (Baseline vs Advanced)")
    print("═" * 68)
    ticks = args.ticks
    seed  = 42

    # ── 1. Baseline backtest ───────────────────────────────────────────
    print(f"\n[CMP] Running BASELINE backtest ({ticks} ticks, seed={seed})...")
    bt            = Backtester(ticks=ticks, seed=seed)
    bt.run()
    baseline_m    = bt.metrics
    fault_windows = bt.fault_windows
    ground_truth  = bt.ground_truth

    # ── 2. Advanced backtest (same faults) ────────────────────────────
    print(f"[CMP] Running ADVANCED  backtest ({ticks} ticks, seed={seed})...")
    random.seed(seed)
    network  = SensorNetwork()
    pipeline = AdvancedMLPipeline(window=20)

    for tick in range(1, ticks + 1):
        readings = network.read_all(tick)
        for start, end, sid, ftype in fault_windows:
            if start <= tick <= end and sid in readings:
                sensor = network.sensors[sid]
                r      = readings[sid]
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
                readings[sid].is_injected_anomaly = True
        pipeline.process(readings, tick)

    cmp        = pipeline.compare_with_baseline(ground_truth)
    advanced_m = cmp["advanced"]

    # ── 3. Print comparison table ──────────────────────────────────────
    ANSI = {
        "reset": "\033[0m", "bold":   "\033[1m",
        "green": "\033[92m","yellow": "\033[93m",
        "red":   "\033[91m","cyan":   "\033[96m",
    }
    def C(col, t): return f"{ANSI[col]}{t}{ANSI['reset']}"

    print()
    print(C("bold", "═" * 68))
    print(C("bold", "  F1 Score Comparison  (seed=42) — Baseline vs Advanced"))
    print(C("bold", "═" * 68))
    hdr = f"  {'SENSOR':<15} {'BASE_F1':>8} {'ADV_F1':>8} {'DELTA':>8} {'WINNER':>10}"
    print(C("bold", hdr))
    print("  " + "─" * 62)

    b_f1s, a_f1s = [], []
    for sid in baseline_m:
        bm  = baseline_m[sid]
        am  = advanced_m.get(sid, {})
        bf  = bm["f1"]
        af  = am.get("f1", 0.0)
        delta  = af - bf
        winner = (C("green",  "ADVANCED") if delta >  0.01 else
                  C("yellow", "BASELINE") if delta < -0.01 else
                  C("cyan",   "TIE     "))
        af_col = "green" if af >= 0.7 else ("yellow" if af >= 0.4 else "red")
        bf_col = "green" if bf >= 0.7 else ("yellow" if bf >= 0.4 else "red")
        print(f"  {sid:<15} "
              f"{C(bf_col, f'{bf:.3f}'):>8} "
              f"{C(af_col, f'{af:.3f}'):>8} "
              f"{delta:>+8.3f} "
              f"{winner}")
        b_f1s.append(bf)
        a_f1s.append(af)

    avg_b = sum(b_f1s) / len(b_f1s) if b_f1s else 0
    avg_a = sum(a_f1s) / len(a_f1s) if a_f1s else 0
    print("  " + "─" * 62)
    print(f"  {'Macro avg':<15} {avg_b:>8.3f} {avg_a:>8.3f} {avg_a - avg_b:>+8.3f}")
    print(C("bold", "═" * 68))
    print()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args  = parse_args()
    modes = {
        "demo":     run_demo,
        "live":     run_live,
        "backtest": run_backtest,
        "export":   run_export,
        "advanced": run_advanced,
        "compare":  run_compare,
    }
    modes[args.mode](args)


if __name__ == "__main__":
    main()
