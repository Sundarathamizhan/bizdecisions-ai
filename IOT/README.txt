
NeuralFarm — IoT + AI + ML Smart Greenhouse System
===================================================

A full-stack intelligent agriculture platform that fuses three layers of
technology into a working, novelty-grade smart greenhouse monitoring system.

PROJECT OVERVIEW
────────────────
NeuralFarm simulates a real greenhouse where 6 IoT sensors stream data
continuously. A machine learning pipeline processes every reading in real
time — detecting anomalies, forecasting future values, and computing a
composite ecosystem health score. When needed, Claude AI synthesizes all
of this into natural-language crop diagnostics that a farmer can act on.

NOVELTY
───────
Most IoT + AI demos are either:
  (a) a dashboard that just displays raw values, or
  (b) an AI chatbot that answers questions about agriculture.

NeuralFarm is different: the ML layer produces structured features
(z-scores, IF scores, slopes, forecasts) that become the AI's input.
The AI is NOT looking at raw sensor values — it reasons at the agronomic
level using pre-computed statistics. This bidirectional ML→AI pipeline
is the core novel design.


ARCHITECTURE
════════════

  ┌─────────────────────────────────────────────────────────┐
  │                     IoT LAYER                           │
  │  6 Sensors: temp, humidity, soil, CO₂, light, pH        │
  │  Physics: sinusoidal cycles + Gaussian noise + spikes   │
  │  Cross-sensor correlations (temp↑ → humidity↓)          │
  └──────────────────────┬──────────────────────────────────┘
                         │ SensorReading objects
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │                      ML LAYER                           │
  │                                                         │
  │  ① Z-Score Anomaly Detection                           │
  │     Rolling window (n=20), Welford's online algorithm   │
  │     Flag when |z| = |(x−μ)/σ| > 2.5                   │
  │                                                         │
  │  ② Isolation Forest (Pure Python)                       │
  │     50 random isolation trees                           │
  │     Score = 2^(−avg_path / E(n))                       │
  │     Anomaly when score > 0.60                           │
  │                                                         │
  │  ③ Linear Regression Forecasting                        │
  │     OLS on rolling window                               │
  │     Outputs: slope, trend, 5-step forecast              │
  │                                                         │
  │  ④ Health Score                                         │
  │     Score = 100 − Σ(anomaly_rate_i × 25) per sensor    │
  └──────────────────────┬──────────────────────────────────┘
                         │ MLResult objects (features)
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │                      AI LAYER                           │
  │  Claude API (claude-sonnet-4-20250514)                  │
  │  Input: structured ML telemetry (NOT raw values)        │
  │  Output: agronomic diagnosis + 3 recommendations        │
  │  Modes: full diagnostic / quick alert / trend analysis  │
  └─────────────────────────────────────────────────────────┘


FILE STRUCTURE
══════════════

  neuralfarm/
  ├── main.py          Entry point, CLI argument parsing, run modes
  ├── iot_layer.py     SensorConfig, Sensor, SensorNetwork classes
  ├── ml_layer.py      RollingWindow, IsolationForest, LinearRegression,
  │                    MLPipeline, MLResult dataclass
  ├── ai_layer.py      AIAdvisor, Claude API client, prompt engineering
  ├── dashboard.py     Terminal dashboard + Backtester + DataExporter
  └── requirements.txt Pure Python (no pip installs needed for core)


SENSORS EXPLAINED
═════════════════

  Sensor         Range         Optimal        Correlations
  ─────────────  ────────────  ─────────────  ─────────────────────
  Temperature    5–50 °C       18–28 °C       Drives humidity offset
  Humidity       10–99 %       50–80 %        Inversely ↓ with temp
  Soil Moisture  5–95 %        40–70 %        Independent
  CO₂ Level      250–900 ppm   400–1200 ppm   Diurnal cycle
  Light          100–10000 lx  2000–6000 lux  Strong sine (day/night)
  Soil pH        3.5–9.0       6.0–7.0        Slow drift

  Physics model per tick:
    value = base
          + A·sin(2π·tick / period)     ← day/night cycle
          + Gaussian(0, σ·0.4)          ← sensor noise
          + optional_spike              ← random anomaly event
    clamped to [min, max]


ML ALGORITHMS IN DEPTH
═══════════════════════

① Z-SCORE ANOMALY DETECTION
   ─────────────────────────
   Uses Welford's online algorithm for O(1) rolling mean and variance.
   This avoids recomputing the sum over the entire window each tick.

   Welford's update (numerically stable):
     n    ← n + 1
     δ    ← x − mean
     mean ← mean + δ/n
     δ2   ← x − mean
     M2   ← M2 + δ·δ2
     σ²   ← M2 / n

   Anomaly: |z| > 2.5  (flags ≈1% of normal readings as false positives)

② ISOLATION FOREST
   ─────────────────
   Algorithm (per tree):
     1. Draw random split in [min(samples), max(samples)]
     2. Recurse into the partition containing the target value
     3. Stop at max_depth=12 or n≤1; return path length
   
   Anomaly score: s(x,n) = 2^(−E[h(x)] / c(n))
   where c(n) = 2·H(n−1) − 2(n−1)/n  (expected path length in random BST)
   
   Anomalies are isolated faster (shorter paths) → higher score.
   Threshold: 0.60 (top ~10% most isolated points)

③ LINEAR REGRESSION FORECASTING
   ───────────────────────────────
   OLS slope:     b = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²
   OLS intercept: a = ȳ − b·x̄
   Predict:       ŷ(t+k) = a + b·(n−1+k)
   
   R² is computed for regression quality.
   Trend: slope > 0.05 → rising, < −0.05 → falling, else stable.

④ HEALTH SCORE
   ────────────
   Per sensor: anomaly_rate = (anomalies in last 30) / 30
   Score = max(0, 100 − Σ anomaly_rate_i × 25)
   
   Interpretation:
     90–100 : Excellent — all sensors nominal
     70–89  : Good — minor fluctuations
     50–69  : Warning — investigate 1–2 sensors
     < 50   : Critical — immediate action needed


RUNNING THE PROJECT
═══════════════════

  # Demo mode (60 ticks, prints full report)
  python main.py --mode demo

  # Live monitoring (continuous, Ctrl+C to stop)
  python main.py --mode live --interval 0.5

  # Backtest ML detection accuracy
  python main.py --mode backtest --ticks 200

  # Export CSV + JSON
  python main.py --mode export --ticks 100 --out my_farm

  # With Claude AI diagnostic
  export ANTHROPIC_API_KEY='sk-ant-...'
  python main.py --mode demo --ai


AI PROMPT DESIGN
════════════════

  The prompt engineering follows the Chain-of-Thought + Constrained Output pattern:

  System: "You are Dr. Greenhouse, expert agronomist..."
          + structured output format (STATUS / KEY FINDINGS / RECOMMENDATIONS / RISK)
  
  User:   structured telemetry table with:
          - current value
          - rolling mean + std dev
          - z-score (already computed by ML)
          - isolation forest score
          - trend + slope
          - 5-step ML forecast
          - recent anomaly count
          - optimal agronomic range (from lookup table)

  Why structured input?
  The AI should not re-implement ML — it should interpret ML results.
  Giving it pre-computed z-scores and forecasts means it can focus on
  agronomic reasoning rather than signal processing.


BACKTEST METHODOLOGY
════════════════════

  The Backtester injects 4 random fault windows of 3 known types:
    • Spike : single tick at 95% of max value
    • Step  : sustained +4σ offset for the window duration
    • Drift : linearly growing offset 0→5σ across the window

  Ground truth labels are stored per tick per sensor.
  After simulation, evaluation metrics are computed:

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2·P·R / (P+R)

  Expected results (seed=42, 200 ticks):
    Spike faults: high precision, moderate recall (fast spikes sometimes missed)
    Step faults:  high recall, moderate precision (sustained offset well detected)
    Drift faults: variable F1 (detection latency depends on window fill)


EXTENSION IDEAS
═══════════════

  Real hardware integration:
    Replace iot_layer.py with MQTT client:
      import paho.mqtt.client as mqtt
      client.subscribe("greenhouse/sensors/#")

  Database persistence:
    Add InfluxDB / TimescaleDB writer to data_export.py

  Web dashboard:
    Wrap with FastAPI + WebSocket for real-time browser updates

  Advanced ML:
    - LSTM for multivariate forecasting
    - XGBoost with engineered features for classification
    - DBSCAN for spatial clustering of anomaly zones

  Alert integration:
    Add Twilio SMS / email notification when health score < 60
