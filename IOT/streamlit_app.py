"""
streamlit_app.py — NeuralFarm Web Dashboard
============================================
Real-time greenhouse monitoring dashboard.

Run:
    streamlit run c:/new/IOT/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import math
import streamlit as st
import plotly.graph_objects as go

from iot_layer   import SensorNetwork
from ml_layer    import MLPipeline
from ai_layer    import AIAdvisor, AutoDecisionEngine
from database    import NeuralFarmDB
from digital_twin import DigitalTwin
from telegram_alert import TelegramAlerter
from ml_advanced import AdvancedMLPipeline
from backtest import Backtester

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌱 NeuralFarm Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp {
    background: linear-gradient(135deg,#071220 0%,#0d2137 60%,#071a2e 100%);
  }
  section[data-testid="stSidebar"] {
    background: rgba(7,18,32,0.97);
    border-right: 1px solid #1a3a5c;
  }
  .metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,212,255,0.18);
    border-radius: 10px; padding:12px 16px; text-align:center;
  }
  .alert-card {
    padding:8px 12px; border-radius:6px; margin:4px 0; font-size:.82em;
  }
  .alert-critical { background:rgba(239,68,68,.14); border-left:3px solid #ef4444; }
  .alert-warning  { background:rgba(245,158,11,.14); border-left:3px solid #f59e0b; }
  .action-card { padding:7px 11px; border-radius:6px; margin:3px 0; font-size:.80em; }
  .action-HIGH   { background:rgba(239,68,68,.12);  border-left:3px solid #ef4444; }
  .action-MEDIUM { background:rgba(245,158,11,.12); border-left:3px solid #f59e0b; }
  .action-LOW    { background:rgba(34,197,94,.10);  border-left:3px solid #22c55e; }
  h1,h2,h3 { color:#00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
SENSORS = ["temp", "humidity", "soil", "co2", "light", "ph"]
SNAMES  = {
    "temp":     "Temperature (°C)",
    "humidity": "Humidity (%)",
    "soil":     "Soil Moisture (%)",
    "co2":      "CO₂ (ppm)",
    "light":    "Light (lux)",
    "ph":       "Soil pH",
}
COLORS  = {
    "temp":"#ff6b6b","humidity":"#4ecdc4","soil":"#95e17d",
    "co2":"#ffd93d","light":"#a8edea","ph":"#c77dff",
}
BG = "rgba(0,0,0,0)"

# ── Session state ─────────────────────────────────────────────────────────
def _init():
    if "ready" in st.session_state:
        return
    st.session_state.ready    = True
    st.session_state.network  = SensorNetwork()
    st.session_state.ml       = AdvancedMLPipeline(window=20)
    st.session_state.advisor  = AIAdvisor()
    st.session_state.engine   = AutoDecisionEngine()
    st.session_state.db       = NeuralFarmDB()
    st.session_state.twin     = DigitalTwin("tomato")
    st.session_state.alerter  = TelegramAlerter()
    st.session_state.tick     = 0
    st.session_state.twin_metrics = {}
    st.session_state.hist     = {
        s: {"ticks":[],"values":[],"at":[],"av":[]} for s in SENSORS
    }
    st.session_state.h_hist   = []   # health score history
    st.session_state.alerts   = []   # alert log
    st.session_state.actions  = []   # auto-decision log
    st.session_state.ai_log   = []   # AI report log
    st.session_state.running  = False

_init()

# ── Simulate one tick ─────────────────────────────────────────────────────
def simulate_tick():
    ss = st.session_state
    ss.tick += 1
    readings = ss.network.read_all(ss.tick)
    results  = ss.ml.process(readings, ss.tick)
    score    = ss.ml.compute_health_score()
    
    # Update DB and Twin
    ss.db.insert_telemetry(ss.tick, results)
    ss.twin_metrics = ss.twin.update(readings)
    
    ss.h_hist.append(score)

    MAX = 120
    for sid, res in results.items():
        # Handle both MLPipeline (MLResult) and AdvancedMLPipeline (dict)
        r = res["base"] if isinstance(res, dict) else res
        
        h = ss.hist[sid]
        h["ticks"].append(ss.tick);  h["values"].append(r.value)
        if len(h["ticks"]) > MAX:    h["ticks"]  = h["ticks"][-MAX:]
        if len(h["values"]) > MAX:   h["values"] = h["values"][-MAX:]
        if r.anomaly:
            h["at"].append(ss.tick); h["av"].append(r.value)
            if len(h["at"]) > MAX:   h["at"] = h["at"][-MAX//2:]
            if len(h["av"]) > MAX:   h["av"] = h["av"][-MAX//2:]
            ss.alerts.insert(0, {
                "tick": ss.tick, "sensor": sid,
                "value": r.value, "unit": r.unit,
                "severity": r.severity,
                "z": round(r.z_score, 2) if r.z_score else "?",
                "if": round(r.if_score, 3) if r.if_score else "?",
                "t": time.strftime("%H:%M:%S"),
            })
            # Trigger real-time Telegram alert for critical 
            if r.severity == "critical" and ss.alerter.enabled:
                ss.alerter.send_alert(r.severity, sid, r.value, r.unit, action="CHECK_DASHBOARD")

    if len(ss.alerts) > 60: ss.alerts = ss.alerts[:60]
    if len(ss.h_hist)  > MAX: ss.h_hist = ss.h_hist[-MAX:]

    summary = ss.ml.get_summary()
    fired   = ss.engine.evaluate(summary)
    for sid2, acts in fired.items():
        for a in acts:
            ss.actions.insert(0, {"tick": ss.tick, "sensor": sid2,
                                  "t": time.strftime("%H:%M:%S"), **a})
            # Trigger real-time Telegram alert for auto-decision
            if ss.alerter.enabled:
                ss.alerter.send_decision(
                    sensor_id=sid2, 
                    action=a["action"], 
                    priority=a.get("priority", "LOW"), 
                    value=a["value"], 
                    trigger=a.get("trigger", "rule-based")
                )
                ss.db.insert_event(ss.tick, "ACTION", {"action": a["action"], "sensor": sid2, "value": a["value"]})

    if len(ss.actions) > 30: ss.actions = ss.actions[:30]

    # Periodic Health Report
    if ss.tick % 20 == 0 and ss.alerter.enabled:
        active_anom = sum(1 for r in results.values() if getattr(r, "anomaly", False))
        ss.alerter.send_health_report(score, active_anom, ss.actions[:5])

# ── Charts ────────────────────────────────────────────────────────────────
def gauge(score):
    color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text":"Health Score","font":{"size":15,"color":"#94a3b8"}},
        gauge={
            "axis":  {"range":[0,100],"tickcolor":"#555"},
            "bar":   {"color":color,"thickness":0.22},
            "bgcolor":"rgba(255,255,255,0.03)",
            "steps": [{"range":[0,40],"color":"rgba(239,68,68,.12)"},
                      {"range":[40,70],"color":"rgba(245,158,11,.12)"},
                      {"range":[70,100],"color":"rgba(34,197,94,.12)"}],
        },
        number={"font":{"size":52,"color":color}},
    ))
    fig.update_layout(height=220,margin=dict(l=20,r=20,t=20,b=10),
                      paper_bgcolor=BG,font_color="#fff")
    return fig

def health_line(h_hist):
    ticks = list(range(len(h_hist)))
    cols  = ["#22c55e" if h>=70 else "#f59e0b" if h>=40 else "#ef4444" for h in h_hist]
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=ticks, y=h_hist, mode="lines",
        line=dict(color="#00d4ff",width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)"))
    fig.add_trace(go.Scatter(x=ticks, y=h_hist, mode="markers",
        marker=dict(color=cols, size=4), showlegend=False))
    fig.update_layout(height=150,margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor=BG, plot_bgcolor="rgba(255,255,255,0.02)",
        xaxis=dict(showgrid=False,color="#555"),
        yaxis=dict(range=[0,105],showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#555"),
        showlegend=False)
    return fig

def sensor_chart(sid, h):
    color = COLORS.get(sid,"#00d4ff")
    fig   = go.Figure()
    if h["ticks"]:
        fig.add_trace(go.Scatter(x=h["ticks"],y=h["values"],mode="lines",
            line=dict(color=color,width=1.7),
            hovertemplate=f"%{{y:.2f}}<extra>{SNAMES[sid]}</extra>"))
    if h["at"]:
        fig.add_trace(go.Scatter(x=h["at"],y=h["av"],mode="markers",
            marker=dict(color="#ff3333",size=8,symbol="x",
                        line=dict(width=2,color="#ff3333")),
            name="Anomaly", hovertemplate="%{y:.2f}<extra>Anomaly</extra>"))
    fig.update_layout(
        title=dict(text=SNAMES[sid],font=dict(color=color,size=12)),
        height=175,margin=dict(l=0,r=0,t=32,b=0),
        paper_bgcolor=BG, plot_bgcolor="rgba(255,255,255,0.02)",
        xaxis=dict(showgrid=False,color="#555"),
        yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)",color="#555"),
        showlegend=False)
    return fig

def heatmap_chart(hist):
    ids = SENSORS
    n   = len(ids)
    corr = [[1.0]*n for _ in range(n)]
    for i,s1 in enumerate(ids):
        for j,s2 in enumerate(ids):
            if i == j: continue
            xs = hist[s1]["values"][-60:]
            ys = hist[s2]["values"][-60:]
            m  = min(len(xs), len(ys))
            if m < 5: corr[i][j] = 0.0; continue
            mx,my = sum(xs)/m, sum(ys)/m
            num  = sum((xs[k]-mx)*(ys[k]-my) for k in range(m))
            dx   = math.sqrt(sum((x-mx)**2 for x in xs) or 1e-9)
            dy   = math.sqrt(sum((y-my)**2 for y in ys) or 1e-9)
            corr[i][j] = round(num/(dx*dy), 2)
    labels = [s[:4].upper() for s in ids]
    fig = go.Figure(go.Heatmap(z=corr, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr, texttemplate="%{text:.2f}",
        colorbar=dict(tickfont=dict(color="#94a3b8"),len=0.8)))
    fig.update_layout(
        title=dict(text="Sensor Correlation Heatmap",
                   font=dict(color="#94a3b8",size=13)),
        height=310, margin=dict(l=0,r=0,t=40,b=0),
        paper_bgcolor=BG,
        xaxis=dict(tickfont=dict(color="#94a3b8")),
        yaxis=dict(tickfont=dict(color="#94a3b8")))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌱 NeuralFarm Control")
    st.divider()

    ca, cb = st.columns(2)
    with ca:
        if st.button("▶ Step", use_container_width=True):
            simulate_tick()
    with cb:
        if st.button("🔄 Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    running = st.toggle("⚡ Auto Run", value=st.session_state.running)
    st.session_state.running = running
    speed   = st.slider("Tick interval (ms)", 100, 3000, 600, 100)

    st.divider()
    st.markdown("### 🤖 Claude AI")
    use_ai = st.checkbox("Enable AI Advisory")
    if use_ai and st.button("🔍 Get Diagnosis", use_container_width=True):
        with st.spinner("Consulting Claude..."):
            summ  = st.session_state.ml.get_summary()
            sc    = st.session_state.ml.compute_health_score()
            rep   = st.session_state.advisor.diagnose(summ, sc)
            st.session_state.ai_log.insert(0, {
                "tick":       st.session_state.tick,
                "t":          time.strftime("%H:%M:%S"),
                "confidence": rep.confidence,
                "text":       rep.raw_insight,
                "actions":    rep.actions,
            })

    st.divider()
    st.markdown("### 📊 ML Model Comparison")
    if st.button("Run Evaluation", use_container_width=True):
        with st.spinner("Running 200-tick offline backtest..."):
            bt = Backtester(ticks=200)
            bt.pipeline = AdvancedMLPipeline(window=20)
            bt.run()
            res = bt.pipeline.compare_with_baseline(bt.ground_truth)
            b_f1 = sum(m["f1"] for m in res["baseline"].values()) / max(len(res["baseline"]), 1)
            a_f1 = sum(m["f1"] for m in res["advanced"].values()) / max(len(res["advanced"]), 1)
            st.session_state.eval_res = (b_f1, a_f1)
            st.rerun()
    
    if "eval_res" in st.session_state:
        b_f1, a_f1 = st.session_state.eval_res
        imp = ((a_f1 - b_f1) / max(b_f1, 0.001)) * 100
        st.markdown(
            f"<div class='metric-box'>"
            f"<b>Baseline F1:</b> {b_f1:.2f}<br>"
            f"<b>Advanced F1:</b> {a_f1:.2f}<br>"
            f"<span style='color:#22c55e'><b>Improvement: +{imp:.1f}%</b></span>"
            f"</div>", unsafe_allow_html=True
        )

    st.divider()
    st.markdown("### 🚨 Live Alerts")
    shown = st.session_state.alerts[:10]
    if not shown:
        st.caption("No alerts yet — run ticks to start")
    for al in shown:
        cls = "alert-critical" if al["severity"] == "critical" else "alert-warning"
        st.markdown(
            f"<div class='alert-card {cls}'>"
            f"<b>{al['t']}</b> T{al['tick']}<br>"
            f"<b>{al['sensor'].upper()}</b>: {al['value']}{al['unit']} "
            f"Z={al['z']}σ [{al['severity'].upper()}]"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Main area ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;margin-bottom:2px'>🌱 NeuralFarm Smart Greenhouse</h1>"
    "<p style='text-align:center;color:#64748b;margin-top:0'>Real-Time IoT · ML · AI Dashboard</p>",
    unsafe_allow_html=True,
)

ss    = st.session_state
score = ss.h_hist[-1] if ss.h_hist else 100
prev  = ss.h_hist[-2] if len(ss.h_hist) > 1 else score
last_log = ss.ml.baseline.results_log if hasattr(ss.ml, "baseline") else getattr(ss.ml, "results_log", [])
last  = last_log[-1] if last_log else {}
anom  = sum(1 for r in last.values() if r.anomaly)

# KPI row
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("⏱ Tick",            ss.tick)
k2.metric("❤ Health",          score,  delta=int(score-prev))
k3.metric("🚨 Alerts",         len(ss.alerts))
k4.metric("⚡ Actions",        len(ss.actions))
k5.metric("🔴 Anomalous",      anom)

if getattr(ss, "twin_metrics", {}):
    tw1, tw2 = st.columns(2)
    tw1.metric("🌱 Yield %", f"{ss.twin_metrics.get('yield_pct', 0):.1f}%")
    tw2.metric("📅 Days to Harvest", ss.twin_metrics.get("days_to_harvest", 0))
    if ss.twin_metrics.get("stresses"):
        st.warning("⚠️ " + " | ".join(ss.twin_metrics["stresses"]))

st.divider()

# Gauge + health history
g_col, h_col = st.columns([1, 2])
with g_col:
    st.plotly_chart(gauge(score), use_container_width=True,
                    config={"displayModeBar": False})
with h_col:
    if ss.h_hist:
        st.markdown("#### Health Score History")
        st.plotly_chart(health_line(ss.h_hist), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("Click ▶ Step or enable Auto Run to start streaming")

st.divider()

# 6 sensor charts (2 rows × 3 cols)
st.markdown("### 📈 Sensor Time Series  *(red × = anomaly)*")
r1 = st.columns(3);  r2 = st.columns(3)
grid = r1 + r2
for i, sid in enumerate(SENSORS):
    with grid[i]:
        h = ss.hist[sid]
        if h["ticks"]:
            st.plotly_chart(sensor_chart(sid, h), use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.markdown(
                f"<div style='height:175px;display:flex;align-items:center;"
                f"justify-content:center;color:#334155;border:1px solid #1e3a5f;"
                f"border-radius:8px'>{SNAMES[sid]}</div>",
                unsafe_allow_html=True,
            )

st.divider()

# Bottom row: heatmap | auto-decisions | AI log
hc, ac, aic = st.columns([1.5, 1, 1.5])

with hc:
    st.markdown("### 🔥 Correlation Heatmap")
    if any(ss.hist[s]["ticks"] for s in SENSORS):
        st.plotly_chart(heatmap_chart(ss.hist), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("Needs ≥ 5 ticks")

with ac:
    st.markdown("### ⚙ Auto-Decisions")
    if ss.actions:
        for a in ss.actions[:10]:
            pri = a.get("priority", "LOW")
            st.markdown(
                f"<div class='action-card action-{pri}'>"
                f"<b>{a['t']} · T{a['tick']} · {a['sensor'].upper()}</b><br>"
                f"{a.get('action','—')}<br>"
                f"<small>val={a.get('value','?')} ({a.get('trigger','?')})</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No actions triggered yet")

with aic:
    st.markdown("### 🤖 AI Advisory")
    if ss.ai_log:
        for entry in ss.ai_log[:3]:
            with st.expander(
                f"T{entry['tick']} · {entry['t']} · Confidence {entry['confidence']}%"
            ):
                st.markdown(f"```\n{entry['text'][:800]}\n```")
    else:
        st.caption("Enable AI Advisory in sidebar → click Get Diagnosis")

# ── Auto-refresh ──────────────────────────────────────────────────────────
if st.session_state.running:
    simulate_tick()
    time.sleep(speed / 1000)
    st.rerun()
