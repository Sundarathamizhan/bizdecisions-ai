"""
ai_layer.py — Claude AI Diagnostic Advisor  (v2 — Stateful)
════════════════════════════════════════════════════════════
Upgrades over v1:
  • AI Memory      : retains last 5 diagnoses, enabling temporal reasoning
                     ("Temperature has been rising for 3 reports")
  • Confidence Score: 0-100 computed from anomaly density + trend signals
  • AutoDecisionEngine: rule-based control actions (no API call needed)
                        e.g. temp>32°C → TURN_ON_COOLING
  • Retry + Backoff : up to 3 attempts with exponential wait

API: Anthropic Messages API  (claude-sonnet-4-20250514)
Auth: ANTHROPIC_API_KEY environment variable
"""

import os
import json
import time
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Agronomic Context
# ─────────────────────────────────────────────────────────────────────────────

AGRONOMIC_CONTEXT = {
    "temp": {
        "optimal_range": "18–28°C",
        "critical_low":  "< 10°C — chilling injury, stunted growth",
        "critical_high": "> 35°C — heat stress, wilting, flower drop",
        "crops_affected": "tomatoes, peppers, cucumbers",
    },
    "humidity": {
        "optimal_range": "50–80%",
        "critical_low":  "< 30% — drought stress, increased transpiration",
        "critical_high": "> 90% — fungal disease risk (Botrytis, powdery mildew)",
        "crops_affected": "all leafy crops",
    },
    "soil": {
        "optimal_range": "40–70% field capacity",
        "critical_low":  "< 20% — wilting, root damage",
        "critical_high": "> 85% — root hypoxia, nutrient leaching",
        "crops_affected": "all root crops",
    },
    "co2": {
        "optimal_range": "400–1200 ppm",
        "critical_low":  "< 350 ppm — photosynthesis limited",
        "critical_high": "> 2000 ppm — stomatal closure, reduced transpiration",
        "crops_affected": "all C3 plants (tomatoes, lettuce, herbs)",
    },
    "light": {
        "optimal_range": "2000–6000 lux",
        "critical_low":  "< 500 lux — etiolation, poor fruit set",
        "critical_high": "> 8000 lux — photobleaching possible",
        "crops_affected": "fruiting crops, microgreens",
    },
    "ph": {
        "optimal_range": "6.0–7.0",
        "critical_low":  "< 5.0 — Al/Mn toxicity, P deficiency",
        "critical_high": "> 7.5 — Fe/Mn/Zn deficiency, nutrient lockout",
        "crops_affected": "all crops; hydroponic systems especially sensitive",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# API Configuration — supports Claude (paid) OR Groq (FREE) OR Gemini (FREE)
# ─────────────────────────────────────────────────────────────────────────────
#
#  HOW TO USE FREE AI:
#  ───────────────────
#  Option A — Groq (RECOMMENDED, completely FREE):
#    1. Sign up at: https://console.groq.com  (free, no credit card)
#    2. Create API key
#    3. Set below: GROQ_API_KEY = "gsk_..."
#
#  Option B — Google Gemini (FREE tier — 15 req/min):
#    1. Sign up at: https://aistudio.google.com  (free)
#    2. Create API key
#    3. Set below: GEMINI_API_KEY = "AIza..."
#

#  Option C — Anthropic Claude (PAID, ~$0.003/call):
#    1. Sign up at: https://console.anthropic.com
#    2. Set below: ANTHROPIC_API_KEY = "sk-ant-..."
# ─────────────────────────────────────────────────────────────────────────────

# ✏️  PASTE YOUR FREE API KEY HERE:
GROQ_API_KEY      = ""   # ← Paste your Groq key here (free at console.groq.com)
GEMINI_API_KEY    = ""   # ← Gemini (FREE) — get at aistudio.google.com
ANTHROPIC_API_KEY = ""   # ← Claude (PAID) — get at console.anthropic.com

# API endpoints
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GROQ_API_URL      = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_API_URL    = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

DEFAULT_MODEL     = "claude-sonnet-4-20250514"
GROQ_MODEL        = "llama-3.3-70b-versatile"   # free, very capable


def call_claude(messages: list, system: str, max_tokens: int = 1024,
                model: str = DEFAULT_MODEL) -> Optional[str]:
    """
    Call the Anthropic Messages API using only stdlib (urllib).
    Includes exponential-backoff retry (up to 3 attempts).
    """
    api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "[Claude Unavailable] Set ANTHROPIC_API_KEY in ai_layer.py"

    payload = json.dumps({
        "model":      model,
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   messages,
    }).encode("utf-8")

    last_err = ""
    for attempt in range(3):
        req = urllib.request.Request(
            ANTHROPIC_API_URL,
            data    = payload,
            method  = "POST",
            headers = {
                "Content-Type":      "application/json",
                "x-api-key":         api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            last_err = f"[Claude Error {e.code}] {body[:200]}"
            if e.code in (400, 401, 403):
                return last_err
        except Exception as e:
            last_err = f"[Claude Network Error] {e}"
        if attempt < 2:
            time.sleep(2 ** attempt)
    return last_err


def call_groq(messages: list, system: str,
              max_tokens: int = 1024) -> Optional[str]:
    """FREE Groq API — llama-3.3-70b. Get key at console.groq.com"""
    api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "[Groq Unavailable] Set GROQ_API_KEY in ai_layer.py"
    msgs    = [{"role": "system", "content": system}] + messages
    payload = json.dumps({
        "model": GROQ_MODEL, "messages": msgs,
        "max_tokens": max_tokens, "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(
        GROQ_API_URL, data=payload, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 403 and "1010" in body:
            return "❌ **[Groq Error 403: Region Blocked]**\nGroq API is not available in your region. \n\n**Fixes:**\n1. Use a VPN (US/Europe)\n2. Switch to Gemini (Paste GEMINI_API_KEY on line 91)"
        return f"[Groq Error {e.code}] {body[:200]}"
    except Exception as e:
        return f"[Groq Network Error] {e}"


def call_gemini(messages: list, system: str,
                max_tokens: int = 1024) -> Optional[str]:
    """FREE Gemini API — 15 req/min. Get key at aistudio.google.com"""
    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "[Gemini Unavailable] Set GEMINI_API_KEY in ai_layer.py"
    user_text = system + "\n\n" + "\n".join(m["content"] for m in messages)
    payload   = json.dumps({
        "contents": [{"parts": [{"text": user_text}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3},
    }).encode("utf-8")
    url = f"{GEMINI_API_URL}?key={api_key}"
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        return f"[Gemini Error {e.code}] {e.read().decode()[:200]}"
    except Exception as e:
        return f"[Gemini Network Error] {e}"


def call_ai(messages: list, system: str, max_tokens: int = 800) -> Optional[str]:
    """
    Smart dispatcher — auto-picks whichever API key is set in ai_layer.py.
    Priority: Groq (FREE) → Gemini (FREE) → Claude (PAID)
    """
    if GROQ_API_KEY      or os.environ.get("GROQ_API_KEY"):
        return call_groq(messages, system, max_tokens)
    if GEMINI_API_KEY    or os.environ.get("GEMINI_API_KEY"):
        return call_gemini(messages, system, max_tokens)
    if ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY"):
        return call_claude(messages, system, max_tokens)
    return (
        "[No AI Key configured]\n"
        "Paste any ONE key in ai_layer.py (top of file):\n"
        "  GROQ_API_KEY   = 'gsk_...'   ← FREE at console.groq.com\n"
        "  GEMINI_API_KEY = 'AIza...'   ← FREE at aistudio.google.com"
    )



# ─────────────────────────────────────────────────────────────────────────────
# Auto Decision Engine
# ─────────────────────────────────────────────────────────────────────────────

# Rule table: sensor_id → list of {condition, threshold, action, priority}
_CONTROL_RULES: Dict[str, list] = {
    "temp":     [
        {"condition": "high", "threshold": 32.0,   "action": "TURN_ON_COOLING",      "priority": "HIGH"},
        {"condition": "low",  "threshold": 15.0,   "action": "TURN_ON_HEATING",      "priority": "HIGH"},
    ],
    "humidity": [
        {"condition": "high", "threshold": 88.0,   "action": "ACTIVATE_VENTILATION", "priority": "MEDIUM"},
        {"condition": "low",  "threshold": 35.0,   "action": "ACTIVATE_MISTING",     "priority": "MEDIUM"},
    ],
    "soil":     [
        {"condition": "low",  "threshold": 25.0,   "action": "START_IRRIGATION",     "priority": "HIGH"},
        {"condition": "high", "threshold": 82.0,   "action": "REDUCE_IRRIGATION",    "priority": "LOW"},
    ],
    "co2":      [
        {"condition": "high", "threshold": 1500.0, "action": "INCREASE_VENTILATION", "priority": "MEDIUM"},
        {"condition": "low",  "threshold": 380.0,  "action": "CLOSE_VENTS",          "priority": "LOW"},
    ],
    "light":    [
        {"condition": "high", "threshold": 8500.0, "action": "DEPLOY_SHADE_NET",     "priority": "LOW"},
        {"condition": "low",  "threshold": 800.0,  "action": "ACTIVATE_GROW_LIGHTS", "priority": "MEDIUM"},
    ],
    "ph":       [
        {"condition": "high", "threshold": 7.3,    "action": "INJECT_PH_DOWN",       "priority": "HIGH"},
        {"condition": "low",  "threshold": 5.5,    "action": "INJECT_PH_UP",         "priority": "HIGH"},
    ],
}


class AutoDecisionEngine:
    """
    Rule-based control system that fires automated actions WITHOUT an API call.
    Acts as the first-responder layer beneath the LLM advisory.

    Usage
    -----
        engine  = AutoDecisionEngine()
        actions = engine.evaluate(summary)   # Dict[str, list[dict]]
    """

    def evaluate(self, summary: Dict) -> Dict[str, List[dict]]:
        """
        Check each sensor's current_value against the rule table.
        Returns dict of sensor_id → list of triggered actions.
        """
        triggered: Dict[str, List[dict]] = {}
        for sid, data in summary.items():
            rules    = _CONTROL_RULES.get(sid, [])
            val      = data.get("current_value", 0)
            fired    = []
            for rule in rules:
                if rule["condition"] == "high" and val > rule["threshold"]:
                    fired.append({"action": rule["action"],
                                  "priority": rule["priority"],
                                  "value":    val,
                                  "trigger":  f"> {rule['threshold']}"})
                elif rule["condition"] == "low" and val < rule["threshold"]:
                    fired.append({"action": rule["action"],
                                  "priority": rule["priority"],
                                  "value":    val,
                                  "trigger":  f"< {rule['threshold']}"})
            if fired:
                triggered[sid] = fired
        return triggered

    def format_actions(self, actions: Dict[str, List[dict]]) -> str:
        if not actions:
            return "No automated actions triggered."
        lines = []
        for sid, acts in actions.items():
            for a in acts:
                pri_sym = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                    a["priority"], "⚪")
                lines.append(
                    f"  {pri_sym} [{a['priority']}] {sid.upper():10s} "
                    f"value={a['value']} ({a['trigger']}) → {a['action']}"
                )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# AI Advisor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagnosticReport:
    timestamp:     str
    health_score:  int
    confidence:    int       # 0-100 computed confidence score
    raw_insight:   str
    sensor_flags:  Dict      # sensors with anomalies
    top_risk:      str       # extracted from response
    actions:       List[dict] = field(default_factory=list)  # auto-decisions


class AIAdvisor:
    """
    Wraps the Claude API and manages the prompt engineering for
    precision agriculture diagnostics.

    Prompt design principles
    ────────────────────────
      • System prompt: agronomist persona with domain tables
      • User prompt: structured ML telemetry in a consistent format
      • Constrained output: numbered sections for easy parsing
      • Concrete values: AI must cite actual sensor readings
    """

    SYSTEM_PROMPT = """You are Dr. Greenhouse, an expert precision agriculture AI with 20 years of
controlled environment agriculture experience. You analyze IoT sensor telemetry 
pre-processed by ML algorithms (Z-score anomaly detection, Isolation Forest, 
linear regression forecasting).

Your diagnostic reports follow this EXACT structure:
── STATUS ──────────────────────────────────────────────
[One sentence: overall system status with health score interpretation]

── KEY FINDINGS ────────────────────────────────────────
[3 bullet points citing specific sensor values, ML anomaly counts, and trends]

── RECOMMENDATIONS ─────────────────────────────────────
1. [Immediate action within next hour]
2. [Short-term adjustment within 24 hours]
3. [Preventive measure for next 7 days]

── RISK WATCH ──────────────────────────────────────────
[One paragraph: the single highest risk and what to monitor]

Be specific with numbers. Reference ML forecasts when relevant.
Keep total response under 250 words."""

    def __init__(self):
        self._call_count   = 0
        self._last_call    = 0.0
        self._min_interval = 10.0          # minimum seconds between API calls
        self.history: deque = deque(maxlen=5)   # last 5 STATUS lines
        self.decisions = AutoDecisionEngine()

    # ── Confidence Score ──────────────────────────────────────────────────

    def compute_confidence(self, summary: Dict, health_score: int) -> int:
        """
        Compute a 0-100 confidence score based on:
          • health_score        (base)
          • number of critical alerts (−15 each)
          • anomalous sensors trending away from normal (−8 each)
          • total recent alerts density (−up to 20)
          • stable sensors (small positive bonus)
        """
        conf = health_score
        n_crit    = sum(1 for s in summary.values() if s.get("severity") == "critical")
        n_trend   = sum(1 for s in summary.values()
                        if s.get("trend") in ("rising","falling") and s.get("anomaly"))
        n_stable  = sum(1 for s in summary.values() if not s.get("anomaly"))
        total_alerts = sum(s.get("recent_anomalies", 0) for s in summary.values())

        conf -= n_crit   * 15
        conf -= n_trend  * 8
        conf -= min(20, total_alerts // 3)
        conf += n_stable * 2
        return max(0, min(100, conf))

    def _format_telemetry(self, summary: Dict, health_score: int,
                          confidence: int) -> str:
        """
        Formats ML summary + AI memory into structured prompt.
        History gives the LLM temporal context across calls.
        """
        lines = [
            f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"ML HEALTH SCORE: {health_score}/100",
            f"CONFIDENCE: {confidence}/100",
        ]

        # ── AI Memory: last N status lines ──────────────────────────────
        if self.history:
            lines += ["", "SYSTEM MEMORY (previous diagnoses, oldest→newest):",
                      "─" * 50]
            for i, h in enumerate(self.history, 1):
                lines.append(f"  [{i}] {h}")

        lines += ["", "SENSOR TELEMETRY (ML-processed):", "─" * 50]

        for sid, data in summary.items():
            ctx = AGRONOMIC_CONTEXT.get(sid, {})
            anomaly_str = (
                f"ANOMALOUS [Z={data.get('z_score','?')}, "
                f"IF={data.get('if_score','?'):.2f}, "
                f"severity={data.get('severity','?')}]"
                if data.get("anomaly") else "normal"
            )
            lines += [
                f"\n{sid.upper()} — {data['name'].upper() if 'name' in data else sid}",
                f"  Current:      {data['current_value']} {data['unit']}",
                f"  Rolling mean: {data.get('mean', '—')} {data['unit']}",
                f"  Std dev:      {data.get('std', '—')}",
                f"  Trend:        {data.get('trend', '?')} "
                f"(slope={data.get('slope', '?')})",
                f"  ML forecast:  {data.get('forecast_5', '?')} {data['unit']} (5 steps ahead)",
                f"  Status:       {anomaly_str}",
                f"  Recent alerts:{data.get('recent_anomalies', 0)} in last 30 samples",
                f"  Optimal range:{ctx.get('optimal_range', 'N/A')}",
            ]

        return "\n".join(lines)

    def diagnose(self, summary: Dict, health_score: int) -> DiagnosticReport:
        """
        Send ML telemetry to Claude and return a full DiagnosticReport.
        Includes: confidence score, automated actions, and AI memory.
        Applies rate limiting between calls.
        """
        # 1. Compute confidence
        confidence = self.compute_confidence(summary, health_score)

        # 2. Auto decisions (no API needed)
        actions     = self.decisions.evaluate(summary)
        actions_flat = [a for acts in actions.values() for a in acts]

        # 3. Rate limiting
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        # 4. Build prompt with memory
        telemetry = self._format_telemetry(summary, health_score, confidence)
        messages  = [{"role": "user", "content": telemetry}]

        response = call_ai(
            messages   = messages,
            system     = self.SYSTEM_PROMPT,
            max_tokens = 800,
        ) or "[No response from AI]"

        # 5. Store STATUS line in memory (temporal context for next call)
        status_line = next(
            (ln.strip() for ln in response.splitlines() if ln.strip()
             and not ln.startswith("─") and not ln.startswith("[") ), ""
        )
        self.history.append(status_line[:120])  # trim for token budget

        self._call_count += 1
        self._last_call   = time.time()

        # 6. Build report
        flags    = {sid: d for sid, d in summary.items() if d.get("anomaly")}
        top_risk = next(
            (ln.strip() for ln in response.splitlines() if "RISK" in ln.upper()), ""
        )
        return DiagnosticReport(
            timestamp    = time.strftime("%Y-%m-%d %H:%M:%S"),
            health_score = health_score,
            confidence   = confidence,
            raw_insight  = response,
            sensor_flags = flags,
            top_risk     = top_risk,
            actions      = actions_flat,
        )

    def quick_alert(self, sensor_id: str, value: float, unit: str,
                    z_score: float, trend: str) -> str:
        """
        Fast single-sensor alert explanation (shorter, faster prompt).
        Used for real-time critical alerts.
        """
        ctx = AGRONOMIC_CONTEXT.get(sensor_id, {})
        msg = (
            f"URGENT: {sensor_id} sensor reading {value}{unit} "
            f"(Z-score={z_score:.1f}σ, trend={trend}). "
            f"Optimal range: {ctx.get('optimal_range', 'unknown')}. "
            f"Critical thresholds: {ctx.get('critical_high', 'N/A')} / "
            f"{ctx.get('critical_low', 'N/A')}. "
            f"Give a 2-sentence immediate action recommendation."
        )
        messages = [{"role": "user", "content": msg}]
        quick_system = (
            "You are an expert greenhouse agronomist. "
            "Provide very brief (2 sentences max), actionable advice."
        )
        return call_claude(messages, quick_system, max_tokens=150) or "—"

    def trend_analysis(self, sensor_id: str, values: list, forecast: float, unit: str) -> str:
        """
        Ask Claude to interpret a sensor's trend and ML forecast narrative.
        """
        msg = (
            f"Sensor: {sensor_id}. Last 20 values (chronological): "
            f"{[round(v, 2) for v in values[-20:]]}. "
            f"ML 5-step linear regression forecast: {forecast}{unit}. "
            f"Optimal range: {AGRONOMIC_CONTEXT.get(sensor_id, {}).get('optimal_range', 'unknown')}. "
            f"In 3 sentences: interpret this trend, explain what it means for crop health, "
            f"and recommend one preventive action."
        )
        messages = [{"role": "user", "content": msg}]
        system = "You are a precision agriculture specialist. Be specific and cite values."
        return call_claude(messages, system, max_tokens=200) or "—"
