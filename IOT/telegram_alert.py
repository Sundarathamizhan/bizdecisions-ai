"""
telegram_alert.py — NeuralFarm Telegram Alert Bot
==================================================
HOW TO SET UP (3 steps):

  STEP 1 — Create bot & get BOT_TOKEN
    • Open Telegram → search @BotFather → send /newbot
    • Name it "NeuralFarm Alert", username: neuralfarm_alert_bot
    • BotFather gives you a token like: 123456789:AAF-abc123XYZ...

  STEP 2 — Get your CHAT_ID
    • Send /start to your new bot
    • Open in browser: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
    • Find "chat": {"id": 987654321}  ← that number is your CHAT_ID

  STEP 3 — Paste both values below ↓↓↓
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional, List

# ══════════════════════════════════════════════════════════════════
#  ✏️  PASTE YOUR CREDENTIALS HERE  (Step 3) - SECURED
# ══════════════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ══════════════════════════════════════════════════════════════════
_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlerter:
    """
    Sends structured Markdown messages to a Telegram chat.

    Rate-limits automatically to one message per 30 s to avoid
    Telegram flood restrictions. Falls back silently when unconfigured.
    """

    def __init__(self):
        # Prefer hardcoded values; fall back to environment variables
        self.token    = TELEGRAM_BOT_TOKEN or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id  = TELEGRAM_CHAT_ID   or os.environ.get("TELEGRAM_CHAT_ID",   "")
        self._enabled     = bool(self.token and self.chat_id)
        self._last_send   = 0.0
        self._min_interval = 30.0
        self._sent_count  = 0

    @property
    def is_configured(self) -> bool:
        return self._enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Core send ──────────────────────────────────────────────────────

    def send_message(self, text: str, force: bool = False) -> bool:
        """
        Send raw Markdown message.
        Returns True on success, False if rate-limited or unconfigured.
        Set force=True to bypass rate-limit (e.g. for critical alerts).
        """
        if not self._enabled:
            return False
        elapsed = time.time() - self._last_send
        if not force and elapsed < self._min_interval:
            return False        # silently skip

        url     = _API.format(token=self.token, method="sendMessage")
        payload = json.dumps({
            "chat_id":    self.chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                self._last_send  = time.time()
                self._sent_count += 1
                return True
        except Exception:
            return False

    # ── Formatted alert types ─────────────────────────────────────────

    def send_alert(self, severity: str, sensor_id: str,
                   value: float, unit: str,
                   z_score: Optional[float] = None,
                   if_score: Optional[float] = None,
                   action: Optional[str] = None) -> bool:
        """Send a per-sensor anomaly alert."""
        emoji  = "🔴" if severity == "critical" else "🟡"
        force  = severity == "critical"          # bypass rate limit for critical
        lines  = [
            f"{emoji} *NeuralFarm Alert* — `{severity.upper()}`",
            "",
            f"*Sensor* : `{sensor_id.upper()}`",
            f"*Value*  : `{value}{unit}`",
        ]
        if z_score  is not None: lines.append(f"*Z-Score*: `{z_score:+.2f}σ`")
        if if_score is not None: lines.append(f"*IF-Score*: `{if_score:.3f}`")
        if action:               lines.append(f"*⚙ Action*: `{action}`")
        lines.append(f"*Time*   : `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
        return self.send_message("\n".join(lines), force=force)

    def send_health_report(self, health_score: int, anomaly_count: int,
                           actions: List[dict]) -> bool:
        """Send a periodic health summary (call every N ticks)."""
        emoji = "✅" if health_score >= 70 else "⚠️" if health_score >= 40 else "🚨"
        lines = [
            f"{emoji} *NeuralFarm Health Report*",
            "",
            f"*Health Score*     : `{health_score}/100`",
            f"*Active Anomalies* : `{anomaly_count}`",
        ]
        if actions:
            lines.append("*Auto-Actions*:")
            for a in actions[:4]:
                lines.append(f"  • `{a.get('action','—')}`")
        lines.append(f"*Time* : `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
        return self.send_message("\n".join(lines))

    def send_decision(self, sensor_id: str, action: str,
                      priority: str, value: float, trigger: str) -> bool:
        """Notify when AutoDecisionEngine fires a control action."""
        emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
        text  = (
            f"{emoji} *Auto-Control Action* — `{priority}`\n\n"
            f"*Sensor* : `{sensor_id.upper()}`\n"
            f"*Value*  : `{value}` ({trigger})\n"
            f"*Action* : `{action}`\n"
            f"*Time*   : `{time.strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        return self.send_message(text, force=(priority == "HIGH"))

    # ── Test ──────────────────────────────────────────────────────────

    def test_connection(self) -> str:
        """Test Telegram connectivity. Returns a status string."""
        if not self._enabled:
            return (
                "❌ Not configured.\n"
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."
            )
        ok = self.send_message(
            "🌱 *NeuralFarm*: Connection test ✅\n"
            f"Time: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
            force=True,
        )
        return "✅ Connected — test message sent!" if ok else "❌ Send failed."

    def stats(self) -> str:
        return f"TelegramAlerter: enabled={self._enabled}, sent={self._sent_count}"
