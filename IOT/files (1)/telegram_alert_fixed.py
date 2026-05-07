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

  STEP 3 — Set environment variables before running:
    Linux/macOS : export TELEGRAM_BOT_TOKEN="123456..."
                  export TELEGRAM_CHAT_ID="987654321"
    Windows CMD : set TELEGRAM_BOT_TOKEN=123456...
                  set TELEGRAM_CHAT_ID=987654321
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional, List

# ── Credentials (environment variables only — never hard-code tokens) ────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")

_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlerter:
    """
    Sends structured Markdown messages to a Telegram chat.

    Rate-limits automatically to one message per 30 s to avoid
    Telegram flood restrictions. Falls back silently when unconfigured,
    but prints a clear warning at startup so the misconfiguration is visible.
    """

    def __init__(self):
        self.token     = TELEGRAM_BOT_TOKEN or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id   = TELEGRAM_CHAT_ID   or os.environ.get("TELEGRAM_CHAT_ID",   "")
        self._enabled  = bool(self.token and self.chat_id)
        self._last_send    = 0.0
        self._min_interval = 30.0
        self._sent_count   = 0
        self._skipped_count = 0

        # FIX: warn at startup instead of silently disabling
        if not self._enabled:
            missing = []
            if not self.token:    missing.append("TELEGRAM_BOT_TOKEN")
            if not self.chat_id:  missing.append("TELEGRAM_CHAT_ID")
            print(
                f"[Telegram] ⚠  Alerter DISABLED — missing env vars: "
                f"{', '.join(missing)}\n"
                f"           Set them to enable real-time Telegram alerts.\n"
                f"           (See telegram_alert.py for setup instructions.)"
            )

    @property
    def is_configured(self) -> bool:
        return self._enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Core send ─────────────────────────────────────────────────────────

    def send_message(self, text: str, force: bool = False) -> bool:
        """
        Send a raw Markdown message.

        Parameters
        ----------
        text  : Markdown-formatted message body.
        force : bypass rate-limit (use for critical alerts only).

        Returns True on success, False if disabled, rate-limited, or error.
        """
        if not self._enabled:
            return False

        elapsed = time.time() - self._last_send
        if not force and elapsed < self._min_interval:
            self._skipped_count += 1
            return False    # rate-limited — caller can retry or drop

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
                self._last_send   = time.time()
                self._sent_count += 1
                return True
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            print(f"[Telegram] HTTP {e.code} error: {body}")
            return False
        except Exception as exc:
            print(f"[Telegram] Send failed: {exc}")
            return False

    # ── Formatted alert types ─────────────────────────────────────────────

    def send_alert(
        self,
        severity:  str,
        sensor_id: str,
        value:     float,
        unit:      str,
        z_score:   Optional[float] = None,
        if_score:  Optional[float] = None,
        action:    Optional[str]   = None,
    ) -> bool:
        """Send a per-sensor anomaly alert."""
        emoji = "🔴" if severity == "critical" else "🟡"
        force = severity == "critical"      # bypass rate-limit for critical
        lines = [
            f"{emoji} *NeuralFarm Alert* — `{severity.upper()}`",
            "",
            f"*Sensor* : `{sensor_id.upper()}`",
            f"*Value*  : `{value}{unit}`",
        ]
        if z_score  is not None: lines.append(f"*Z-Score* : `{z_score:+.2f}σ`")
        if if_score is not None: lines.append(f"*IF-Score*: `{if_score:.3f}`")
        if action:               lines.append(f"*⚙ Action*: `{action}`")
        lines.append(f"*Time*   : `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
        return self.send_message("\n".join(lines), force=force)

    def send_health_report(
        self,
        health_score:  int,
        anomaly_count: int,
        actions:       List[dict],
    ) -> bool:
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
                lines.append(f"  • `{a.get('action', '—')}`")
        lines.append(f"*Time* : `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
        return self.send_message("\n".join(lines))

    def send_decision(
        self,
        sensor_id: str,
        action:    str,
        priority:  str,
        value:     float,
        trigger:   str,
    ) -> bool:
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

    # ── Test ──────────────────────────────────────────────────────────────

    def test_connection(self) -> str:
        """Test Telegram connectivity. Returns a human-readable status string."""
        if not self._enabled:
            missing = []
            if not self.token:   missing.append("TELEGRAM_BOT_TOKEN")
            if not self.chat_id: missing.append("TELEGRAM_CHAT_ID")
            return (
                f"❌ Not configured. Missing: {', '.join(missing)}\n"
                "Set environment variables and restart."
            )
        ok = self.send_message(
            "🌱 *NeuralFarm*: Connection test ✅\n"
            f"Time: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
            force=True,
        )
        return "✅ Connected — test message sent!" if ok else "❌ Send failed (check token/chat_id)."

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> str:
        return (
            f"TelegramAlerter | enabled={self._enabled} | "
            f"sent={self._sent_count} | skipped(rate-limit)={self._skipped_count}"
        )
