"""
database.py — SQLite Persistence Layer
══════════════════════════════════════
Provides disk-based persistence for NeuralFarm telemetry and events.

FIX: Previously opened a new sqlite3.connect() on every insert_telemetry()
     call. Under sub-second polling this caused connection contention and
     unnecessary file-open overhead. Now uses a single persistent connection
     with check_same_thread=False so it can be shared across threads safely.
"""

import sqlite3
import time
import json
import threading
from typing import Dict, Any


class NeuralFarmDB:
    """
    Persistent SQLite store for NeuralFarm sensor telemetry and events.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (created on first run).

    Tables
    ------
    telemetry : per-tick sensor readings with anomaly flags
    events    : system events — AI actions, Telegram alerts, twin stress
    """

    def __init__(self, db_path: str = "neuralfarm.db"):
        self.db_path = db_path
        # FIX: single persistent connection; check_same_thread=False so
        # Streamlit's threaded runner and the main loop can share it safely.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
        self._init_db()

    def _init_db(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick      INTEGER,
                    timestamp REAL,
                    sensor_id TEXT,
                    value     REAL,
                    unit      TEXT,
                    anomaly   BOOLEAN,
                    severity  TEXT
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick       INTEGER,
                    timestamp  REAL,
                    event_type TEXT,
                    details    TEXT
                )
            ''')
            self._conn.commit()

    # ── Telemetry ─────────────────────────────────────────────────────────

    def insert_telemetry(self, tick: int, results: Dict[str, Any]):
        """
        Insert ML results into the telemetry table.
        Accepts both MLResult objects and AdvancedMLPipeline dicts.
        """
        now     = time.time()
        records = []
        for sid, res in results.items():
            # AdvancedMLPipeline returns dicts; MLPipeline returns MLResult objects
            r        = res.get("base", res) if isinstance(res, dict) else res
            val      = r.value
            unit     = r.unit
            anomaly  = bool(getattr(r, "anomaly", False))
            severity = getattr(r, "severity", "ok")
            records.append((tick, now, sid, val, unit, anomaly, severity))

        with self._lock:
            self._conn.executemany('''
                INSERT INTO telemetry
                    (tick, timestamp, sensor_id, value, unit, anomaly, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            self._conn.commit()

    # ── Events ────────────────────────────────────────────────────────────

    def insert_event(self, tick: int, event_type: str, details: Dict):
        """Insert system events (AI actions, telegram alerts, twin stress)."""
        with self._lock:
            self._conn.execute('''
                INSERT INTO events (tick, timestamp, event_type, details)
                VALUES (?, ?, ?, ?)
            ''', (tick, time.time(), event_type, json.dumps(details)))
            self._conn.commit()

    # ── Query helpers ─────────────────────────────────────────────────────

    def get_recent_telemetry(self, sensor_id: str, n: int = 50):
        """Return the last n telemetry rows for a given sensor."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT tick, value, anomaly, severity "
                "FROM telemetry WHERE sensor_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (sensor_id, n),
            )
            return cur.fetchall()

    def get_anomaly_count(self, since_tick: int = 0) -> int:
        """Count anomalous readings since a given tick."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM telemetry WHERE anomaly = 1 AND tick >= ?",
                (since_tick,),
            )
            return cur.fetchone()[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self):
        """Explicitly close the database connection."""
        with self._lock:
            self._conn.close()

    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        try:
            self._conn.close()
        except Exception:
            pass
