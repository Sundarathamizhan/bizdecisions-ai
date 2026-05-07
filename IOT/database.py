"""
database.py — SQLite Persistence Layer
══════════════════════════════════════
Provides disk-based persistence for NeuralFarm telemetry and events.
"""

import sqlite3
import time
from typing import Dict, List, Any
import json

class NeuralFarmDB:
    def __init__(self, db_path: str = "neuralfarm.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Telemetry metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick INTEGER,
                    timestamp REAL,
                    sensor_id TEXT,
                    value REAL,
                    unit TEXT,
                    anomaly BOOLEAN,
                    severity TEXT
                )
            ''')
            # Events and actions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick INTEGER,
                    timestamp REAL,
                    event_type TEXT,
                    details TEXT
                )
            ''')
            conn.commit()

    def insert_telemetry(self, tick: int, results: Dict[str, Any]):
        """Insert ML results into DB."""
        now = time.time()
        records = []
        for sid, res in results.items():
            # Support both base MLResult and Advanced ML dicts
            r = res.get("base", res) if isinstance(res, dict) else res
            val = r.value
            unit = r.unit
            anomaly = getattr(r, "anomaly", False)
            severity = getattr(r, "severity", "ok")

            records.append((tick, now, sid, val, unit, bool(anomaly), severity))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT INTO telemetry (tick, timestamp, sensor_id, value, unit, anomaly, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()

    def insert_event(self, tick: int, event_type: str, details: Dict):
        """Insert system events like AI actions, telegram alerts, or twin stress."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (tick, timestamp, event_type, details)
                VALUES (?, ?, ?, ?)
            ''', (tick, time.time(), event_type, json.dumps(details)))
            conn.commit()
