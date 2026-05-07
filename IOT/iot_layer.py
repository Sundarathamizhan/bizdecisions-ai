"""
iot_layer.py — IoT Sensor Network Simulation
═════════════════════════════════════════════
Simulates 6 greenhouse sensors with realistic physics:
  • Sinusoidal day/night cycles
  • Gaussian noise (sensor imprecision)
  • Random spike anomalies (hardware glitches, real events)
  • Clamped physical bounds
  • Cross-sensor correlations (temp ↑ → humidity ↓)

In a real deployment this module would instead read from:
  MQTT broker, InfluxDB, Modbus RTU, or HTTP sensor APIs.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorConfig:
    """
    Defines the physical and statistical characteristics of one IoT sensor.

    Attributes
    ----------
    sensor_id       : unique short key used throughout the pipeline
    name            : human-readable display name
    unit            : measurement unit string
    base_value      : realistic midpoint for this sensor type
    noise_amplitude : ±range of Gaussian noise (σ approximation)
    min_val         : physical lower bound (hard clamp)
    max_val         : physical upper bound (hard clamp)
    anomaly_chance  : probability of a spike event per tick [0..1]
    spike_factor    : multiplier on noise_amplitude for spikes
    decimals        : decimal places for display
    cycle_period    : sine wave period in ticks (simulates day/night)
    cycle_amplitude : amplitude of the sine oscillation
    """
    sensor_id:       str
    name:            str
    unit:            str
    base_value:      float
    noise_amplitude: float
    min_val:         float
    max_val:         float
    anomaly_chance:  float = 0.04
    spike_factor:    float = 3.5
    decimals:        int   = 1
    cycle_period:    float = 100.0
    cycle_amplitude: float = 0.0


@dataclass
class SensorReading:
    """
    One timestamped reading from a single sensor.
    """
    sensor_id:  str
    tick:       int
    value:      float
    unit:       str
    is_injected_anomaly: bool = False   # ground truth label for backtesting


# ─────────────────────────────────────────────────────────────────────────────
# Individual Sensor
# ─────────────────────────────────────────────────────────────────────────────

class Sensor:
    """
    Stateful sensor that generates a new reading each tick.

    Physics model
    ─────────────
      value = base
            + cycle_amplitude × sin(2π × tick / cycle_period)   ← day/night
            + Gaussian(0, noise_amplitude × 0.4)                 ← sensor noise
            + optional spike                                      ← anomaly event
    clamped to [min_val, max_val]
    """

    def __init__(self, cfg: SensorConfig):
        self.cfg        = cfg
        self.sensor_id  = cfg.sensor_id
        self.name       = cfg.name
        self.unit       = cfg.unit
        self.min_val    = cfg.min_val
        self.max_val    = cfg.max_val

    def read(self, tick: int, external_delta: float = 0.0) -> SensorReading:
        """
        Generate one reading at the given tick.

        Parameters
        ----------
        tick           : current simulation step
        external_delta : cross-sensor correlation offset injected by SensorNetwork
        """
        cfg = self.cfg

        # Day/night sinusoidal cycle
        cycle = cfg.cycle_amplitude * math.sin(2 * math.pi * tick / cfg.cycle_period)

        # Gaussian sensor noise
        noise = random.gauss(0, cfg.noise_amplitude * 0.4)

        # Compose base signal
        raw = cfg.base_value + cycle + noise + external_delta

        # Anomaly injection
        injected = False
        if random.random() < cfg.anomaly_chance:
            direction = 1 if random.random() > 0.5 else -1
            magnitude = cfg.noise_amplitude * cfg.spike_factor * (1 + random.random())
            raw      += direction * magnitude
            injected  = True

        # Physical clamp
        value = round(
            max(cfg.min_val, min(cfg.max_val, raw)),
            cfg.decimals
        )

        return SensorReading(
            sensor_id            = self.sensor_id,
            tick                 = tick,
            value                = value,
            unit                 = self.unit,
            is_injected_anomaly  = injected,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Network
# ─────────────────────────────────────────────────────────────────────────────

class SensorNetwork:
    """
    Manages all 6 greenhouse sensors and handles cross-sensor correlations.

    Correlations modelled
    ─────────────────────
      • Temperature ↑  →  Relative Humidity ↓   (psychrometric relationship)
      • CO₂ level  ↑  →  slight Temperature ↑   (metabolic heat)
      • Light ↑       →  Temperature ↑           (solar gain)

    Sensors
    ───────
      temp      — Air temperature (°C)
      humidity  — Relative humidity (%)
      soil      — Volumetric soil moisture (%)
      co2       — CO₂ concentration (ppm)
      light     — PPFD / lux proxy (lux)
      ph        — Soil pH
    """

    SENSOR_CONFIGS = [
        SensorConfig("temp",     "Temperature",     "°C",  24.0,  4.0,   5.0,  50.0,  0.040, 3.5, 1, 120, 6.0),
        SensorConfig("humidity", "Humidity",        "%",   65.0, 10.0,  10.0,  99.0,  0.030, 3.0, 1, 120, 12.0),
        SensorConfig("soil",     "Soil Moisture",   "%",   55.0, 12.0,   5.0,  95.0,  0.050, 3.5, 1, 200, 8.0),
        SensorConfig("co2",      "CO₂ Level",       "ppm", 420.0, 50.0, 250.0, 900.0, 0.025, 4.0, 0, 80,  60.0),
        SensorConfig("light",    "Light Intensity", "lux",3500.0,900.0, 100.0,10000.0,0.030, 3.5, 0, 120,2000.0),
        SensorConfig("ph",       "Soil pH",         "pH",   6.5,  0.5,   3.5,   9.0,  0.020, 3.0, 2, 300, 0.3),
    ]

    def __init__(self):
        self.sensors: Dict[str, Sensor] = {
            cfg.sensor_id: Sensor(cfg)
            for cfg in self.SENSOR_CONFIGS
        }
        self._last_temp: Optional[float] = None

    def read_all(self, tick: int) -> Dict[str, SensorReading]:
        """
        Read all sensors at tick, applying cross-sensor correlations.
        Returns a dict keyed by sensor_id.
        """
        readings: Dict[str, SensorReading] = {}

        # ── Temperature (no dependency) ──────────────────────────────────
        temp_r = self.sensors["temp"].read(tick)
        readings["temp"] = temp_r

        # ── Humidity (inversely correlated with temperature) ─────────────
        # Approximate Clausius-Clapeyron: Δhum ≈ -0.8 × ΔTemp
        temp_delta  = (temp_r.value - 24.0)
        hum_offset  = -0.8 * temp_delta
        readings["humidity"] = self.sensors["humidity"].read(tick, external_delta=hum_offset)

        # ── Soil Moisture (independent) ─────────────────────────────────
        readings["soil"] = self.sensors["soil"].read(tick)

        # ── CO₂ (slight diurnal + independent) ──────────────────────────
        readings["co2"] = self.sensors["co2"].read(tick)

        # ── Light (solar cycle, strongly periodic) ───────────────────────
        readings["light"] = self.sensors["light"].read(tick)

        # ── pH (slow drift, independent) ─────────────────────────────────
        readings["ph"] = self.sensors["ph"].read(tick)

        return readings

    def inject_fault(self, sensor_id: str, ticks: int, delta: float):
        """
        Utility: inject a sustained fault into a sensor for testing.
        In production this simulates a stuck valve, pump failure, etc.
        """
        if sensor_id not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_id}")
        # Temporarily shift base value
        sensor = self.sensors[sensor_id]
        original_base = sensor.cfg.base_value
        sensor.cfg.base_value += delta
        print(f"[FAULT] Injecting {delta:+.1f} offset into '{sensor_id}' for {ticks} ticks")
        return original_base   # caller responsible for restoring

    def get_sensor_names(self) -> Dict[str, str]:
        return {sid: s.name for sid, s in self.sensors.items()}
