"""
digital_twin.py — NeuralFarm Digital Twin
==========================================
Physiological crop growth model that:
  • Accumulates Growing Degree Days (GDD)
  • Computes Water Stress, Light, and pH stress factors
  • Estimates current yield % of potential
  • Projects future yield using ML forecasts
  • Logs all stress events

Scientific basis
----------------
  GDD/tick  = max(0, T_mean − T_base) / ticks_per_day
  WSF       = 1 − |SM − SM_opt| / SM_range       [0..1]
  LIF       = min(1, light / light_opt)           [0..1]
  PSF       = 1 − max(0, |pH − pH_opt| − 0.5)/2  [0..1]
  growth    = GDD × WSF × LIF × PSF  (per tick)
  yield_pct = min(100, total_GDD / GDD_to_harvest × 100)

Usage
-----
    from digital_twin import DigitalTwin
    twin = DigitalTwin(crop="tomato")
    metrics = twin.update(readings)   # call each tick
    print(twin.report())
    future  = twin.project_future(ml_summary, steps=50)
"""


class DigitalTwin:
    """
    Crop growth digital twin supporting three crop types.

    Parameters
    ----------
    crop          : "tomato" | "lettuce" | "pepper"
    ticks_per_day : how many simulation ticks equal one real day (default 120)
    """

    CROP_PARAMS = {
        "tomato": {
            "name": "Tomato", "T_base": 10.0, "T_opt": 24.0, "T_max": 35.0,
            "SM_opt": 65.0,  "SM_range": 40.0, "light_opt": 4000.0,
            "pH_opt": 6.5,   "gdd_to_harvest": 1500,
        },
        "lettuce": {
            "name": "Lettuce", "T_base": 4.0, "T_opt": 18.0, "T_max": 28.0,
            "SM_opt": 70.0,  "SM_range": 35.0, "light_opt": 3000.0,
            "pH_opt": 6.2,   "gdd_to_harvest": 600,
        },
        "pepper": {
            "name": "Bell Pepper", "T_base": 12.0, "T_opt": 26.0, "T_max": 38.0,
            "SM_opt": 60.0,  "SM_range": 40.0, "light_opt": 5000.0,
            "pH_opt": 6.8,   "gdd_to_harvest": 2000,
        },
    }

    def __init__(self, crop: str = "tomato", ticks_per_day: int = 120):
        self.p             = self.CROP_PARAMS.get(crop, self.CROP_PARAMS["tomato"])
        self.ticks_per_day = ticks_per_day
        self.tick          = 0
        self.total_gdd     = 0.0
        self.yield_pct     = 0.0
        self._cum_wsf      = 0.0   # for avg water stress
        self._n            = 0
        self.growth_log    = []    # list of per-tick metric dicts
        self.stress_log    = []    # list of {"tick", "event"} dicts

    # ── Core update ───────────────────────────────────────────────────────

    def update(self, readings: dict) -> dict:
        """
        Ingest one tick's sensor readings dict (from SensorNetwork.read_all).
        Returns a metrics dict for this tick.
        """
        self.tick += 1
        p = self.p

        def _val(key, default):
            r = readings.get(key)
            return r.value if r is not None else default

        temp  = _val("temp",     p["T_opt"])
        soil  = _val("soil",     p["SM_opt"])
        light = _val("light",    p["light_opt"])
        ph    = _val("ph",       p["pH_opt"])

        # Growing Degree Days (per tick)
        T_eff = max(0.0, min(temp - p["T_base"], p["T_opt"] - p["T_base"]))
        gdd   = T_eff / self.ticks_per_day

        # Stress factors [0..1]
        wsf   = max(0.0, 1.0 - abs(soil  - p["SM_opt"])  / p["SM_range"])
        lif   = min(1.0, light / p["light_opt"])
        ph_d  = max(0.0, abs(ph - p["pH_opt"]) - 0.5)
        psf   = max(0.0, 1.0 - ph_d / 2.0)

        growth_rate   = gdd * wsf * lif * psf
        self.total_gdd += gdd
        self._cum_wsf  += wsf
        self._n        += 1
        self.yield_pct  = min(100.0, self.total_gdd / p["gdd_to_harvest"] * 100.0)

        # Stress events
        stresses = []
        if wsf < 0.6:
            stresses.append(f"Water stress ({soil:.1f}% SM, WSF={wsf:.2f})")
        if lif < 0.5:
            stresses.append(f"Light deficiency ({light:.0f} lux, LIF={lif:.2f})")
        if psf < 0.7:
            stresses.append(f"pH stress (pH={ph:.2f}, PSF={psf:.2f})")
        if temp > p["T_max"] - 3:
            stresses.append(f"Heat stress ({temp:.1f}°C)")
        if temp < p["T_base"] + 2:
            stresses.append(f"Chilling stress ({temp:.1f}°C)")

        for s in stresses:
            self.stress_log.append({"tick": self.tick, "event": s})

        metrics = {
            "tick":             self.tick,
            "gdd_tick":         round(gdd,         4),
            "total_gdd":        round(self.total_gdd, 2),
            "yield_pct":        round(self.yield_pct, 2),
            "wsf":              round(wsf,  3),
            "lif":              round(lif,  3),
            "psf":              round(psf,  3),
            "growth_rate":      round(growth_rate, 6),
            "stresses":         stresses,
            "days_elapsed":     round(self.tick / self.ticks_per_day, 2),
            "days_to_harvest":  self._days_to_harvest(),
        }
        self.growth_log.append(metrics)
        return metrics

    # ── Helpers ───────────────────────────────────────────────────────────

    def _days_to_harvest(self) -> float:
        if self.tick == 0 or self.total_gdd == 0:
            return float("inf")
        avg_gdd    = self.total_gdd / self.tick
        remaining  = self.p["gdd_to_harvest"] - self.total_gdd
        if remaining <= 0:
            return 0.0
        return round(remaining / max(avg_gdd, 1e-9) / self.ticks_per_day, 1)

    # ── Future projection ─────────────────────────────────────────────────

    def project_future(self, ml_summary: dict, steps: int = 30) -> dict:
        """
        Project yield gain over next `steps` ticks using ML 5-step forecasts
        (linearly extrapolated). Requires MLPipeline.get_summary() output.
        """
        p = self.p
        def _f(key, default):
            return ml_summary.get(key, {}).get("forecast_5") or default

        f_temp  = _f("temp",  p["T_opt"])
        f_soil  = _f("soil",  p["SM_opt"])
        f_light = _f("light", p["light_opt"])
        f_ph    = _f("ph",    p["pH_opt"])

        proj_gdd = 0.0
        for _ in range(steps):
            T_eff   = max(0.0, min(f_temp - p["T_base"], p["T_opt"] - p["T_base"]))
            wsf     = max(0.0, 1.0 - abs(f_soil  - p["SM_opt"])  / p["SM_range"])
            lif     = min(1.0, f_light / p["light_opt"])
            ph_d    = max(0.0, abs(f_ph - p["pH_opt"]) - 0.5)
            psf     = max(0.0, 1.0 - ph_d / 2.0)
            proj_gdd += (T_eff / self.ticks_per_day) * wsf * lif * psf

        future_yield = min(100.0, (self.total_gdd + proj_gdd) / p["gdd_to_harvest"] * 100)
        return {
            "current_yield_pct":   round(self.yield_pct, 2),
            "projected_yield_pct": round(future_yield,   2),
            "yield_gain":          round(future_yield - self.yield_pct, 3),
            "steps_ahead":         steps,
            "projected_gdd":       round(proj_gdd, 4),
        }

    # ── Report ────────────────────────────────────────────────────────────

    def report(self) -> str:
        if not self.growth_log:
            return "No data — call update() first."
        lt  = self.growth_log[-1]
        avg_wsf = self._cum_wsf / max(self._n, 1)
        lines = [
            "=" * 58,
            f"  Digital Twin — {self.p['name']}",
            "=" * 58,
            f"  Tick           : {self.tick}",
            f"  Days elapsed   : {lt['days_elapsed']:.1f}",
            f"  Total GDD      : {lt['total_gdd']:.2f}",
            f"  Yield estimate : {lt['yield_pct']:.1f} % of potential",
            f"  Days to harvest: {lt['days_to_harvest']}",
            f"  Avg water stress: {(1-avg_wsf)*100:.1f} %",
            f"  Stress events  : {len(self.stress_log)} total",
        ]
        if self.stress_log:
            for e in self.stress_log[-3:]:
                lines.append(f"    • [T{e['tick']}] {e['event']}")
        lines.append("=" * 58)
        return "\n".join(lines)
