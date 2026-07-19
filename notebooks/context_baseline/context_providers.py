"""
context_providers.py — Task 1 (the missing half): ESTABLISH context from data.

The Context Library (context_library.py) defines WHAT a context is. This module
is the set of PROVIDERS that predict those attributes from the initial dataset,
before the main pipeline runs — realising the meeting's "predicted age / fitness /
location type" and Leelanga's "context is established externally, the pipeline
receives it".

Subject-level providers (predict the cohort descriptor from HR / weather):
  * predict_age_band      — invert Tanaka HRmax from observed peak HR.
  * predict_fitness_level — resting-HR percentile classified through the age/sex
    Resting-HR Chart (Athlete..Poor), collapsed to the 4-level FitnessLevel vocab.
  * predict_home_climate  — from the weather distribution (temp + humidity),
    grounded in the geocoded home place name when the location_context track has run.
Episode-level context (activity / location_type) is already derived in File 2 /
the enrichment C2 geocoder, so those are wired in, not re-implemented here.

Every provider returns a `FieldEstimate(value, confidence, evidence)` and degrades
to ('unknown', 0.0, why) when its inputs are missing — the library stays robust to
partial data. `build_subject_context` fuses them with any user-provided overrides
(user value always wins) into a `SubjectContext`.

Runs on real data/processed/*.parquet when present; otherwise callers pass frames
directly (see demo_context_providers.py for synthetic-persona validation).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_library import SubjectContext             # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = REPO_ROOT / "data" / "processed"

# Tanaka (2001): HRmax = 208 - 0.7*age  =>  age = (208 - HRmax)/0.7
_TANAKA_A, _TANAKA_B = 208.0, 0.7


@dataclass
class FieldEstimate:
    value: str
    confidence: float          # 0..1
    evidence: str

    def known(self) -> bool:
        return self.value != "unknown"


@dataclass
class ContextEstimation:
    context: SubjectContext
    estimates: dict[str, FieldEstimate] = field(default_factory=dict)

    def confidence(self) -> dict[str, float]:
        return {k: e.confidence for k, e in self.estimates.items()}

    def report(self) -> str:
        lines = []
        for k, e in self.estimates.items():
            src = "user" if e.evidence.startswith("user-provided") else "predicted"
            lines.append(f"  {k:18s} = {e.value:<12} conf={e.confidence:.2f}  [{src}] {e.evidence}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# helpers to pull HR values / workout flag from whatever frame we were given
# --------------------------------------------------------------------------- #
def _hr_values(frames: dict) -> np.ndarray:
    """All per-reading HR values (prefer raw; fall back to per-window avg_hr)."""
    if "hr_raw" in frames and "value" in frames["hr_raw"]:
        return frames["hr_raw"]["value"].to_numpy(dtype=float)
    if "hr_features" in frames and "avg_hr" in frames["hr_features"]:
        return frames["hr_features"]["avg_hr"].to_numpy(dtype=float)
    return np.array([])


def _peak_hr(frames: dict) -> tuple[float | None, int, str, bool]:
    """Observed peak HR (robust high percentile). The bool is `from_workout`:
    a peak seen during exercise is a trustworthy lower bound on true HRmax; a peak
    from resting-only data badly underestimates it, so callers must not trust it."""
    # 1) best: max_hr during workout windows (people approach true max in exercise)
    ep = frames.get("episodes")
    if ep is not None and "is_workout" in ep and "max_hr" in ep:
        w = ep.loc[ep["is_workout"] == True, "max_hr"].dropna()          # noqa: E712
        if len(w) >= 20:
            return float(np.percentile(w, 99)), len(w), "workout max_hr p99", True
    # 2) workouts table + raw HR inside workout intervals
    if "workouts" in frames and "hr_raw" in frames:
        wk = frames["workouts"].dropna(subset=["start_time", "end_time"])
        hr = frames["hr_raw"]
        if len(wk) and "datetime" in hr:
            mask = np.zeros(len(hr), bool)
            dt = hr["datetime"]
            for s, e in zip(wk["start_time"], wk["end_time"]):
                mask |= (dt >= s) & (dt <= e)
            vals = hr.loc[mask, "value"].to_numpy(dtype=float)
            if len(vals) >= 50:
                return float(np.percentile(vals, 99.5)), int(len(vals)), "workout-window HR p99.5", True
    # 3) fallback: high percentile of all HR (weak — resting data rarely nears max)
    v = _hr_values(frames)
    if len(v) >= 100:
        return float(np.percentile(v, 99.9)), int(len(v)), "all-HR p99.9 (no workout data)", False
    return None, 0, "insufficient HR data", False


def _resting_hr(frames: dict) -> tuple[float | None, int, str]:
    """Resting-HR estimate = low percentile of non-workout HR."""
    ep = frames.get("episodes")
    if ep is not None and "avg_hr" in ep:
        rest = ep.loc[ep.get("is_workout", False) != True, "avg_hr"].dropna()  # noqa: E712
        if len(rest) >= 30:
            return float(np.percentile(rest, 10)), len(rest), "non-workout avg_hr p10"
    v = _hr_values(frames)
    if len(v) >= 100:
        return float(np.percentile(v, 10)), int(len(v)), "all-HR p10"
    return None, 0, "insufficient HR data"


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def _age_to_band(age: float) -> str:
    return ("under_18" if age < 18 else "18_29" if age < 30 else "30_39" if age < 40
            else "40_49" if age < 50 else "50_59" if age < 60 else "60_plus")


# Observed workout peak (p99.5) is a near-max effort, ~97% of true HRmax; de-bias.
_PEAK_TO_MAX = 0.97


def predict_age_band(frames: dict) -> FieldEstimate:
    peak, n, how, from_wk = _peak_hr(frames)
    if peak is None:
        return FieldEstimate("unknown", 0.0, f"no peak-HR signal ({how})")
    if from_wk:
        hrmax_est = peak / _PEAK_TO_MAX                     # de-bias toward true max
        age = (_TANAKA_A - hrmax_est) / _TANAKA_B
        # DECISIVE gate: a low observed peak means the subject never approached max,
        # so we cannot tell an old person at max from a young person who never exerted
        # -> low confidence regardless of sample count. (Real Apple-Health subject:
        # 62% of workouts are walking, workout peak p99.5=140 -> age must NOT be trusted.)
        # peak < 160 bpm cannot be a plausible near-max for anyone but the very old,
        # so it can't pin age -> force low confidence (=> 'unknown'), consistent with
        # the note below and independent of which data path produced the peak.
        peak_factor = 1.0 if peak >= 175 else 0.6 if peak >= 160 else 0.15
        conf = round(peak_factor * (0.5 + 0.5 * min(1.0, n / 200)), 2)
        note = (f"HRmax_obs={peak:.0f} ({how}); HRmax_est={hrmax_est:.0f} -> age~{age:.0f}."
                + ("" if peak >= 160 else " Peak far below a plausible near-max effort, so "
                   "this age estimate is unreliable (subject rarely elevates HR)."))
    else:
        hrmax_est = peak                                    # unreliable lower bound
        age = (_TANAKA_A - hrmax_est) / _TANAKA_B
        conf = 0.2                                          # -> falls below min_conf => 'unknown'
        note = (f"HRmax_obs={peak:.0f} ({how}) -> age~{age:.0f}. No workout-level HR seen, so "
                "the peak underestimates true HRmax; low confidence.")
    return FieldEstimate(_age_to_band(age), conf, note)


# --------------------------------------------------------------------------- #
# Resting-HR fitness chart (age x sex). Standard "Resting Heart Rate Chart":
# 7 categories Athlete..Poor. Stored as the INCLUSIVE UPPER BOUND of each of the
# first six categories per (sex, age-bracket); anything above the 6th is 'poor'.
# (bpm)              Athlete Excellent Great Good Average BelowAvg   [Poor = else]
# --------------------------------------------------------------------------- #
_RHR_MEN = {
    "18_25": (55, 61, 65, 69, 73, 81), "26_35": (54, 61, 65, 70, 74, 81),
    "36_45": (56, 62, 66, 70, 75, 82), "46_55": (57, 63, 67, 71, 76, 83),
    "56_65": (56, 61, 67, 71, 75, 81), "65p":   (55, 61, 65, 69, 73, 79),
}
_RHR_WOMEN = {
    "18_25": (60, 65, 69, 73, 78, 84), "26_35": (59, 64, 68, 72, 76, 82),
    "36_45": (59, 64, 69, 73, 78, 84), "46_55": (60, 65, 69, 73, 77, 83),
    "56_65": (59, 64, 68, 73, 77, 83), "65p":   (59, 64, 68, 72, 76, 84),
}
_CHART_CATS = ("athlete", "excellent", "great", "good", "average", "below_average")
# collapse the chart's 7 categories onto the project's 4-level FitnessLevel vocab
_CAT_TO_LEVEL = {"athlete": "athlete", "excellent": "trained", "great": "trained",
                 "good": "recreational", "average": "recreational",
                 "below_average": "sedentary", "poor": "sedentary"}
# SubjectContext age_band -> chart age bracket (by band midpoint)
_AGE_TO_BRACKET = {"under_18": "18_25", "18_29": "18_25", "30_39": "26_35",
                   "40_49": "36_45", "50_59": "46_55", "60_plus": "56_65"}


def _pool(rows) -> tuple:
    """Element-wise mean of several (a,b,c,d,e,f) bound rows -> one rounded row."""
    cols = list(zip(*rows))
    return tuple(round(sum(c) / len(c)) for c in cols)


def _chart_bounds(age_band: str, sex: str) -> tuple[tuple, str, bool]:
    """Return (six upper bounds, human descriptor, is_precise) for (age, sex).
    Falls back to pooling across the missing dimension so it never crashes and
    stays chart-derived (no arbitrary numbers) when age/sex is unknown."""
    tables = {"male": _RHR_MEN, "female": _RHR_WOMEN}
    bracket = _AGE_TO_BRACKET.get(age_band)
    precise = sex in tables and bracket is not None
    if sex in tables:
        tbl = tables[sex]
        rows = [tbl[bracket]] if bracket else list(tbl.values())
        desc = f"{sex} {bracket or 'age-pooled'}"
    else:                                   # sex unknown -> pool men+women
        both = list(_RHR_MEN.values()) + list(_RHR_WOMEN.values()) if not bracket \
            else [_RHR_MEN[bracket], _RHR_WOMEN[bracket]]
        rows, desc = both, f"sex-pooled {bracket or 'age-pooled'}"
    return _pool(rows), desc, precise


def _chart_category(rhr: float, bounds: tuple) -> str:
    for cat, hi in zip(_CHART_CATS, bounds):
        if rhr <= hi:
            return cat
    return "poor"


def predict_fitness_level(frames: dict, age_band: str = "unknown",
                          sex: str = "unknown") -> FieldEstimate:
    """Classify fitness from resting HR via the age/sex Resting-HR Chart.
    age_band + sex index the chart (they come from the resolved SubjectContext);
    if either is unknown the chart is pooled over that dimension. Output stays in
    the 4-level FitnessLevel vocab; the exact 7-level chart category is reported."""
    rhr, n, how = _resting_hr(frames)
    if rhr is None:
        return FieldEstimate("unknown", 0.0, f"no resting-HR signal ({how})")
    bounds, desc, precise = _chart_bounds(age_band, sex)
    cat = _chart_category(rhr, bounds)
    level = _CAT_TO_LEVEL[cat]
    conf = round((0.2 + 0.7 * min(1.0, n / 300)) * (1.0 if precise else 0.7), 2)
    return FieldEstimate(level, conf, f"resting_HR~{rhr:.0f} ({how}); "
                         f"chart={cat} [{desc}]")


def predict_home_climate(frames: dict) -> FieldEstimate:
    # geocoded home region (from the location_context track, if it has run) grounds
    # the climate in a real place name. Passed as a plain dict via load_frames'
    # "home_geo" key - an artefact contract, no cross-track import.
    geo = frames.get("home_geo") or {}
    place = geo.get("place")
    place_note = f", home={place}" if place else ""

    wx = frames.get("weather")
    if wx is None or "weather_temp" not in getattr(wx, "columns", []):
        # no weather -> fall back to the geocoded band if location_context provided one
        if geo.get("band") and geo["band"] != "unknown":
            return FieldEstimate(geo["band"], 0.4, f"geocoded home climate{place_note}")
        return FieldEstimate("unknown", 0.0, "no weather data")
    t = pd.to_numeric(wx["weather_temp"], errors="coerce").dropna()
    if len(t) < 100:
        return FieldEstimate("unknown", 0.0, "too little weather data")
    mean_t, warm_frac = float(t.mean()), float((t >= 30).mean())
    h = pd.to_numeric(wx.get("weather_humidity", pd.Series(dtype=float)),
                      errors="coerce").dropna()
    humid = float(h.mean()) if len(h) else np.nan
    if mean_t <= 10:
        val = "cold"
    elif warm_frac >= 0.15 or mean_t >= 22:
        val = "hot_humid" if (humid == humid and humid >= 60) else "hot_arid"
    else:
        val = "temperate"
    conf = round(0.4 + 0.5 * min(1.0, len(t) / 2000), 2)
    hnote = f", humidity~{humid:.0f}%" if humid == humid else ""
    return FieldEstimate(val, conf,
                         f"mean_temp~{mean_t:.1f}C{hnote}, warm_frac={warm_frac:.2f}{place_note}")


# --------------------------------------------------------------------------- #
# Assemble the SubjectContext (predicted fields + user overrides)
# --------------------------------------------------------------------------- #
_PREDICTORS = {
    "age_band": predict_age_band,
    "fitness_level": predict_fitness_level,
    "home_climate": predict_home_climate,
}
# fields only a person can supply (no data predictor) — accepted via `user`
_USER_ONLY = ("sex", "health_conditions", "goal")


def build_subject_context(frames: dict, user: dict | None = None,
                          min_conf: float = 0.25) -> ContextEstimation:
    """Fuse data-predicted fields with user overrides into a SubjectContext.

    * A user-supplied value always wins (source of truth) and gets confidence 1.
    * A predicted field is accepted only if confidence >= min_conf, else 'unknown'
      (robustness: a weak signal must not masquerade as known context).
    """
    user = user or {}
    est: dict[str, FieldEstimate] = {}
    values: dict[str, object] = {}

    for f, fn in _PREDICTORS.items():
        if f in user and user[f] not in (None, "unknown"):
            est[f] = FieldEstimate(str(user[f]), 1.0, "user-provided (overrides prediction)")
            values[f] = user[f]
            continue
        if f == "fitness_level":
            # the chart needs age + sex; age_band is resolved earlier in this loop,
            # sex is user-only -> read both from what we have so far.
            e = fn(frames, age_band=values.get("age_band", "unknown"),
                   sex=user.get("sex", "unknown"))
        else:
            e = fn(frames)
        if e.confidence < min_conf:
            e = FieldEstimate("unknown", e.confidence, e.evidence + " -> below min_conf")
        est[f] = e
        values[f] = e.value

    for f in _USER_ONLY:
        if f in user and user[f] not in (None, "unknown"):
            v = user[f]
            est[f] = FieldEstimate(str(v), 1.0, "user-provided")
            values[f] = v
        else:
            est[f] = FieldEstimate("unknown", 0.0, "not provided (no data predictor)")

    hc = values.get("health_conditions", "unknown")
    ctx = SubjectContext(
        age_band=values.get("age_band", "unknown"),
        sex=values.get("sex", "unknown"),
        fitness_level=values.get("fitness_level", "unknown"),
        health_conditions=hc if isinstance(hc, list) else [hc],
        goal=values.get("goal", "unknown"),
        home_climate=values.get("home_climate", "unknown"),
    )
    return ContextEstimation(context=ctx, estimates=est)


def load_frames(proc_dir: Path = PROC_DIR) -> dict:
    """Load whatever processed parquets exist (real-data path). Missing = skipped.

    Also picks up the location_context track's home_climate.json artefact (if the
    module has run) as frames["home_geo"], so predict_home_climate can ground the
    climate in a real place name without importing that track."""
    frames: dict = {}
    names = {"hr_raw": "hr_raw.parquet", "hr_features": "hr_features.parquet",
             "workouts": "workouts.parquet", "episodes": "behavioral_episodes.parquet",
             "weather": "weather_hourly.parquet"}
    for key, fn in names.items():
        p = proc_dir / fn
        if p.exists():
            frames[key] = pd.read_parquet(p)
    hc = REPO_ROOT / "results" / "location_context" / "home_climate.json"
    if hc.exists():
        try:
            frames["home_geo"] = json.loads(hc.read_text(encoding="utf-8"))
        except Exception:
            pass
    return frames


def attach_weather(episodes: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """Populate episode weather_temp / weather_humidity from the hourly weather
    table by matching the local clock hour.

    File 2 fetches weather into weather_hourly.parquet but does not join it back
    onto the episodes, so episodes arrive with all-NaN weather columns. This
    repairs that so downstream context (band modulation, lifestyle, GCN feature)
    can actually use temperature + humidity. No-op if weather is missing.
    """
    if weather is None or "wx_hour" not in getattr(weather, "columns", []):
        return episodes
    ep = episodes.copy()
    wx = weather.dropna(subset=["wx_hour"]).copy()

    def _local_hour(s):
        # match on the local wall-clock hour; strip tz first so the autumn DST
        # transition (an ambiguous 02:00) does not raise, unlike flooring in tz.
        naive = s.dt.tz_localize(None) if s.dt.tz is not None else s
        return naive.dt.floor("h")

    wx["_h"] = _local_hour(wx["wx_hour"])
    wx = wx.drop_duplicates("_h").set_index("_h")
    ep_hour = _local_hour(ep["datetime"])
    ep["weather_temp"] = ep_hour.map(wx["weather_temp"]).to_numpy()
    ep["weather_humidity"] = ep_hour.map(wx["weather_humidity"]).to_numpy()
    return ep
