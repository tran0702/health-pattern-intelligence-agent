"""
global_baseline.py — Task 2 (approach (a)) + the anomaly-vs-baseline detector.

Asara direction, point 2: "Once the context defines a cohort, LLMs establish a
global baseline of normality for that group." We take **approach (a)**: the LLM
emits the expected-normal HR / HRV ranges *directly from physiological priors*,
indexed by the SubjectContext — no labelled cohort dataset is required.

Guardrail preserved: the LLM sets the REFERENCE (a static, per-cohort baseline).
It never flags individual samples. A separate deterministic detector
(`detect_against_baseline`) computes per-episode deviations and the interpretation
step translates them. LLM = the ruler; the detector = the measuring.

Every LLM call has a deterministic OFFLINE fallback (`population_default_baseline`)
built from textbook formulas (Tanaka HRmax, Karvonen HR-reserve, age/fitness
resting-HR and SDNN references). So the whole track runs and is verifiable with no
API key; the live LLM is used only when a key is present and `source="llm"`.

NOT medical advice — these are population reference ranges for anomaly detection.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from context_library import SubjectContext, EpisodeContext

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
DATA_DIR = REPO_ROOT / "data" / "context_baseline"
CACHE_DIR = DATA_DIR / "baseline_cache"
RESULTS_DIR = REPO_ROOT / "results" / "context_baseline"
GEMINI_MODEL = "gemini-3.1-flash-lite"


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Baseline schema — this is ALSO the LLM structured-output schema.
# --------------------------------------------------------------------------- #
class HrRange(BaseModel):
    low: float
    high: float

    def center(self) -> float:
        return 0.5 * (self.low + self.high)

    def half_width(self) -> float:
        return max(1e-6, 0.5 * (self.high - self.low))


class NormativeBaseline(BaseModel):
    """Expected-normal ranges for one cohort. Emitted by the LLM (approach (a))
    or by the deterministic fallback. `source` records which produced it."""
    resting_hr: HrRange
    sleep_hr: HrRange
    light_activity_hr: HrRange
    vigorous_activity_hr: HrRange
    hrv_sdnn_ms: HrRange
    max_hr_bpm: float
    rationale: str
    caveats: str
    source: str = "unset"          # "llm" | "population_default"


# --------------------------------------------------------------------------- #
# Deterministic fallback baseline (textbook physiology) — approach (a) offline.
# --------------------------------------------------------------------------- #
_AGE_MID = {"under_18": 16, "18_29": 24, "30_39": 35, "40_49": 45,
            "50_59": 55, "60_plus": 68, "unknown": 40}

# resting-HR band by fitness level (bpm). 'unknown' is deliberately wide.
_RESTING = {"athlete": (40, 58), "trained": (46, 66), "recreational": (54, 76),
            "sedentary": (62, 92), "unknown": (55, 92)}

# short-term SDNN reference band (ms) by age band. Wide when age unknown.
_SDNN = {"under_18": (55, 105), "18_29": (45, 95), "30_39": (40, 85),
         "40_49": (32, 75), "50_59": (28, 65), "60_plus": (22, 55),
         "unknown": (28, 95)}


def population_default_baseline(ctx: SubjectContext) -> NormativeBaseline:
    """Physiology-based baseline used offline and as the LLM fallback.

    Tanaka (2001): HRmax = 208 - 0.7*age.
    Karvonen HR-reserve: target = resting + intensity*(HRmax - resting),
      light = 30-50% HRR, vigorous = 60-85% HRR.
    Ranges widen automatically when age/fitness are 'unknown'.
    """
    age = _AGE_MID.get(ctx.age_band, 40)
    hrmax = 208 - 0.7 * age

    r_lo, r_hi = _RESTING.get(ctx.fitness_level, _RESTING["unknown"])
    conds = set(ctx.health_conditions)
    if "pregnancy" in conds:                      # resting HR rises ~10-15 bpm
        r_lo, r_hi = r_lo + 8, r_hi + 12
    if {"arrhythmia", "cardiac_history"} & conds:  # widen; do not pretend precision
        r_lo, r_hi = r_lo - 4, r_hi + 8
    if ctx.age_band == "unknown" or ctx.fitness_level == "unknown":
        r_lo, r_hi = r_lo - 3, r_hi + 6           # extra slack for missing context

    resting = HrRange(low=max(35.0, r_lo), high=r_hi)
    sleep = HrRange(low=max(33.0, resting.low - 10), high=max(40.0, resting.high - 12))

    def hrr(intensity):                            # Karvonen target HR
        return resting.center() + intensity * (hrmax - resting.center())

    light = HrRange(low=round(hrr(0.30), 1), high=round(hrr(0.50), 1))
    vigorous = HrRange(low=round(hrr(0.60), 1), high=round(hrr(0.85), 1))
    sdnn_lo, sdnn_hi = _SDNN.get(ctx.age_band, _SDNN["unknown"])

    rationale = (f"HRmax~{hrmax:.0f} bpm from Tanaka (208-0.7*{age}); resting band from "
                 f"fitness='{ctx.fitness_level}'; activity bands via Karvonen HR-reserve "
                 f"(light 30-50%, vigorous 60-85%); SDNN band by age='{ctx.age_band}'.")
    caveats = ("Population reference for anomaly detection, not diagnosis. "
               "Wearable HRV averaging differs from clinical 5-min SDNN. "
               f"Known cohort fields: {sum(ctx.availability().values())}/6 "
               "→ wider bands when unknown.")
    return NormativeBaseline(
        resting_hr=resting, sleep_hr=sleep, light_activity_hr=light,
        vigorous_activity_hr=vigorous, hrv_sdnn_ms=HrRange(low=sdnn_lo, high=sdnn_hi),
        max_hr_bpm=round(hrmax, 1), rationale=rationale, caveats=caveats,
        source="population_default")


# --------------------------------------------------------------------------- #
# LLM path (approach (a) live) — Gemini structured output, cached, key from .env
# --------------------------------------------------------------------------- #
def _load_key() -> str | None:
    """GEMINI_API_KEY from the first .env found: this folder, repo root, or the
    enrichment_experiment folder (where the project key currently lives)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    for p in (_THIS.parent / ".env", REPO_ROOT / ".env",
              REPO_ROOT / "notebooks" / "enrichment_experiment" / ".env"):
        if p.exists():
            load_dotenv(p, override=False)
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _prompt(ctx: SubjectContext) -> str:
    known = {k: v for k, v in ctx.model_dump().items()}
    return (
        "You are a physiology reference. Given a person's context, return the "
        "population EXPECTED-NORMAL ranges of heart-rate and HRV for that cohort "
        "(NOT this person's data — the group norm). Base it on established "
        "physiology (Tanaka HRmax, Karvonen HR-reserve, age/fitness resting-HR and "
        "SDNN references). If a field is 'unknown', widen the range accordingly. "
        "Fill every field. Do not flag anomalies; only give the reference ranges.\n"
        f"Context: {json.dumps(known)}"
    )


def _cache_path(prompt: str) -> Path:
    h = hashlib.sha256((GEMINI_MODEL + "\n" + prompt).encode()).hexdigest()[:32]
    return CACHE_DIR / f"{h}.json"


def establish_baseline(ctx: SubjectContext, source: str = "auto") -> NormativeBaseline:
    """Approach (a): produce the cohort's global baseline of normality.

    source="auto" : use the LLM if a key is available, else the offline default.
    source="llm"  : force the LLM (raises if no key).
    source="default": force the offline physiology baseline.
    """
    if source == "default":
        return population_default_baseline(ctx)

    key = _load_key()
    if source == "llm" and not key:
        raise RuntimeError("source='llm' but no GEMINI_API_KEY found (.env).")
    if source == "auto" and not key:
        return population_default_baseline(ctx)

    ensure_dirs()
    prompt = _prompt(ctx)
    cp = _cache_path(prompt)
    if cp.exists():
        return NormativeBaseline(**json.loads(cp.read_text()))
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        cfg = types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json",
            response_schema=NormativeBaseline,
            thinking_config=types.ThinkingConfig(thinking_budget=0))
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg)
        base = r.parsed
        base.source = "llm"
        cp.write_text(base.model_dump_json())
        return base
    except Exception as e:                          # any failure -> safe fallback
        base = population_default_baseline(ctx)
        base.caveats += f" [LLM unavailable: {type(e).__name__}; used offline default]"
        return base


# --------------------------------------------------------------------------- #
# Anomaly detection vs the baseline (deterministic; the LLM does NOT do this)
# --------------------------------------------------------------------------- #
def _range_for(activity: str, hour: int | None, base: NormativeBaseline) -> tuple[HrRange, str]:
    """Pick which baseline sub-range applies to an episode."""
    if activity == "sleep" or (activity in ("rest", "unknown") and hour is not None
                               and 0 <= hour <= 5):
        return base.sleep_hr, "sleep"
    if activity in ("rest", "unknown"):
        return base.resting_hr, "resting"
    if activity == "walk":
        return base.light_activity_hr, "light"
    if activity in ("run", "cycle", "strength", "other"):
        return base.vigorous_activity_hr, "vigorous"
    return base.resting_hr, "resting"


def _is_num(v) -> bool:
    return v is not None and not pd.isna(v)


def weather_hr_shift(temp_c, humidity=None) -> float:
    """bpm to add to the RESTING/SLEEP expected band for heat.

    Physiology: heat -> cutaneous vasodilation -> higher cardiac output, so a
    resting HR that would be abnormal on a cool day is expected when it is hot.
    The band is shifted up (not widened) so a hot-day resting HR is not falsely
    flagged. Modest and capped (<=~14 bpm); humidity compounds; cool/cold add 0.
    """
    if not _is_num(temp_c):
        return 0.0
    shift = 0.0
    if temp_c >= 26:
        shift = min(12.0, 0.6 * (temp_c - 26))
        if _is_num(humidity) and humidity >= 65:
            shift += 2.0
    return round(shift, 1)


def lifestyle_context(activity, location_type, temp_c, humidity=None) -> str:
    """Combine location + weather + activity into a coarse lifestyle situation
    (Asara point 3: interpret the state, not just flag it).

    NOTE: the weather feed carries temperature + humidity only (no sunshine/cloud
    field), so 'favourable' is a thermal-comfort proxy, not a measured clear-sky
    signal. Returns one of: favourable_outdoor, outdoor_heat, outdoor, hot_indoor,
    cold, neutral.
    """
    outdoor = (location_type in ("outdoors", "beach")
               or activity in ("walk", "run", "cycle"))
    mild = _is_num(temp_c) and 12 <= temp_c <= 24 and (not _is_num(humidity) or humidity <= 70)
    hot = _is_num(temp_c) and temp_c >= 30
    cold = _is_num(temp_c) and temp_c <= 6
    if outdoor and mild:
        return "favourable_outdoor"      # comfortable -> good for a hike / workout
    if outdoor and hot:
        return "outdoor_heat"
    if hot:
        return "hot_indoor"
    if cold:
        return "cold"
    if outdoor:
        return "outdoor"
    return "neutral"


def detect_against_baseline(episodes: pd.DataFrame, base: NormativeBaseline,
                            flag_at: float = 1.0) -> pd.DataFrame:
    """Per-episode deviation from the (optionally weather-adjusted) cohort baseline.

    `episodes` needs columns: avg_hr, activity, hour_of_day (int or NaN). Optional:
    location_type, weather_temp, weather_humidity -> when present the resting/sleep
    band is shifted up for heat (see `weather_hr_shift`) and a `lifestyle` tag
    (location+weather+activity) is attached.

    deviation = (distance OUTSIDE the applicable band) / band half-width; 0 if inside.
    An episode is flagged when deviation >= flag_at (default: one half-width outside).
    """
    devs, dirs, subr, band_lo, band_hi, wshift, life = [], [], [], [], [], [], []
    for _, row in episodes.iterrows():
        hour = None if pd.isna(row.get("hour_of_day")) else int(row["hour_of_day"])
        act = str(row.get("activity", "unknown"))
        loc = str(row.get("location_type", "unknown"))
        temp, hum = row.get("weather_temp"), row.get("weather_humidity")
        rng, which = _range_for(act, hour, base)
        # heat only modulates the passive (resting/sleep) expectation
        shift = weather_hr_shift(temp, hum) if which in ("resting", "sleep") else 0.0
        lo, hi = rng.low + shift, rng.high + shift
        hw = max(1e-6, 0.5 * (hi - lo))
        wshift.append(shift)
        life.append(lifestyle_context(act, loc, temp, hum))
        hr = row.get("avg_hr")
        if pd.isna(hr):
            devs.append(np.nan); dirs.append("na"); subr.append(which)
            band_lo.append(lo); band_hi.append(hi); continue
        if hr < lo:
            d, direction = (lo - hr) / hw, "below"
        elif hr > hi:
            d, direction = (hr - hi) / hw, "above"
        else:
            d, direction = 0.0, "inside"
        devs.append(round(d, 3)); dirs.append(direction); subr.append(which)
        band_lo.append(lo); band_hi.append(hi)

    out = episodes.copy()
    out["sub_range"] = subr
    out["band_lo"] = band_lo
    out["band_hi"] = band_hi
    out["weather_shift"] = wshift
    out["lifestyle"] = life
    out["deviation"] = devs
    out["direction"] = dirs
    out["is_anomaly"] = (out["deviation"] >= flag_at) & (out["direction"] != "inside")
    return out


# --------------------------------------------------------------------------- #
# Translate: what a deviation MEANS for lifestyle / health (Asara point 3).
# Non-diagnostic, context-aware, deterministic. An LLM could phrase this too,
# but the reference ranges (not the wording) are the scientific claim.
# --------------------------------------------------------------------------- #
def translate(row: pd.Series, ctx: SubjectContext) -> str:
    sub, direction = row["sub_range"], row["direction"]
    life = row.get("lifestyle", "neutral")
    if direction == "inside" or direction == "na":
        # even when HR is normal, the lifestyle context is worth surfacing
        if life == "favourable_outdoor":
            return ("Normal HR in favourable outdoor conditions — consistent with "
                    "recreational activity (e.g. a hike/walk in mild weather).")
        return ""
    if sub in ("resting", "sleep"):
        if direction == "above":
            base = ("Resting/sleep HR above the cohort norm — commonly acute stress, "
                    "illness, dehydration, alcohol/caffeine, poor sleep, or early "
                    "overtraining.")
            if life in ("outdoor_heat", "hot_indoor") or ctx.home_climate in ("hot_arid", "hot_humid"):
                base += (" Heat at this time/location plausibly contributes (the band is "
                         "already heat-adjusted, so this exceeds even the hot-day expectation).")
            return base
        return ("Resting/sleep HR below the cohort norm — often high aerobic fitness "
                "(athletic bradycardia); worth review only if symptomatic.")
    if sub in ("light", "vigorous"):
        if direction == "above":
            msg = ("Exercise HR above the expected band (near/over age-predicted max) — "
                   "check for a sensor artifact or unusually hard effort for this activity.")
            if life == "outdoor_heat":
                msg += " Outdoor heat pushes exercise HR above the same effort in cool weather."
            return msg
        msg = ("Exercise HR below the expected band — a lighter session or improving "
               "cardiovascular efficiency for the same activity.")
        if life == "favourable_outdoor":
            msg += " Conditions were favourable for outdoor activity (mild, comfortable)."
        return msg
    return ""
