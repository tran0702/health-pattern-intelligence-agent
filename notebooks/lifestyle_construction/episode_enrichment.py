"""
episode_enrichment.py — Stage A of the Lifestyle Construction track.

Turns File 2's *normalize-only* context step into real **enrichment**: each 15-min episode is
given a richer, controlled-vocabulary context from the signal it actually has (physiological +
temporal fingerprint, plus a geocoded place when one exists). This is the project's core pattern —
**code measures the numbers, the LLM names them** — so the LLM never ingests raw rows; it only maps
a compact per-episode SIGNATURE onto a controlled label set.

Why a signature (not per-row): the 104,565 episodes collapse to a few hundred distinct
(is_workout, activity, location, hour-bucket, weekday, HR-band, temp-band) signatures, so the LLM is
called a few hundred times (temp=0, cached) rather than per row — cheap and reproducible.

Enriched fields added per episode:
  * activity_context  — the per-episode SITUATION (a small controlled enum defined here, because the
                        Task-1 vocab is subject-level and has no per-episode dimension). This is the
                        LLM's inferential contribution; on non-workout episodes it is a *plausible
                        inference* from HR + time + weather, NOT verified ground truth.
  * weather_ctx       — thermal-stress band, drawn from the frozen Task-1 WeatherVocab (deterministic
                        from temp/humidity; the LLM is not needed for it).
  * workout_type_ctx  — for workout episodes, from the frozen Task-1 WorkoutTypeVocab.
  * location_semantic — the geocoded place (from location_context) when present, else 'unknown'.
  * enrich_source     — 'llm' | 'fallback', so a deterministic-fallback label is never reported as
                        an LLM result (honest-reporting guardrail).

Guardrails: semantics only (no anomaly/health judgment); temp=0 + disk cache + deterministic
fallback (runs fully offline); English identifiers; isolated track — reads data/processed/ +
results/location_context/ (read-only), writes only results/lifestyle_construction/. Does not touch
File 1-4.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Paths (isolated track)
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))                 # ...\Apple Health Data
PROC_DIR = os.path.join(_REPO_ROOT, "data", "processed")
LOC_RESULTS = os.path.join(_REPO_ROOT, "results", "location_context")
RESULTS_DIR = os.path.join(_REPO_ROOT, "results", "lifestyle_construction")
CACHE_DIR = os.path.join(_REPO_ROOT, "data", "lifestyle_construction", "enrich_cache")

EPISODES = os.path.join(PROC_DIR, "behavioral_episodes.parquet")
ENRICHED_OUT = os.path.join(RESULTS_DIR, "behavioral_episodes_enriched.parquet")
META_OUT = os.path.join(RESULTS_DIR, "l0_enrich_meta.json")

GEMINI_MODEL = "gemini-3.1-flash-lite"


# --------------------------------------------------------------------------- #
# Frozen Task-1 vocab (one-way import of the committed artifact; local fallback)
# --------------------------------------------------------------------------- #
try:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "notebooks", "context_vocab"))
    from generated_vocab import WeatherVocab, WorkoutTypeVocab  # type: ignore
    _VOCAB_SOURCE = "context_vocab.generated_vocab"
except Exception:                                                   # offline-safe fallback
    WeatherVocab = Literal["unknown", "thermoneutral", "cold_stress", "heat_stress",
                           "high_humidity", "extreme_heatwave"]
    WorkoutTypeVocab = Literal["unknown", "steady_state_cardio", "high_intensity_interval",
                               "resistance_training", "low_impact_steady", "mind_body",
                               "aquatic_exercise"]
    _VOCAB_SOURCE = "local-fallback"

# Per-episode SITUATION enum — complements the subject-level Task-1 vocab (which has no
# per-episode dimension). Deliberately small and controlled.
ActivityContext = Literal[
    "overnight_rest",     # low HR, night hours, not a workout
    "daytime_rest",       # resting tone during the working day
    "evening_rest",       # resting/wind-down in the evening
    "light_activity",     # elevated HR at rest (chores/errands/short exertion)
    "active_workout",     # a recorded workout episode
    "recovery",           # elevated HR shortly framed around workouts
    "unknown",
]
_ACTIVITY_CONTEXT_VALUES = list(ActivityContext.__args__)  # type: ignore[attr-defined]

# raw activity -> Task-1 workout_type (subset of the frozen ALIASES that applies here)
_ACT_TO_WORKOUT_TYPE = {
    "walk": "low_impact_steady", "row": "steady_state_cardio", "cycle": "steady_state_cardio",
    "run": "steady_state_cardio", "strength": "resistance_training",
}


class EnrichedContext(BaseModel):
    """LLM output for one signature — controlled vocab only, no anomaly/confidence field."""
    activity_context: ActivityContext
    workout_type: WorkoutTypeVocab


# --------------------------------------------------------------------------- #
# Deterministic banding (pure code — this is the "measure" half)
# --------------------------------------------------------------------------- #
def weather_band(temp: Optional[float], humidity: Optional[float]) -> str:
    """Map temperature/humidity to the frozen WeatherVocab thermal-stress band."""
    if temp is None or (isinstance(temp, float) and np.isnan(temp)):
        return "unknown"
    if temp >= 35:
        return "extreme_heatwave"
    if temp >= 27:
        return "heat_stress"
    if temp <= 8:
        return "cold_stress"
    if humidity is not None and not (isinstance(humidity, float) and np.isnan(humidity)) \
            and humidity >= 80 and temp >= 20:
        return "high_humidity"
    return "thermoneutral"


def _hour_bucket(hour: int) -> str:
    if 0 <= hour <= 5:
        return "night"
    if 6 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 17:
        return "afternoon"
    return "evening"


def personal_hr_bands(ep: pd.DataFrame) -> dict[str, float]:
    """The subject's own resting-HR percentiles (non-workout episodes) — used to band each
    episode's avg_hr RELATIVE to this person, never against a population number."""
    rest = ep.loc[~ep["is_workout"], "avg_hr"].dropna()
    return {"p25": float(rest.quantile(0.25)), "p50": float(rest.quantile(0.50)),
            "p90": float(rest.quantile(0.90))}


def _hr_band(avg_hr: Optional[float], bands: dict[str, float]) -> str:
    if avg_hr is None or (isinstance(avg_hr, float) and np.isnan(avg_hr)):
        return "unknown"
    if avg_hr < bands["p25"]:
        return "low"
    if avg_hr < bands["p50"]:
        return "normal"
    if avg_hr < bands["p90"]:
        return "elevated"
    return "high"


# --------------------------------------------------------------------------- #
# Signature (dedup key for the LLM) + description
# --------------------------------------------------------------------------- #
SIG_COLS = ["is_workout", "activity", "has_place", "hour_bucket", "is_weekday",
            "hr_band", "weather_ctx"]


def add_signature_columns(ep: pd.DataFrame, bands: dict[str, float]) -> pd.DataFrame:
    ep = ep.copy()
    hour = ep["datetime"].dt.hour
    ep["hour_bucket"] = hour.map(_hour_bucket)
    ep["is_weekday"] = ep["datetime"].dt.weekday < 5
    ep["has_place"] = ep["location_place"].notna()
    ep["hr_band"] = ep["avg_hr"].map(lambda v: _hr_band(v, bands))
    ep["weather_ctx"] = [weather_band(t, h)
                         for t, h in zip(ep["weather_temp"], ep["weather_humidity"])]
    return ep


def describe_signature(sig: dict) -> str:
    """A short natural-language description of one signature for the LLM (no raw rows)."""
    when = {"night": "overnight (00:00-05:59)", "morning": "morning (06:00-11:59)",
            "afternoon": "afternoon (12:00-17:59)", "evening": "evening (18:00-23:59)"}[sig["hour_bucket"]]
    day = "a weekday" if sig["is_weekday"] else "a weekend day"
    ctx = "during a recorded workout" if sig["is_workout"] else "at rest (no workout)"
    hr = {"low": "well below", "normal": "around", "elevated": "above",
          "high": "far above", "unknown": "at an unknown level relative to"}[sig["hr_band"]]
    place = "a known GPS place is attached" if sig["has_place"] else "no location is known"
    wx = sig["weather_ctx"].replace("_", " ")
    return (f"A person is {ctx} on {day}, in the {when}. Their average heart rate is {hr} their "
            f"personal resting baseline. Recorded activity hint: '{sig['activity']}'. Location: "
            f"{place}. Ambient thermal context: {wx}. Choose the single best per-episode situation "
            f"label and, if this is a workout, its workout type.")


# --------------------------------------------------------------------------- #
# Gemini (structured, temp=0, disk-cached) + deterministic fallback
# --------------------------------------------------------------------------- #
def _load_gemini_key() -> Optional[str]:
    try:
        from dotenv import load_dotenv
        for p in (os.path.join(_REPO_ROOT, ".env"),
                  os.path.join(_REPO_ROOT, "notebooks", "enrichment_experiment", ".env")):
            if os.path.exists(p):
                load_dotenv(p, override=False)
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


_SYSTEM = ("You label short wearable context windows using a fixed controlled vocabulary. "
           "Return ONLY the activity_context enum and (for workouts) the workout_type enum. "
           "Never judge anomalies or health. Base the label only on the described fingerprint.")


def _cache_path(desc: str) -> str:
    h = hashlib.sha256((GEMINI_MODEL + "\n" + _SYSTEM + "\n" + desc).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


def classify_llm(desc: str, key: str) -> Optional[EnrichedContext]:
    """Gemini structured output, cached on disk. None on any failure -> caller uses fallback."""
    cp = _cache_path(desc)
    if os.path.exists(cp):
        try:
            return EnrichedContext(**json.load(open(cp)))
        except Exception:
            pass
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        cfg = types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json",
            response_schema=EnrichedContext, system_instruction=_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_budget=0))
        r = client.models.generate_content(model=GEMINI_MODEL, contents=desc, config=cfg)
        out: EnrichedContext = r.parsed
        os.makedirs(CACHE_DIR, exist_ok=True)
        json.dump(out.model_dump(), open(cp, "w"))
        return out
    except Exception:
        return None


def classify_fallback(sig: dict) -> EnrichedContext:
    """Deterministic controlled-vocab labelling used when no LLM is available (or it fails)."""
    wt = _ACT_TO_WORKOUT_TYPE.get(sig["activity"], "unknown") if sig["is_workout"] else "unknown"
    if sig["is_workout"]:
        ac = "active_workout"
    elif sig["hr_band"] == "high":
        ac = "light_activity"
    elif sig["hour_bucket"] == "night":
        ac = "overnight_rest"
    elif sig["hour_bucket"] == "evening":
        ac = "evening_rest"
    elif sig["hr_band"] == "elevated":
        ac = "light_activity"
    else:
        ac = "daytime_rest"
    return EnrichedContext(activity_context=ac, workout_type=wt)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class EnrichResult:
    episodes: pd.DataFrame
    meta: dict


def enrich_signatures(sig_df: pd.DataFrame, use_llm: bool, key: Optional[str]) -> pd.DataFrame:
    """One row per distinct signature -> enriched labels + source. Cached/deterministic."""
    out = []
    for _, sig in sig_df.iterrows():
        s = sig.to_dict()
        res, source = None, "fallback"
        if use_llm and key:
            res = classify_llm(describe_signature(s), key)
            if res is not None:
                source = "llm"
        if res is None:
            res = classify_fallback(s)
        out.append({**s, "activity_context": res.activity_context,
                    "workout_type_ctx": res.workout_type, "enrich_source": source})
    return pd.DataFrame(out)


def apply_enrichment(ep: pd.DataFrame, enriched_sigs: pd.DataFrame) -> pd.DataFrame:
    """Broadcast the per-signature labels back onto every episode + attach location_semantic."""
    ep = ep.merge(enriched_sigs[SIG_COLS + ["activity_context", "workout_type_ctx", "enrich_source"]],
                  on=SIG_COLS, how="left")
    ep["location_semantic"] = ep["location_place"].where(ep["location_place"].notna(), "unknown")
    return ep


def run(offline: bool = False) -> EnrichResult:
    """Full Stage A: load episodes -> band -> dedup signatures -> enrich -> broadcast -> save."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ep = pd.read_parquet(EPISODES)
    bands = personal_hr_bands(ep)
    ep = add_signature_columns(ep, bands)

    sig_df = ep[SIG_COLS].drop_duplicates().reset_index(drop=True)
    key = None if offline else _load_gemini_key()
    use_llm = (not offline) and (key is not None)
    enriched_sigs = enrich_signatures(sig_df, use_llm=use_llm, key=key)

    ep_out = apply_enrichment(ep, enriched_sigs)

    src_counts = ep_out["enrich_source"].value_counts().to_dict()
    meta = {
        "vocab_source": _VOCAB_SOURCE,
        "n_episodes": int(len(ep_out)),
        "n_signatures": int(len(sig_df)),
        "used_llm": bool(use_llm),
        "enrich_source_episodes": {k: int(v) for k, v in src_counts.items()},
        "hr_bands": bands,
        "activity_context_dist": {k: int(v) for k, v in
                                  ep_out["activity_context"].value_counts().items()},
        "weather_ctx_dist": {k: int(v) for k, v in ep_out["weather_ctx"].value_counts().items()},
        "location_semantic_known": int((ep_out["location_semantic"] != "unknown").sum()),
        "sample_signature_labels": enriched_sigs.head(12).to_dict(orient="records"),
    }
    ep_out.to_parquet(ENRICHED_OUT, index=False)
    json.dump(meta, open(META_OUT, "w"), indent=2)
    return EnrichResult(episodes=ep_out, meta=meta)
