"""
context_library.py — Task 1: the Context Library.

Realises the Asara direction (2026-07): "Context (location, health conditions,
goals) is EXTERNAL meta-information, defined outside the pipeline and ingested.
The pipeline stays generic and never decides what the context is."

This module is the single source of truth for:
  * the controlled vocabulary of every context attribute (Pydantic Literals),
  * `SubjectContext` — the per-person cohort descriptor that indexes the global
    baseline (approach (a): fed to the LLM to establish expected-normal ranges),
  * `EpisodeContext` — the per-window dynamic context that selects WHICH part of
    the baseline applies (rest vs vigorous, etc.),
  * `CONTEXT_REGISTRY` — machine-readable metadata (source / fallback / how it is
    consumed) so the design is documented in code, not only prose.

Robustness to missing data is built in: every field defaults to the "unknown"
sentinel. `availability()` reports which fields are actually known, and the
baseline layer widens its ranges when a field is unknown instead of crashing.

ISOLATED track: reads/writes only under data/context_baseline/ and
results/context_baseline/. Does NOT import or mutate the 01-04 / 03b pipeline
or the enrichment_experiment track.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Controlled vocabulary — the ONLY values any provider may emit.
# 'unknown' is the missing-data sentinel everywhere (graceful degradation).
#
# One-way dependency (Task 1, PROJECT_STATUS 4-(3)): the dimensions below marked
# "from vocab" are sourced from the FROZEN, committed AI-generated vocabulary
# (context_vocab/generated_vocab.py) so those tokens have a single source of truth —
# this is the whole point of Task 1. It imports only the emitted artifact, never the
# vocab track's runtime, and falls back to identical local Literals if the artifact is
# not importable, so the library always loads.
# --------------------------------------------------------------------------- #
import os as _os
import sys as _sys

_VOCAB_DIR = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                           "context_vocab")
if _VOCAB_DIR not in _sys.path:
    _sys.path.insert(0, _VOCAB_DIR)
try:
    from generated_vocab import (AgeBandVocab, SexVocab, OccupationVocab,       # noqa: E402
                                 HeartHealthVocab, SleepVocab)
    _VOCAB_WIRED = True
except Exception:                                    # artifact missing -> local fallback
    _VOCAB_WIRED = False

# from vocab (age_band + sex tokens are pinned to match these exactly)
Sex = SexVocab if _VOCAB_WIRED else Literal["male", "female", "unknown"]
AgeBand = AgeBandVocab if _VOCAB_WIRED else Literal[
    "under_18", "18_29", "30_39", "40_49", "50_59", "60_plus", "unknown"]
Occupation = OccupationVocab if _VOCAB_WIRED else Literal[
    "unknown", "sedentary_office", "manual_labor", "shift_worker",
    "professional_athlete", "healthcare_professional", "student", "retired"]
HeartHealth = HeartHealthVocab if _VOCAB_WIRED else Literal[
    "unknown", "elite_athletic", "healthy_active", "average_sedentary", "at_risk"]
Sleep = SleepVocab if _VOCAB_WIRED else Literal[
    "unknown", "restful", "adequate", "short", "fragmented", "irregular"]

# kept local — the anomaly pipeline's baseline layer depends on these exact tokens.
FitnessLevel = Literal["sedentary", "recreational", "trained", "athlete", "unknown"]
Goal = Literal["general_health", "weight_loss", "endurance", "strength",
               "stress_management", "unknown"]
Climate = Literal["temperate", "hot_arid", "hot_humid", "cold", "unknown"]
HealthCondition = Literal["none", "hypertension", "arrhythmia", "diabetes",
                          "pregnancy", "cardiac_history", "thyroid", "unknown"]

# Dynamic (episode-level) context.
LocationType = Literal["home", "work", "gym", "beach", "outdoors", "transit", "unknown"]
Activity = Literal["sleep", "rest", "walk", "run", "cycle", "strength", "other", "unknown"]

_MISSING = "unknown"


# --------------------------------------------------------------------------- #
# Runtime context objects
# --------------------------------------------------------------------------- #
class SubjectContext(BaseModel):
    """External meta-information about the PERSON — i.e. the cohort descriptor.

    This is what defines the "specific cohort" in the Asara design: it is passed
    to the LLM (approach (a)) to establish the group's global baseline of
    normality. Every field is optional via the 'unknown' sentinel, so a subject
    with only partial metadata still yields a (wider) baseline.
    """
    age_band: AgeBand = _MISSING
    sex: Sex = _MISSING
    age_years: float | None = None          # exact age when known; Tanaka HRmax wants a
                                            # real number, not the age_band midpoint
    fitness_level: FitnessLevel = _MISSING
    heart_health: HeartHealth = _MISSING    # from vocab: coarse cardiac tier (≠ fitness)
    health_conditions: list[HealthCondition] = Field(default_factory=lambda: [_MISSING])
    occupation: Occupation = _MISSING       # from vocab: daily-load archetype
    sleep: Sleep = _MISSING                 # from vocab: habitual sleep-quality tier
    goal: Goal = _MISSING
    home_climate: Climate = _MISSING

    def availability(self) -> dict[str, bool]:
        """Which cohort fields are actually known (not the 'unknown' sentinel).
        age_years is excluded — it is a precision refinement of age_band, not a
        separate axis, so counting it would double-count age."""
        hc = [c for c in self.health_conditions if c != _MISSING]
        return {
            "age_band": self.age_band != _MISSING,
            "sex": self.sex != _MISSING,
            "fitness_level": self.fitness_level != _MISSING,
            "heart_health": self.heart_health != _MISSING,
            "health_conditions": len(hc) > 0,
            "occupation": self.occupation != _MISSING,
            "sleep": self.sleep != _MISSING,
            "goal": self.goal != _MISSING,
            "home_climate": self.home_climate != _MISSING,
        }

    def coverage(self) -> float:
        """Fraction of cohort fields that are known — a crude context confidence."""
        a = self.availability()
        return sum(a.values()) / len(a)


class EpisodeContext(BaseModel):
    """Per-window dynamic context. Selects WHICH sub-range of the baseline
    applies (e.g. a 'run' window is judged against the vigorous-activity range,
    a 00:00-05:00 rest window against the sleep range)."""
    activity: Activity = _MISSING
    location_type: LocationType = _MISSING
    hour_of_day: int | None = None          # 0-23 local; None if unknown
    weather_temp_c: float | None = None
    is_workout: bool = False


# --------------------------------------------------------------------------- #
# Registry — self-documenting metadata for every context attribute.
# `source`     : how the value is obtained before the pipeline runs.
# `consumed_as`: how the generic pipeline uses it (never as a hard rule / target).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContextField:
    name: str
    level: str            # "subject" (cohort descriptor) | "episode" (dynamic)
    vocab: tuple          # allowed values
    source: str
    fallback: str
    consumed_as: str
    note: str = ""


CONTEXT_REGISTRY: list[ContextField] = [
    ContextField(
        "age_band", "subject", get_args(AgeBand),
        source="predicted_from_HR (Tanaka max-HR / resting-HR & HRV decline) OR user_provided",
        fallback=_MISSING, consumed_as="baseline_index",
        note="Primary axis of the global baseline; drives max-HR and HRV expectations."),
    ContextField(
        "sex", "subject", get_args(Sex),
        source="user_provided (optional)", fallback=_MISSING, consumed_as="baseline_index",
        note="Minor modulation of resting-HR/HRV expectations."),
    ContextField(
        "fitness_level", "subject", get_args(FitnessLevel),
        source="predicted_from_exertion (HRR + workout volume) OR user_provided",
        fallback=_MISSING, consumed_as="baseline_index",
        note="Shifts resting-HR band (athletic bradycardia) and HR-reserve intensities. "
             "Exertion-based, NOT resting-HR (circularity fix, 4-(1))."),
    ContextField(
        "heart_health", "subject", get_args(HeartHealth),
        source="predicted_from_HR (resting-HR, HR-recovery, HRV) OR user_provided",
        fallback=_MISSING, consumed_as="baseline_index",
        note="From AI-generated vocab: coarse cardiac-fitness tier; distinct from fitness_level."),
    ContextField(
        "health_conditions", "subject", get_args(HealthCondition),
        source="user_provided / clinical", fallback=_MISSING, consumed_as="baseline_index",
        note="Widens or shifts ranges (e.g. pregnancy raises resting HR); adds caveats."),
    ContextField(
        "occupation", "subject", get_args(Occupation),
        source="user_provided (hinted from 9-5 work/rest rhythm)", fallback=_MISSING,
        consumed_as="interpretation",
        note="From AI-generated vocab: daily-load archetype; colours what an elevated "
             "daytime HR means. No data predictor yet (Task 2)."),
    ContextField(
        "sleep", "subject", get_args(Sleep),
        source="derived_from_signal (overnight 0-5h HR dip/coverage) OR user_provided",
        fallback=_MISSING, consumed_as="interpretation",
        note="From AI-generated vocab: habitual sleep-quality tier; context for morning-HR "
             "elevation. No data predictor yet."),
    ContextField(
        "goal", "subject", get_args(Goal),
        source="user_provided", fallback=_MISSING, consumed_as="interpretation",
        note="Colours the 'translate what the anomaly means' step, not the ranges."),
    ContextField(
        "home_climate", "subject", get_args(Climate),
        source="derived_from_location (geocode) OR user_provided", fallback=_MISSING,
        consumed_as="interpretation", note="Heat context for elevated-HR interpretation."),
    ContextField(
        "activity", "episode", get_args(Activity),
        source="derived_from_signal (workout + motion)", fallback=_MISSING,
        consumed_as="baseline_subrange_selector",
        note="Picks resting vs light vs vigorous expected range for the window."),
    ContextField(
        "location_type", "episode", get_args(LocationType),
        source="derived_from_signal (GPS + reverse-geocode, see enrichment C2)",
        fallback=_MISSING, consumed_as="interpretation",
        note="home/gym/travel context for what a deviation means."),
]


def registry_table() -> str:
    """Pretty text table of the registry (for docs / the demo)."""
    rows = [("field", "level", "source", "fallback", "consumed_as")]
    rows += [(f.name, f.level, f.source.split(" OR ")[0][:38], f.fallback, f.consumed_as)
             for f in CONTEXT_REGISTRY]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    line = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))
    out = [line(rows[0]), "  ".join("-" * wi for wi in w)]
    out += [line(r) for r in rows[1:]]
    return "\n".join(out)
