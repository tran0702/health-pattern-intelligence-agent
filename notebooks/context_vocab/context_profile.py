"""
context_profile.py — Task 1 (Asara): the DEFINED context profile (the "spine").

Asara gave a context profile to seed vocabulary generation:
  weather, geographic location, workout type, workout duration, workout location,
  sleep, heart health, medical conditions (current & past), occupation, demographic.
Asara's "demographic" is split here into two dimensions — `age_band` and `sex` — because
the physiology tables index them separately (Tanaka needs age; the resting-HR chart needs
age x sex), so a fused token like "adult_female_middle" is unusable. (PROJECT_STATUS 4-(3))

Design decision — profile "defined upfront" vs "LLM-derived from the dataset" -> HYBRID:
  * The SPINE below is DEFINED UPFRONT. It is the small set of physiology-grounded
    dimensions that apply to ANY wearable/health dataset, so it stays STABLE ->
    reproducibility (a fixed Pydantic `Literal` target), cross-dataset comparability,
    and it matches Asara's own rule "context is external meta-information, defined &
    ingested; the pipeline stays generic" (the diagram's 'Context Definitions' input).
  * The VALUES for each dimension are AI-GENERATED (see vocab_generator.py) — that is
    Asara Item 1 ("use AI to generate a rich vocabulary, do not hand-author it").
  * For a NEW dataset, the global profiler (Task 2) may PROPOSE extra dimensions the
    data surfaces (e.g. 'stroke_type' for a swim dataset); `extend_profile` appends
    them. So the profile is defined-but-extensible — general (Item 2 "any dataset")
    without losing the stable, reproducible core.

This module holds ONLY the spec (dimension names + how the LLM should populate them);
it emits no values itself.

ISOLATED track: does not import or mutate the 01-04 / 03b pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileDimension:
    name: str
    level: str        # "subject" (cohort descriptor) | "episode" (per-window / dynamic)
    kind: str         # "categorical" | "numeric_bucketed"
    description: str  # rich meaning handed to the LLM so it can generate the vocab
    source: str       # how the value is obtained before / at pipeline time


# The DEFINED spine — Asara's 10-dimension context profile. Order groups the
# subject-level (cohort descriptor) dimensions first, then the episode-level ones.
CONTEXT_PROFILE: list[ProfileDimension] = [
    ProfileDimension(
        "age_band", "subject", "categorical",
        "Coarse age bracket used to index physiological norms (max-HR via Tanaka, and "
        "the age-related decline of resting HR / HRV). Use EXACTLY these value tokens "
        "(and no others): under_18, 18_29, 30_39, 40_49, 50_59, 60_plus, unknown. Name "
        "age brackets ONLY — sex is a separate dimension; do NOT fold sex into a value. "
        "Put the rich content in each value's aliases and physio_note.",
        "predicted_from_HR (age via Tanaka HRmax) OR user_provided"),
    ProfileDimension(
        "sex", "subject", "categorical",
        "Biological sex used to index resting-HR / HRV reference tables, which differ "
        "for male vs female. Use EXACTLY these value tokens (and no others): male, "
        "female, unknown. Name sex categories ONLY — do NOT fold age in; age is its own "
        "dimension. Put the rich content in each value's aliases and physio_note.",
        "user_provided (optional)"),
    ProfileDimension(
        "heart_health", "subject", "categorical",
        "Cardiovascular fitness / cardiac-risk standing inferred from resting HR, HR "
        "recovery and HRV — e.g. athletic, healthy, average, at-risk. NOT a diagnosis; "
        "a coarse normality tier.",
        "predicted_from_HR (resting-HR percentile, HR recovery)"),
    ProfileDimension(
        "medical_conditions", "subject", "categorical",
        "Cardio-metabolic conditions that shift expected HR/HRV, each markable as "
        "CURRENT or PAST (e.g. hypertension, arrhythmia, diabetes, pregnancy, cardiac "
        "history, thyroid). Include a 'none' value.",
        "user_provided / clinical"),
    ProfileDimension(
        "occupation", "subject", "categorical",
        "Daily-activity archetype of the person's work that shapes baseline load — e.g. "
        "sedentary office worker (9-5), shift worker, manual labour, professional "
        "athlete, healthcare, student, retired.",
        "user_provided (hinted from 9-5 activity/location rhythm)"),
    ProfileDimension(
        "geographic_location", "subject", "categorical",
        "Home geographic / climate context (e.g. temperate, hot-arid, hot-humid, cold, "
        "high-altitude) that sets the ambient physiological baseline.",
        "derived_from_location (geocode) OR user_provided"),
    ProfileDimension(
        "weather", "episode", "categorical",
        "Ambient weather condition during a window that modulates HR — e.g. hot, cold, "
        "humid, mild/comfortable, heatwave. Thermal-comfort oriented.",
        "derived_from_signal (weather API by time + location)"),
    ProfileDimension(
        "workout_type", "episode", "categorical",
        "Kind of physical activity in a window — e.g. run, walk, cycle, swim, row, "
        "strength, yoga, HIIT, hike. Rich but physiologically distinct types.",
        "derived_from_signal (workout metadata + motion)"),
    ProfileDimension(
        "workout_duration", "episode", "numeric_bucketed",
        "Duration of a workout as named buckets with minute ranges — e.g. very short, "
        "short, moderate, long, endurance. Each value's physio_note must carry its "
        "approximate minute range.",
        "derived_from_signal (workout start/end)"),
    ProfileDimension(
        "workout_location", "episode", "categorical",
        "Where a workout happens — e.g. gym, home, outdoor road/trail, park, pool, open "
        "water, studio. Distinct from the home/work location_type.",
        "derived_from_signal (GPS + reverse-geocode)"),
    ProfileDimension(
        "sleep", "subject", "categorical",
        "Habitual sleep pattern inferred from overnight HR coverage and dip — e.g. "
        "restful, adequate, short, fragmented, irregular. Coarse tier, not clinical "
        "staging.",
        "derived_from_signal (overnight 0-5h HR)"),
]

PROFILE_BY_NAME = {d.name: d for d in CONTEXT_PROFILE}


def extend_profile(base: list[ProfileDimension],
                   extra: list[ProfileDimension]) -> list[ProfileDimension]:
    """Return a new profile = defined spine + LLM-proposed dataset-specific dims.

    Used by Task 2's global profiler to add dimensions a novel dataset surfaces.
    De-dups by name (the spine always wins) so the stable core is never overwritten.
    """
    seen = {d.name for d in base}
    return list(base) + [d for d in extra if d.name not in seen]
