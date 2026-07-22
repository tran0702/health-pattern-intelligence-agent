"""
global_context.py — Task 2 (Asara): the GLOBAL half of two-level context.

The individual layer (context_providers.build_subject_context) answers "who is this
PERSON". This module answers the level above it: given ANY dataset, "what is this
DATASET about" — its domain (e.g. clinical cardiac cohort vs consumer wearable) and
population descriptor — with NO human intervention and WITHOUT sending raw patient
rows to the LLM (only column names + dtypes + AGGREGATE statistics).

Three stages (PROJECT_STATUS 4-(2)):
  1. infer_column_roles  — schema-agnostic: map arbitrary columns to standard roles
     (heart_rate, timestamp, subject_id, workout_type, diagnosis, age, sex...) from
     name + dtype + summary stats. LLM (structured, cached) + regex/dtype fallback.
  2. dataset_fingerprint — pure code: dataset-level descriptive statistics computed as
     BETWEEN-subject distributions (not pooled) so the global signal is separable from
     the individual one; PLUS intra-subject longitudinal stability, and a structured-
     missingness assessment that flags heavy night gaps.
  3. classify_global_context — LLM (structured, cached) + rule fallback: the macro
     label {dataset_domain, population_descriptor, dominant_activities, evidence,
     confidence}. Becomes the PRIOR the individual layer falls back on (step 6).

Guardrails: temperature=0 + cache + deterministic fallback (runs offline, no key);
never sends data rows to the LLM (only column-level metadata + a low-cardinality
column's DISTINCT label set, which is dataset metadata, not a person's record); a
heavy-missingness flag forces the classifier to LOWER confidence on sleep/circadian
context rather than invent it. dataset_domain is free-form + light normalization so a
novel dataset is never forced into a frozen enum; dominant_activities ARE normalized
to the Task-1 workout_type vocab. ISOLATED track — reads a passed df (or the processed
Apple-Health frames) and writes only results/context_baseline/global_context.json.

NOT medical advice — dataset-level metadata for anomaly-detection context.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
DATA_DIR = REPO_ROOT / "data" / "context_baseline"
CACHE_DIR = DATA_DIR / "global_context_cache"
RESULTS_DIR = REPO_ROOT / "results" / "context_baseline"
GEMINI_MODEL = "gemini-3.1-flash-lite"

sys.path.insert(0, str(_THIS.parent))
# the frozen Task-1 vocab lives in the sibling track; import it for workout_type norm.
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "context_vocab"))
import vocab_generator as vg                                    # noqa: E402

NIGHT_HOURS = range(0, 6)          # 00:00-05:59 local — the circadian trough
DAY_HOURS = range(8, 22)           # 08:00-21:59 local
# heavy-missing when night is sampled less than this fraction of how often day is
# (scale-free — captures the "night vs day" skew §6 flags at 5.7x, not an absolute rate).
NIGHT_HEAVY_RATIO = 0.5


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Stage 1 schema — column roles. Role is a FIXED internal taxonomy (Literal is
# right here); the OPEN, free-form field is dataset_domain in GlobalContext below.
# --------------------------------------------------------------------------- #
Role = Literal["heart_rate", "timestamp", "subject_id", "workout_type",
               "workout_duration", "diagnosis", "medication", "age", "sex", "other"]


class ColumnRole(BaseModel):
    column: str
    role: Role
    confidence: float = 0.5


class ColumnRoles(BaseModel):
    roles: list[ColumnRole]
    source: str = "unset"                     # "llm" | "regex_default"

    def columns_for(self, role: str) -> list[str]:
        return [r.column for r in self.roles if r.role == role]

    def first(self, role: str) -> str | None:
        cols = self.columns_for(role)
        return cols[0] if cols else None


# --------------------------------------------------------------------------- #
# Stage 2 schema — dataset fingerprint (pure code, no LLM).
# --------------------------------------------------------------------------- #
class SubjectVariance(BaseModel):
    n_subjects: int
    inter_subject_resting_median: float | None = None   # median across subjects of
    inter_subject_resting_iqr: float | None = None      # each subject's resting HR (p10)
    intra_subject_daily_std_median: float | None = None  # median across subjects of the
    #  per-subject std of daily-baseline HR — high => individuals swing day-to-day
    interpretation: str = ""


class MissingnessInfo(BaseModel):
    night_coverage: float                     # fraction of covered days with a 00-05h reading
    day_coverage: float
    night_to_day_ratio: float
    night_missing_heavy: bool                 # night_coverage < NIGHT_HEAVY_THRESHOLD
    longest_gap_days: float


class DatasetFingerprint(BaseModel):
    n_records: int
    n_subjects: int
    span_days: float
    records_per_day: float
    hr_median: float | None = None            # POOLED HR percentiles (labelled pooled;
    hr_p10: float | None = None               # the between-subject view is in
    hr_p90: float | None = None               # subject_variance, which is what separates
    hr_p99: float | None = None               # global from individual)
    workout_fraction: float = 0.0
    workout_type_hist: dict[str, int] = {}
    has_diagnosis: bool = False
    has_medication: bool = False
    subject_variance: SubjectVariance
    missingness: MissingnessInfo
    notes: str = ""


# --------------------------------------------------------------------------- #
# Stage 3 schema — the macro label. dataset_domain / population_descriptor are
# FREE-FORM (schema-agnostic: a novel dataset is never forced into a frozen enum);
# dominant_activities are normalized to the Task-1 workout_type vocab.
# --------------------------------------------------------------------------- #
class GlobalContext(BaseModel):
    dataset_domain: str                       # free-form, lightly normalized
    population_descriptor: str                # free-form
    dominant_activities: list[str] = []       # normalized -> workout_type vocab
    evidence: str = ""
    confidence: float = 0.0
    source: str = "unset"                     # "llm" | "rule_default"


# --------------------------------------------------------------------------- #
# Column summaries — metadata only, NEVER row-aligned data.
# --------------------------------------------------------------------------- #
def summarize_columns(df: pd.DataFrame, max_categories: int = 20) -> list[dict]:
    """Per-column {name, dtype, stats} with NO data rows. For a low-cardinality
    object column the DISTINCT label set is included (that is a controlled vocabulary
    / dataset metadata — e.g. the set of workout types or diagnosis labels — not any
    individual's record). High-cardinality / free-text columns send only cardinality."""
    out = []
    for c in df.columns:
        s = df[c]
        info: dict = {"name": str(c), "dtype": str(s.dtype),
                      "null_frac": round(float(s.isna().mean()), 3),
                      "n_unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_datetime64_any_dtype(s):
            sv = s.dropna()
            if len(sv):
                info.update(kind="datetime", min=str(sv.min()), max=str(sv.max()),
                            monotonic_increasing=bool(sv.is_monotonic_increasing))
        elif pd.api.types.is_numeric_dtype(s):
            sv = pd.to_numeric(s, errors="coerce").dropna()
            if len(sv):
                info.update(kind="numeric", min=round(float(sv.min()), 2),
                            max=round(float(sv.max()), 2), mean=round(float(sv.mean()), 2),
                            median=round(float(sv.median()), 2))
        else:
            info["kind"] = "categorical"
            if info["n_unique"] <= max_categories:
                info["distinct_values"] = sorted(map(str, s.dropna().unique().tolist()))
        out.append(info)
    return out


# --------------------------------------------------------------------------- #
# Stage 1 — infer_column_roles (LLM structured + regex/dtype fallback)
# --------------------------------------------------------------------------- #
_ROLE_PATTERNS = [
    ("heart_rate", r"heart.?rate|(^|_)hr(_|$)|bpm|pulse"),
    ("subject_id", r"subject|participant|patient|person|(^|_)user|(^|_)id(_|$)"),
    ("workout_type", r"workout|activity|exercise|session.?(type|kind|name)|sport"),
    ("workout_duration", r"duration|elapsed|minutes|(^|_)mins?(_|$)"),
    ("diagnosis", r"diagnos|condition|icd|disease|disorder"),
    ("medication", r"medicat|drug|(^|_)med(_|$)|prescription|(^|_)rx(_|$)"),
    ("age", r"(^|_)age(_|$)|years?_old"),
    ("sex", r"(^|_)sex(_|$)|gender"),
    ("timestamp", r"time|date|(^|_)ts(_|$)|datetime|timestamp"),
]


def _regex_roles(df: pd.DataFrame) -> ColumnRoles:
    """Name + dtype fallback. dtype wins for timestamp; a numeric column in the
    HR range is a heart_rate candidate when its NAME gives nothing away."""
    roles: list[ColumnRole] = []
    for c in df.columns:
        name = str(c).lower()
        s = df[c]
        role: str | None = None
        if pd.api.types.is_datetime64_any_dtype(s):
            role = "timestamp"
        if role is None:
            for r, pat in _ROLE_PATTERNS:
                if re.search(pat, name):
                    role = r
                    break
        if role is None and pd.api.types.is_numeric_dtype(s):
            sv = pd.to_numeric(s, errors="coerce").dropna()
            if len(sv) and 25 <= sv.min() and sv.max() <= 230 and 40 <= sv.mean() <= 140:
                role = "heart_rate"
        roles.append(ColumnRole(column=str(c), role=role or "other",
                                confidence=0.5 if role else 0.2))
    return ColumnRoles(roles=roles, source="regex_default")


def _roles_prompt(summaries: list[dict]) -> str:
    return (
        "You map the COLUMNS of a health/wearable dataset to standard roles. You are "
        "given each column's name, dtype and summary statistics — NO data rows. Assign "
        "every column exactly one role from this fixed set: heart_rate, timestamp, "
        "subject_id, workout_type, workout_duration, diagnosis, medication, age, sex, "
        "other. Decide from dtype and value ranges, not names alone: a heart-rate column "
        "holds values ~40-200 bpm; a timestamp increases monotonically; a subject_id has "
        "low-to-moderate cardinality of ids; workout_type is a small categorical of "
        "activity labels. If nothing fits, use 'other'. Give a 0-1 confidence.\n"
        f"Columns: {json.dumps(summaries)}"
    )


def infer_column_roles(df: pd.DataFrame, source: str = "auto") -> ColumnRoles:
    """source: 'auto' (LLM if key else regex) | 'llm' (force) | 'default' (regex)."""
    if source == "default":
        return _regex_roles(df)
    key = _load_key()
    if source == "llm" and not key:
        raise RuntimeError("source='llm' but no GEMINI_API_KEY found (.env).")
    if source == "auto" and not key:
        return _regex_roles(df)
    ensure_dirs()
    prompt = _roles_prompt(summarize_columns(df))
    cp = _cache_path(prompt)
    if cp.exists():
        cr = ColumnRoles(**json.loads(cp.read_text(encoding="utf-8")))
        cr.source = "llm"
        return cr
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key)
        cfg = types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json",
            response_schema=ColumnRoles,
            thinking_config=types.ThinkingConfig(thinking_budget=0))
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg)
        cr: ColumnRoles = r.parsed
        cr.source = "llm"
        cp.write_text(cr.model_dump_json(), encoding="utf-8")
        return cr
    except Exception:                                # any failure -> deterministic regex
        return _regex_roles(df)


# --------------------------------------------------------------------------- #
# Stage 2 — dataset_fingerprint (pure code)
# --------------------------------------------------------------------------- #
def _clean_activity(x: str) -> str:
    return re.sub(r"^HKWorkoutActivityType", "", str(x))


def dataset_fingerprint(df: pd.DataFrame, roles: ColumnRoles) -> DatasetFingerprint:
    """Dataset-level descriptive statistics. Between-subject (inter_subject_*) and
    day-to-day (intra_subject_*) variability are computed separately so the classifier
    can tell a MIXED population (many different people) from an UNSTABLE one (few people
    with big day-to-day swings). Missingness is graded so the classifier can refuse to
    over-read the night when it is barely sampled."""
    ts_c = roles.first("timestamp")
    hr_c = roles.first("heart_rate")
    subj_c = roles.first("subject_id")
    wt_c = roles.first("workout_type")

    d = df.copy()
    if ts_c is not None:
        d[ts_c] = pd.to_datetime(d[ts_c], errors="coerce", utc=False)
    hr = pd.to_numeric(d[hr_c], errors="coerce") if hr_c else pd.Series(dtype=float)

    n_records = int(len(d))
    n_subjects = int(d[subj_c].nunique()) if subj_c else 1

    span_days = 0.0
    if ts_c is not None and d[ts_c].notna().any():
        span_days = (d[ts_c].max() - d[ts_c].min()).total_seconds() / 86400
    records_per_day = round(n_records / span_days, 1) if span_days > 0 else float(n_records)

    hr_stats = {"hr_median": None, "hr_p10": None, "hr_p90": None, "hr_p99": None}
    if len(hr.dropna()):
        v = hr.dropna()
        hr_stats = {"hr_median": round(float(v.median()), 1),
                    "hr_p10": round(float(np.percentile(v, 10)), 1),
                    "hr_p90": round(float(np.percentile(v, 90)), 1),
                    "hr_p99": round(float(np.percentile(v, 99)), 1)}

    # ---- between- vs within-subject variability -------------------------------
    subj_key = subj_c if subj_c else "_one_subject"
    if not subj_c:
        d[subj_key] = "s0"
    resting_per_subject, daily_std_per_subject = [], []
    for _, g in d.groupby(subj_key):
        gv = pd.to_numeric(g[hr_c], errors="coerce").dropna() if hr_c else pd.Series(dtype=float)
        if len(gv) >= 10:
            resting_per_subject.append(float(np.percentile(gv, 10)))     # resting proxy
        if hr_c and ts_c and g[ts_c].notna().any():
            gg = g.dropna(subset=[ts_c]).copy()
            gg["_date"] = gg[ts_c].dt.date
            daily = gg.groupby("_date")[hr_c].apply(
                lambda s: float(np.percentile(pd.to_numeric(s, errors="coerce").dropna(), 10))
                if pd.to_numeric(s, errors="coerce").dropna().size else np.nan).dropna()
            if len(daily) >= 3:
                daily_std_per_subject.append(float(daily.std(ddof=0)))

    def _med(a):
        return round(float(np.median(a)), 1) if a else None

    inter_iqr = (round(float(np.percentile(resting_per_subject, 75)
                             - np.percentile(resting_per_subject, 25)), 1)
                 if len(resting_per_subject) >= 4 else None)
    sv = SubjectVariance(
        n_subjects=n_subjects,
        inter_subject_resting_median=_med(resting_per_subject),
        inter_subject_resting_iqr=inter_iqr,
        intra_subject_daily_std_median=_med(daily_std_per_subject),
        interpretation=("single subject — inter-subject spread is not meaningful; "
                        "day-to-day std reflects this person's own volatility"
                        if n_subjects == 1 else
                        "multiple subjects — compare inter-subject spread (population mix) "
                        "against intra-subject day-to-day std (individual volatility)"))

    # ---- structured missingness (night gap) -----------------------------------
    night_cov = day_cov = 0.0
    longest_gap = 0.0
    if ts_c is not None and d[ts_c].notna().any():
        t = d[ts_c].dropna().sort_values()
        dd = t.dt.normalize()
        n_days = max(dd.nunique(), 1)
        hours = t.dt.hour
        night_days = dd[hours.isin(list(NIGHT_HOURS))].nunique()
        day_days = dd[hours.isin(list(DAY_HOURS))].nunique()
        night_cov = round(night_days / n_days, 3)
        day_cov = round(day_days / n_days, 3)
        if len(t) >= 2:
            longest_gap = round(float(t.diff().dropna().max().total_seconds() / 86400), 2)
    ratio = round(night_cov / day_cov, 3) if day_cov else 0.0
    missing = MissingnessInfo(
        night_coverage=night_cov, day_coverage=day_cov, night_to_day_ratio=ratio,
        night_missing_heavy=bool(day_cov > 0 and ratio < NIGHT_HEAVY_RATIO),
        longest_gap_days=longest_gap)

    # ---- activity mix ----------------------------------------------------------
    workout_fraction, hist = 0.0, {}
    if wt_c is not None:
        wt = d[wt_c].dropna().map(_clean_activity)
        active = wt[~wt.str.lower().isin(["none", "unknown", "nan", ""])]
        workout_fraction = round(len(active) / max(n_records, 1), 3)
        hist = {k: int(v) for k, v in active.value_counts().head(12).items()}

    diag = roles.first("diagnosis")
    med = roles.first("medication")

    return DatasetFingerprint(
        n_records=n_records, n_subjects=n_subjects, span_days=round(span_days, 1),
        records_per_day=records_per_day, **hr_stats,
        workout_fraction=workout_fraction, workout_type_hist=hist,
        has_diagnosis=diag is not None, has_medication=med is not None,
        subject_variance=sv, missingness=missing,
        notes=f"roles source={roles.source}")


# --------------------------------------------------------------------------- #
# Stage 3 — classify_global_context (LLM structured + rule fallback)
# --------------------------------------------------------------------------- #
_DOMAIN_TAXONOMY = {
    "cardiac": "clinical_cardiac", "clinical": "clinical_cohort", "patient": "clinical_cohort",
    "hospital": "clinical_cohort", "disease": "clinical_cohort",
    "athlete": "athletic_performance", "sport": "athletic_performance",
    "endurance": "athletic_performance", "fitness": "consumer_fitness",
    "wearable": "consumer_wearable", "consumer": "consumer_wearable",
    "sleep": "sleep_study", "general": "general_population", "office": "general_population",
}


def _normalize_domain(s: str) -> str:
    """Light, PASS-THROUGH normalization: snake-case and fold a few common synonyms,
    but keep any novel domain verbatim (schema-agnostic — never force a frozen enum)."""
    t = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")
    for k, v in _DOMAIN_TAXONOMY.items():
        if k in t:
            return v
    return t or "unknown"


# common Apple-Health workout types -> Task-1 workout_type vocab (the frozen vocab's
# substring normalizer misses e.g. "Running" vs alias "run"); vg.normalize is the
# fallback for anything not listed.
_ACTIVITY_MAP = {
    "walking": "low_impact_steady", "hiking": "low_impact_steady",
    "running": "steady_state_cardio", "cycling": "steady_state_cardio",
    "rowing": "steady_state_cardio", "elliptical": "steady_state_cardio",
    "swimming": "aquatic_exercise", "yoga": "mind_body", "pilates": "mind_body",
    "strength": "resistance_training", "functionalstrengthtraining": "resistance_training",
    "hiit": "high_intensity_interval",
}


def _normalize_activities(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items or []:
        c = _clean_activity(x).lower()
        v = _ACTIVITY_MAP.get(c) or vg.normalize("workout_type", c)
        if v != "unknown" and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _rule_global_context(fp: DatasetFingerprint) -> GlobalContext:
    """Deterministic fallback used offline and when the LLM is unavailable."""
    sv, ms = fp.subject_variance, fp.missingness
    rest = sv.inter_subject_resting_median
    # workout_fraction is a share of READINGS (rest readings dominate), so it runs
    # small; ~0.03 already means frequent exercise. Thresholds are heuristic — the LLM
    # path is primary, this only has to be sane offline.
    high_ex, low_ex = fp.workout_fraction >= 0.03, fp.workout_fraction < 0.015
    if fp.has_diagnosis or fp.has_medication:
        domain = "clinical_cohort"
    elif high_ex and (rest is None or rest <= 60):
        domain = "athletic_performance"
    elif fp.n_subjects == 1:
        domain = "consumer_wearable"
    else:
        domain = "general_population"

    pop = []
    pop.append("single individual" if fp.n_subjects == 1 else f"{fp.n_subjects} subjects")
    if rest is not None:
        pop.append(f"resting HR ~{rest:.0f} bpm"
                   + (f" (IQR {sv.inter_subject_resting_iqr:.0f})" if sv.inter_subject_resting_iqr else ""))
    pop.append("frequent exercise" if high_ex
               else "low exercise volume" if low_ex else "moderate exercise")
    acts = _normalize_activities(list(fp.workout_type_hist))
    ev = [f"records/day~{fp.records_per_day:.0f}", f"span {fp.span_days:.0f}d"]
    if ms.night_missing_heavy:
        ev.append(f"night coverage {ms.night_coverage:.0%} (heavy gap) -> sleep context unreliable")
    conf = 0.45 + (0.15 if (fp.has_diagnosis or high_ex) else 0.0)
    return GlobalContext(dataset_domain=_normalize_domain(domain),
                         population_descriptor=", ".join(pop),
                         dominant_activities=acts, evidence="; ".join(ev),
                         confidence=round(conf, 2), source="rule_default")


def _classify_prompt(fp: DatasetFingerprint) -> str:
    return (
        "You label a wearable/health DATASET (not a person). From the fingerprint "
        "below infer: dataset_domain (a short free-form label, e.g. clinical_cardiac, "
        "athletic_performance, consumer_wearable, general_population — invent one if "
        "none fits), population_descriptor (one free-form phrase), dominant_activities "
        "(activity labels), evidence (cite the numbers), confidence (0-1). "
        "Use the BETWEEN-subject spread (inter_subject_resting_iqr) to judge how mixed "
        "the population is, and the intra_subject_daily_std to judge individual "
        "volatility. IMPORTANT: if missingness.night_missing_heavy is true the night is "
        "barely sampled — do NOT infer any sleep/circadian context, and lower confidence "
        "for anything nocturnal; base the label on daytime/workout evidence. Do not flag "
        "anomalies.\n"
        f"Fingerprint: {fp.model_dump_json()}"
    )


def classify_global_context(fp: DatasetFingerprint, source: str = "auto") -> GlobalContext:
    """source: 'auto' (LLM if key else rule) | 'llm' (force) | 'default' (rule)."""
    if source == "default":
        return _rule_global_context(fp)
    key = _load_key()
    if source == "llm" and not key:
        raise RuntimeError("source='llm' but no GEMINI_API_KEY found (.env).")
    if source == "auto" and not key:
        return _rule_global_context(fp)
    ensure_dirs()
    prompt = _classify_prompt(fp)
    cp = _cache_path(prompt)
    if cp.exists():
        gc = GlobalContext(**json.loads(cp.read_text(encoding="utf-8")))
    else:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key)
            cfg = types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json",
                response_schema=GlobalContext,
                thinking_config=types.ThinkingConfig(thinking_budget=0))
            r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg)
            gc = r.parsed
            cp.write_text(gc.model_dump_json(), encoding="utf-8")
        except Exception:
            return _rule_global_context(fp)
    # normalize the free-form label + fold activities onto the Task-1 vocab.
    gc.source = "llm"
    gc.dataset_domain = _normalize_domain(gc.dataset_domain)
    gc.dominant_activities = _normalize_activities(gc.dominant_activities)
    return gc


# --------------------------------------------------------------------------- #
# Orchestrator + Apple-Health frames -> one tidy table
# --------------------------------------------------------------------------- #
def apple_health_to_df(frames: dict) -> pd.DataFrame:
    """Fold the split Apple-Health frames (hr_raw + workouts) into ONE tidy table
    (timestamp, heart_rate, subject_id, workout_type) — the single-table contract the
    schema-agnostic path expects. Rows inside a workout interval get that workout type."""
    hr = frames["hr_raw"][["datetime", "value"]].rename(
        columns={"datetime": "timestamp", "value": "heart_rate"}).copy()
    hr["subject_id"] = "s0"
    hr["workout_type"] = "none"
    wk = frames.get("workouts")
    if wk is not None and {"start_time", "end_time", "type"} <= set(wk.columns):
        t = hr["timestamp"]
        for s, e, typ in zip(wk["start_time"], wk["end_time"], wk["type"]):
            if pd.isna(s) or pd.isna(e):
                continue
            hr.loc[(t >= s) & (t <= e), "workout_type"] = _clean_activity(typ)
    return hr


def run_global_context(df: pd.DataFrame | None = None, frames: dict | None = None,
                       source: str = "auto") -> dict:
    """End-to-end: roles -> fingerprint -> global context. Writes the bundle to
    results/context_baseline/global_context.json. Pass a tidy `df`, or `frames`
    (Apple-Health processed parquets); with neither, loads the real frames."""
    if df is None:
        if frames is None:
            from context_providers import load_frames
            frames = load_frames()
        df = apple_health_to_df(frames)
    roles = infer_column_roles(df, source=source)
    fp = dataset_fingerprint(df, roles)
    gc = classify_global_context(fp, source=source)
    bundle = {"roles": roles.model_dump(), "fingerprint": fp.model_dump(),
              "global_context": gc.model_dump()}
    ensure_dirs()
    (RESULTS_DIR / "global_context.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle


# --------------------------------------------------------------------------- #
# LLM key + cache plumbing (same contract as global_baseline.py)
# --------------------------------------------------------------------------- #
def _load_key() -> str | None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    for p in (_THIS.parent / ".env", REPO_ROOT / ".env",
              REPO_ROOT / "notebooks" / "enrichment_experiment" / ".env"):
        if p.exists():
            load_dotenv(p, override=False)
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def key_available() -> bool:
    return bool(_load_key())


def _cache_path(prompt: str) -> Path:
    h = hashlib.sha256((GEMINI_MODEL + "\n" + prompt).encode()).hexdigest()[:32]
    return CACHE_DIR / f"{h}.json"
