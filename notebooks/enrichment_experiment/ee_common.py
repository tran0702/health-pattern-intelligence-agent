"""
ee_common.py — shared helpers for the Context Enrichment Experiment (notebooks E1-E4).

ISOLATED TRACK. Reads only from  data/enrichment_experiment/ ,
writes only to               results/enrichment_experiment/ .
Does NOT import from / write to the health_trajectory pipeline (01-04 / 03b).

Source of truth for the schema + LABEL_MAP:
    notebooks/enrichment_experiment/context_enrichment_experiment_plan.md  (section 2 / 2.2)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Paths (anchored at repo root, independent of the notebook's cwd)
# --------------------------------------------------------------------------- #
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]                              # .../Apple Health Data
DATA_DIR = REPO_ROOT / "data" / "enrichment_experiment"
EXTRASENSORY_DIR = DATA_DIR / "extrasensory"
FEATURES_DIR = EXTRASENSORY_DIR / "features_and_labels"
ABS_LOCATION_DIR = EXTRASENSORY_DIR / "absolute_location"   # RQ1b: absolute lat/long (E5)
FOLDS_DIR = EXTRASENSORY_DIR / "cross_validation_partition" / "cv_5_folds"
LLM_CACHE_DIR = DATA_DIR / "llm_cache"
GEOCODE_CACHE_DIR = DATA_DIR / "geocode_cache"             # RQ1b: reverse-geocode cache (E5)
RESULTS_DIR = REPO_ROOT / "results" / "enrichment_experiment"


def ensure_output_dirs() -> None:
    """Create the (git-ignored) output dirs this track writes to."""
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GEOCODE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Gemini LLM access (E3). Key is loaded from the repo-root .env, never printed.
# --------------------------------------------------------------------------- #
GEMINI_MODEL = "gemini-3.1-flash-lite"   # verified stable id (2026-07-09)


def get_gemini_api_key() -> str:
    """Read GEMINI_API_KEY (fallback GOOGLE_API_KEY) from a .env.

    Searched (first hit wins): this experiment folder, then the repo root.
    """
    import os
    from dotenv import load_dotenv
    for env_path in (_THIS.parent / ".env", REPO_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Put GEMINI_API_KEY=... in a .env in "
            "notebooks/enrichment_experiment/ or the repo root (see .env.example). "
            "The key is never printed or logged."
        )
    return key


def gemini_key_available() -> bool:
    """True if a key can be loaded — lets notebooks skip LLM cells gracefully."""
    try:
        get_gemini_api_key()
        return True
    except RuntimeError:
        return False


def get_gemini_client():
    """A google-genai Client authed from .env (lazy import; SDK optional until E3)."""
    from google import genai
    return genai.Client(api_key=get_gemini_api_key())


# --------------------------------------------------------------------------- #
# Fixed context schema (plan section 2) — the LLM/ML may ONLY fill these fields.
# --------------------------------------------------------------------------- #
LocationT = Literal["home", "work", "gym", "beach", "restaurant",
                    "transit", "outdoors", "unknown"]
ActivityT = Literal["sitting", "standing", "walking", "running", "cycling",
                    "exercise", "sleeping", "eating", "unknown"]
CompanionT = Literal["alone", "with_friends", "with_family",
                     "with_coworkers", "unknown"]


class ContextLabel(BaseModel):
    """Semantic context of one sample. No is_anomaly / confidence fields."""
    location: LocationT
    activity: ActivityT
    companion: CompanionT


FIELDS = ("location", "activity", "companion")

# --------------------------------------------------------------------------- #
# LABEL_MAP (plan 2.2) — real ExtraSensory column names, priority = list order
# (first key = highest priority when several labels are 1 on the same sample).
# --------------------------------------------------------------------------- #
LOCATION_MAP: dict[str, list[str]] = {
    "gym":        ["label:AT_THE_GYM"],
    "beach":      ["label:LOC_beach"],
    "home":       ["label:LOC_home"],
    "work":       ["label:LOC_main_workplace", "label:AT_SCHOOL", "label:IN_CLASS",
                   "label:IN_A_MEETING", "label:LAB_WORK"],
    "restaurant": ["label:FIX_restaurant", "label:AT_A_BAR", "label:AT_A_PARTY"],
    "transit":    ["label:IN_A_CAR", "label:ON_A_BUS", "label:ELEVATOR",
                   "label:DRIVE_-_I_M_THE_DRIVER", "label:DRIVE_-_I_M_A_PASSENGER"],
    "outdoors":   ["label:OR_outside"],
    # label:OR_indoors -> secondary flag, NOT a concrete location (excluded on purpose)
}

ACTIVITY_MAP: dict[str, list[str]] = {
    "sleeping": ["label:SLEEPING"],
    "running":  ["label:FIX_running"],
    "cycling":  ["label:BICYCLING"],
    "exercise": ["label:OR_exercise", "label:STAIRS_-_GOING_UP", "label:STAIRS_-_GOING_DOWN"],
    "walking":  ["label:FIX_walking", "label:STROLLING"],
    "eating":   ["label:EATING"],
    "sitting":  ["label:SITTING"],
    "standing": ["label:OR_standing"],
    "lying":    ["label:LYING_DOWN"],
    # Fine-grained activities (COOKING/CLEANING/SHOPPING/WATCHING_TV/...) -> 'unknown' (plan).
}

COMPANION_MAP: dict[str, list[str]] = {
    "with_friends":   ["label:WITH_FRIENDS"],
    "with_coworkers": ["label:WITH_CO-WORKERS"],
    # else -> 'alone' (inferred, plan 2.2)
}

# 'lying' is a derivation bucket but is NOT in the fixed ContextLabel.activity schema.
# The plan lists LYING_DOWN in the map yet omits 'lying' from the schema, so we fold it
# to 'unknown' at the schema boundary. Flip this to 'lying' (and add it to ActivityT)
# only if the professor wants 'lying' as a first-class activity label.  <-- OPEN DECISION
ACTIVITY_TO_SCHEMA: dict[str, str] = {"lying": "unknown"}

_FIELD_MAP = {"location": LOCATION_MAP, "activity": ACTIVITY_MAP, "companion": COMPANION_MAP}
_FIELD_FALLBACK = {"location": "unknown", "activity": "unknown", "companion": "alone"}

# All label columns referenced as a scoring "source" for each field (used for the
# eligibility mask: a field is only scoreable if at least one source col is observed).
LOCATION_SRC_COLS = [c for cols in LOCATION_MAP.values() for c in cols]
ACTIVITY_SRC_COLS = [c for cols in ACTIVITY_MAP.values() for c in cols]
COMPANION_SRC_COLS = [c for cols in COMPANION_MAP.values() for c in cols]

# The 8 discrete local-time-of-day features (plan phase-(c): no timezone assumption).
TIME_OF_DAY_COLS = [
    "discrete:time_of_day:between0and6", "discrete:time_of_day:between3and9",
    "discrete:time_of_day:between6and12", "discrete:time_of_day:between9and15",
    "discrete:time_of_day:between12and18", "discrete:time_of_day:between15and21",
    "discrete:time_of_day:between18and24", "discrete:time_of_day:between21and3",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def uuid_files() -> dict[str, Path]:
    """Map UUID -> its features_labels csv.gz path (60 users)."""
    out = {}
    for p in sorted(FEATURES_DIR.glob("*.features_labels.csv.gz")):
        out[p.name.split(".")[0]] = p
    return out


def load_user(uuid: str) -> pd.DataFrame:
    """Load one user's samples; adds a 'uuid' column."""
    df = pd.read_csv(uuid_files()[uuid], compression="gzip")
    df.insert(0, "uuid", uuid)
    return df


def load_all(uuids: list[str] | None = None) -> pd.DataFrame:
    """Concatenate all (or the given) users into one frame with a 'uuid' column."""
    files = uuid_files()
    keys = uuids if uuids is not None else list(files.keys())
    frames = []
    for u in keys:
        d = pd.read_csv(files[u], compression="gzip")
        d.insert(0, "uuid", u)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Absolute location (RQ1b / E5) — separate download, joined by (uuid, timestamp).
# Format verified on disk: columns [timestamp, latitude, longitude], one file/user.
# --------------------------------------------------------------------------- #
def abs_location_files() -> dict[str, Path]:
    """Map UUID -> its *.absolute_locations.csv.gz path (60 users)."""
    out = {}
    for p in sorted(ABS_LOCATION_DIR.glob("*.absolute_locations.csv.gz")):
        out[p.name.split(".")[0]] = p
    return out


def load_absolute_location(uuid: str) -> pd.DataFrame:
    """One user's coords indexed by (uuid, timestamp); cols latitude/longitude."""
    df = pd.read_csv(abs_location_files()[uuid], compression="gzip")
    df["uuid"] = uuid
    return df.set_index(["uuid", "timestamp"])[["latitude", "longitude"]]


def load_all_absolute_location(uuids: list[str] | None = None) -> pd.DataFrame:
    """Concatenate all (or given) users' coords, indexed by (uuid, timestamp)."""
    files = abs_location_files()
    keys = uuids if uuids is not None else list(files.keys())
    frames = []
    for u in keys:
        d = pd.read_csv(files[u], compression="gzip")
        d["uuid"] = u
        frames.append(d.set_index(["uuid", "timestamp"])[["latitude", "longitude"]])
    return pd.concat(frames).sort_index()


def label_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("label:")]


def feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"uuid", "timestamp", "label_source"}
    return [c for c in df.columns if c not in drop and not c.startswith("label:")]


# --------------------------------------------------------------------------- #
# Cross-validation folds (official 5-fold, split by user)
# --------------------------------------------------------------------------- #
def _read_uuid_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def load_folds() -> dict[int, dict[str, list[str]]]:
    """
    Return {fold: {'train': [...uuids], 'test': [...uuids]}} combining the
    android + iphone partition files. Split is by USER (no leakage).
    """
    folds: dict[int, dict[str, list[str]]] = {}
    for k in range(5):
        split = {}
        for part in ("train", "test"):
            uu: list[str] = []
            for device in ("android", "iphone"):
                f = FOLDS_DIR / f"fold_{k}_{part}_{device}_uuids.txt"
                if f.exists():
                    uu.extend(_read_uuid_list(f))
            split[part] = sorted(set(uu))
        folds[k] = split
    return folds


def get_split(fold: int) -> tuple[list[str], list[str]]:
    """(train_uuids, test_uuids) for one fold."""
    s = load_folds()[fold]
    return s["train"], s["test"]


# --------------------------------------------------------------------------- #
# Gold-label derivation (plan 2 / 2.2): multi-label -> one label per field.
# --------------------------------------------------------------------------- #
def _derive_field(df: pd.DataFrame, field: str) -> pd.Series:
    """Highest-priority label whose source col == 1, else the field fallback."""
    mapping = _FIELD_MAP[field]
    out = pd.Series(_FIELD_FALLBACK[field], index=df.index, dtype=object)
    # apply low -> high priority so the highest-priority match wins (overwrites last)
    for value, cols in reversed(list(mapping.items())):
        present = [c for c in cols if c in df.columns]
        if not present:
            continue
        mask = (df[present] == 1).any(axis=1)
        out[mask] = value
    if field == "activity":
        out = out.replace(ACTIVITY_TO_SCHEMA)
    return out


def _eligible_mask(df: pd.DataFrame, src_cols: list[str]) -> pd.Series:
    """True where at least one source label col is observed (not NaN)."""
    present = [c for c in src_cols if c in df.columns]
    return df[present].notna().any(axis=1)


def derive_gold_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gold_<field> (schema value) and elig_<field> (scoreable?) columns.

    NaN handling (plan 2 / 2.2):
      - deriving the value: NaN is treated as 'not that label' (== 0).
      - scoring later: drop samples where the whole field is NaN (elig_<field> == False),
        so undefined gold does not pollute F1.
    """
    out = df.copy()
    out["gold_location"] = _derive_field(df, "location")
    out["gold_activity"] = _derive_field(df, "activity")
    out["gold_companion"] = _derive_field(df, "companion")
    out["elig_location"] = _eligible_mask(df, LOCATION_SRC_COLS)
    out["elig_activity"] = _eligible_mask(df, ACTIVITY_SRC_COLS)
    out["elig_companion"] = _eligible_mask(df, COMPANION_SRC_COLS)
    return out
