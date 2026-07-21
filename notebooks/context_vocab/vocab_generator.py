"""
vocab_generator.py — Task 1 (Asara Item 1): AI-GENERATE the context vocabulary.

For each dimension of the DEFINED profile (context_profile.CONTEXT_PROFILE) the LLM
emits a RICH controlled vocabulary: a list of values, each with `aliases` (free-text
synonyms, for later normalization of messy context) and a `physio_note` (how the value
modulates expected HR / HRV / what 'normal' means). The result is FROZEN to disk
(vocabulary.json) and compiled to Pydantic `Literal` types (generated_vocab.py) — so
the vocab is AI-generated yet REPRODUCIBLE (Asara's guardrail: temperature=0 +
controlled enums, generated once then frozen).

Offline-first: every LLM call is cached on disk; with no API key (or source="default")
a deterministic SEED vocabulary is used, so the whole track runs and verifies offline.
The live LLM path (source="auto"/"llm") is used only when GEMINI_API_KEY is present.

ISOLATED track: writes only under notebooks/context_vocab/ (frozen artifacts, committed),
data/context_vocab/vocab_cache/ (LLM cache, git-ignored) and results/context_vocab/.
Does NOT import or mutate the 01-04 / 03b pipeline.

NOT medical advice — these are reference categories for anomaly-detection context.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

import context_profile as cp
from context_profile import CONTEXT_PROFILE, ProfileDimension

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
TRACK_DIR = _THIS.parent
DATA_DIR = REPO_ROOT / "data" / "context_vocab"
CACHE_DIR = DATA_DIR / "vocab_cache"
RESULTS_DIR = REPO_ROOT / "results" / "context_vocab"
VOCAB_JSON = TRACK_DIR / "vocabulary.json"          # frozen vocab (committed artifact)
LITERALS_PY = TRACK_DIR / "generated_vocab.py"      # code-gen Literals (committed)
REPORT_MD = RESULTS_DIR / "vocab_report.md"
GEMINI_MODEL = "gemini-3.1-flash-lite"


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Schema — this is ALSO the LLM structured-output schema (one call per dimension).
# --------------------------------------------------------------------------- #
class VocabTerm(BaseModel):
    value: str                                   # short snake_case controlled token
    aliases: list[str] = Field(default_factory=list)   # synonyms -> normalize to value
    physio_note: str = ""                        # how it shifts expected HR/HRV/normality


class DimensionVocab(BaseModel):
    dimension: str
    terms: list[VocabTerm]


# --------------------------------------------------------------------------- #
# Deterministic SEED vocabulary — offline fallback (small but real). The LLM path
# produces a richer set; this guarantees the track runs and verifies with no key.
# --------------------------------------------------------------------------- #
def _t(value, aliases, note):
    return {"value": value, "aliases": aliases, "physio_note": note}


_SEED: dict[str, list[dict]] = {
    "demographic": [
        _t("age_under_18", ["teen", "adolescent"], "Higher HRmax (~200+), higher resting."),
        _t("age_18_29", ["twenties", "young adult"], "HRmax ~185-190; wide resting band."),
        _t("age_30_39", ["thirties"], "HRmax ~180; resting norms tighten."),
        _t("age_40_49", ["forties"], "HRmax ~175; gradual HRV decline."),
        _t("age_50_59", ["fifties"], "HRmax ~168; lower peaks expected."),
        _t("age_60_plus", ["senior", "elderly"], "HRmax ~165 and below; reduced reserve."),
        _t("male", ["man", "m"], "Slightly lower resting-HR/HRV reference than female."),
        _t("female", ["woman", "f"], "Slightly higher resting-HR reference than male."),
        _t("unknown", ["na", "unspecified"], "Missing -> widen the reference band."),
    ],
    "heart_health": [
        _t("athletic", ["very fit", "endurance-trained"], "Athletic bradycardia; resting 40-58."),
        _t("healthy", ["good", "fit"], "Resting ~55-70; strong recovery."),
        _t("average", ["typical"], "Resting ~65-80; ordinary recovery."),
        _t("at_risk", ["poor", "deconditioned"], "Elevated resting >80; slow recovery."),
        _t("unknown", ["na"], "Missing -> use age/fitness-pooled band."),
    ],
    "medical_conditions": [
        _t("none", ["healthy", "no condition"], "No HR-shifting condition assumed."),
        _t("hypertension", ["high blood pressure", "htn"], "May raise resting HR; caution."),
        _t("arrhythmia", ["afib", "irregular rhythm"], "Widen bands; rhythm irregularity expected."),
        _t("diabetes", ["t2dm", "diabetic"], "Possible autonomic effect on HRV."),
        _t("pregnancy", ["pregnant"], "Resting HR rises ~10-15 bpm; raise the floor."),
        _t("cardiac_history", ["prior MI", "heart disease"], "Widen and flag caveats."),
        _t("thyroid", ["hyperthyroid", "hypothyroid"], "Thyroid shifts resting HR up/down."),
        _t("unknown", ["na"], "Missing -> assume none but keep bands wide."),
    ],
    "occupation": [
        _t("sedentary_office", ["desk job", "9-5 office", "white collar"], "Low daytime load; long rest."),
        _t("shift_worker", ["night shift", "rotating shift"], "Circadian disruption; sleep window shifts."),
        _t("manual_labour", ["blue collar", "physical job"], "Elevated daytime HR from work."),
        _t("professional_athlete", ["pro athlete"], "Very high training load; athletic norms."),
        _t("healthcare", ["nurse", "doctor"], "Long shifts, on-feet; irregular rest."),
        _t("student", ["pupil"], "Variable schedule; mixed load."),
        _t("retired", ["pensioner"], "Low occupational load; more rest."),
        _t("unknown", ["na"], "Missing -> no occupational prior."),
    ],
    "geographic_location": [
        _t("temperate", ["mild climate"], "Neutral thermal baseline."),
        _t("hot_arid", ["desert", "dry heat"], "Heat raises resting HR; dry."),
        _t("hot_humid", ["tropical"], "Heat + humidity compound HR elevation."),
        _t("cold", ["cold climate", "polar"], "Cold; little upward HR shift."),
        _t("high_altitude", ["mountain", "highland"], "Hypoxia raises resting HR."),
        _t("unknown", ["na"], "Missing -> neutral baseline."),
    ],
    "weather": [
        _t("mild", ["comfortable", "pleasant"], "Thermally neutral; no HR shift."),
        _t("hot", ["warm", "heat"], "Cutaneous vasodilation raises resting HR."),
        _t("heatwave", ["extreme heat"], "Strong upward HR shift; heat stress."),
        _t("humid", ["muggy", "sticky"], "Humidity compounds heat effect on HR."),
        _t("cold", ["chilly", "freezing"], "Cold; minimal upward HR shift."),
        _t("unknown", ["na"], "Missing -> no weather adjustment."),
    ],
    "workout_type": [
        _t("run", ["running", "jog"], "High HR; aerobic; approaches vigorous band."),
        _t("walk", ["walking", "stroll"], "Light activity band."),
        _t("cycle", ["cycling", "bike"], "Aerobic; HR depends on effort/terrain."),
        _t("swim", ["swimming"], "Whole-body aerobic; wrist HR less reliable."),
        _t("row", ["rowing", "kayak"], "Full-body; on water; high aerobic load."),
        _t("strength", ["weights", "resistance"], "Intermittent HR spikes between sets."),
        _t("yoga", ["pilates", "stretching"], "Low HR; near resting/light."),
        _t("hiit", ["interval"], "Repeated near-max spikes and recoveries."),
        _t("hike", ["hiking", "trail"], "Sustained light-moderate; elevation matters."),
        _t("unknown", ["na", "other"], "Unclassified activity -> resting/light default."),
    ],
    "workout_duration": [
        _t("very_short", ["<10 min"], "Under ~10 min; little steady-state."),
        _t("short", ["10-25 min"], "~10-25 min; brief session."),
        _t("moderate", ["25-45 min"], "~25-45 min; typical session."),
        _t("long", ["45-90 min"], "~45-90 min; endurance-leaning."),
        _t("endurance", [">90 min"], "Over ~90 min; drift and fatigue expected."),
        _t("unknown", ["na"], "Missing duration -> no bucket."),
    ],
    "workout_location": [
        _t("gym", ["fitness centre", "indoor gym"], "Indoor; controlled temperature."),
        _t("home", ["at home"], "Indoor home workout."),
        _t("outdoor", ["road", "trail", "street"], "Outdoors; weather applies."),
        _t("park", ["greenspace"], "Outdoor park/trail."),
        _t("pool", ["swimming pool"], "Indoor/outdoor water; swim context."),
        _t("open_water", ["lake", "sea", "river"], "On/in water; rowing/swim context."),
        _t("studio", ["yoga studio"], "Indoor class/studio."),
        _t("unknown", ["na"], "Missing GPS -> unknown location."),
    ],
    "sleep": [
        _t("restful", ["good sleep", "deep"], "Full overnight dip; low sleep HR."),
        _t("adequate", ["ok sleep"], "Reasonable dip and duration."),
        _t("short", ["under-slept", "little sleep"], "Truncated night; elevated morning HR."),
        _t("fragmented", ["broken", "restless"], "Interrupted; blunted dip."),
        _t("irregular", ["shifting schedule"], "Inconsistent sleep window; circadian noise."),
        _t("unknown", ["na"], "No overnight coverage -> unknown."),
    ],
}


def _seed_dim(dim: ProfileDimension) -> DimensionVocab:
    terms = _SEED.get(dim.name) or [_t("unknown", ["na"], "No seed for this dimension.")]
    return DimensionVocab(dimension=dim.name, terms=[VocabTerm(**t) for t in terms])


# --------------------------------------------------------------------------- #
# LLM path (live, cached). Key discovered from the first .env found.
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


def _prompt(dim: ProfileDimension) -> str:
    numeric = ("This is a NUMERIC dimension expressed as named buckets; put the "
               "approximate numeric range inside each physio_note. "
               if dim.kind == "numeric_bucketed" else "")
    return (
        "You are building a controlled vocabulary for a wearable-health anomaly-"
        "detection pipeline. Given ONE context dimension, return a RICH but non-"
        "redundant set of controlled values for it. For each value give: `value` "
        "(a short snake_case token), `aliases` (free-text synonyms that should map to "
        "it), and `physio_note` (one line on how it shifts expected heart-rate / HRV / "
        "what 'normal' means). Always include an 'unknown' value for missing data. "
        "Ground it in established physiology; do NOT flag anomalies — only define the "
        f"reference vocabulary. {numeric}\n"
        f"Dimension: {dim.name}\nMeaning: {dim.description}"
    )


def _cache_path(prompt: str) -> Path:
    h = hashlib.sha256((GEMINI_MODEL + "\n" + prompt).encode()).hexdigest()[:32]
    return CACHE_DIR / f"{h}.json"


def _generate_dim_llm(dim: ProfileDimension, key: str) -> DimensionVocab:
    prompt = _prompt(dim)
    cp_ = _cache_path(prompt)
    if cp_.exists():
        dv = DimensionVocab(**json.loads(cp_.read_text(encoding="utf-8")))
        dv.dimension = dim.name
        return dv
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json",
        response_schema=DimensionVocab,
        thinking_config=types.ThinkingConfig(thinking_budget=0))
    r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg)
    dv: DimensionVocab = r.parsed
    dv.dimension = dim.name
    ensure_dirs()
    cp_.write_text(dv.model_dump_json(), encoding="utf-8")
    return dv


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_vocabulary(profile: list[ProfileDimension] = CONTEXT_PROFILE,
                        source: str = "auto") -> dict:
    """Generate the vocabulary for every dimension of `profile`.

    source="auto"   : LLM if GEMINI_API_KEY is present, else the offline seed.
    source="llm"    : force the LLM (raises if no key).
    source="default": force the offline seed (no network).

    Returns a JSON-ready dict: {"meta": {...}, "dimensions": {name: DimensionVocab...}}.
    On a per-dimension LLM error it degrades to that dimension's seed (never crashes).
    """
    key = None if source == "default" else _load_key()
    if source == "llm" and not key:
        raise RuntimeError("source='llm' but no GEMINI_API_KEY found (.env).")
    use_llm = bool(key) and source != "default"

    dims: dict[str, dict] = {}
    per_dim_source: dict[str, str] = {}
    for dim in profile:
        if use_llm:
            try:
                dv = _generate_dim_llm(dim, key)
                per_dim_source[dim.name] = "llm"
            except Exception as e:                       # any failure -> seed fallback
                dv = _seed_dim(dim)
                per_dim_source[dim.name] = f"seed ({type(e).__name__})"
        else:
            dv = _seed_dim(dim)
            per_dim_source[dim.name] = "seed"
        dims[dim.name] = dv.model_dump()

    # Report the EFFECTIVE source honestly, from what actually happened per dimension
    # (never label a seed-fallback run as "llm" — see graph_model.structural_scores).
    n_llm = sum(1 for s in per_dim_source.values() if s == "llm")
    if not use_llm:
        eff_source = "seed_default"
    elif n_llm == len(dims):
        eff_source = "llm"
    elif n_llm == 0:
        eff_source = "seed_fallback (llm requested but every call failed)"
    else:
        eff_source = f"mixed ({n_llm}/{len(dims)} llm, rest seed_fallback)"

    meta = {
        "source": eff_source,
        "model": GEMINI_MODEL if n_llm else None,
        "n_dimensions": len(dims),
        "n_terms": sum(len(d["terms"]) for d in dims.values()),
        "per_dimension_source": per_dim_source,
    }
    return {"meta": meta, "dimensions": dims}


def freeze_vocabulary(vocab: dict) -> Path:
    """Write the frozen vocabulary.json (the committed, reproducible artifact)."""
    VOCAB_JSON.write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    return VOCAB_JSON


def load_vocabulary() -> dict:
    """Read the frozen vocabulary.json."""
    return json.loads(VOCAB_JSON.read_text(encoding="utf-8"))


def _camel(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_")) + "Vocab"


def emit_literals(vocab: dict) -> Path:
    """Code-gen generated_vocab.py: a Pydantic-compatible `Literal` per dimension +
    ALIASES / NOTES lookup dicts. This is how the AI-generated vocab becomes a fixed,
    reproducible enum that Task 2/3 import."""
    lines = [
        '"""',
        "generated_vocab.py — AUTO-GENERATED by vocab_generator.emit_literals().",
        "Do NOT edit by hand; regenerate via `python demo_vocab.py`.",
        f"Source: {vocab['meta']['source']} | dimensions: {vocab['meta']['n_dimensions']} "
        f"| terms: {vocab['meta']['n_terms']}.",
        '"""',
        "from __future__ import annotations",
        "from typing import Literal",
        "",
    ]
    aliases: dict[str, dict[str, str]] = {}
    notes: dict[str, dict[str, str]] = {}
    for name, dv in vocab["dimensions"].items():
        values = [t["value"] for t in dv["terms"]]
        literal = ", ".join(json.dumps(v) for v in values)
        lines.append(f"{_camel(name)} = Literal[{literal}]")
        aliases[name] = {a.lower(): t["value"] for t in dv["terms"] for a in t.get("aliases", [])}
        notes[name] = {t["value"]: t.get("physio_note", "") for t in dv["terms"]}
    lines += [
        "",
        "# free-text alias -> controlled value, per dimension",
        f"ALIASES = {json.dumps(aliases, indent=2, ensure_ascii=False)}",
        "",
        "# controlled value -> physiological note, per dimension",
        f"NOTES = {json.dumps(notes, indent=2, ensure_ascii=False)}",
        "",
    ]
    LITERALS_PY.write_text("\n".join(lines), encoding="utf-8")
    return LITERALS_PY


def write_report(vocab: dict) -> Path:
    """Human-readable markdown report of the generated vocabulary."""
    ensure_dirs()
    m = vocab["meta"]
    out = [f"# Context Vocabulary — generated report",
           "",
           f"- source: **{m['source']}**  ", f"- model: `{m['model']}`  ",
           f"- dimensions: {m['n_dimensions']}, terms: {m['n_terms']}", ""]
    for name, dv in vocab["dimensions"].items():
        dim = cp.PROFILE_BY_NAME.get(name)
        lvl = f"{dim.level}/{dim.kind}" if dim else ""
        src = m["per_dimension_source"].get(name, "")
        out.append(f"## {name}  ({lvl})  — {src}")
        out.append("")
        out.append("| value | aliases | physio_note |")
        out.append("|---|---|---|")
        for t in dv["terms"]:
            al = ", ".join(t.get("aliases", []))
            out.append(f"| `{t['value']}` | {al} | {t.get('physio_note','')} |")
        out.append("")
    REPORT_MD.write_text("\n".join(out), encoding="utf-8")
    return REPORT_MD


def normalize(dimension: str, text: str, vocab: dict | None = None) -> str:
    """Map a free-text context string to the controlled value for `dimension`.

    Deterministic: exact value match, then alias match, then substring — else
    'unknown'. This is the normalization the pipeline uses to fold messy context
    onto the AI-generated vocab.
    """
    vocab = vocab or load_vocabulary()
    dv = vocab["dimensions"].get(dimension)
    if not dv:
        return "unknown"
    s = (text or "").strip().lower()
    values = [t["value"] for t in dv["terms"]]
    if s in values:
        return s
    for t in dv["terms"]:
        if s == t["value"] or s in [a.lower() for a in t.get("aliases", [])]:
            return t["value"]
    for t in dv["terms"]:                                # loose substring fallback
        if s and (s in t["value"] or any(s in a.lower() for a in t.get("aliases", []))):
            return t["value"]
    return "unknown" if "unknown" in values else values[-1]


def build_all(source: str = "auto") -> dict:
    """One-shot: generate -> freeze JSON -> emit Literals -> write report."""
    ensure_dirs()
    vocab = generate_vocabulary(source=source)
    freeze_vocabulary(vocab)
    emit_literals(vocab)
    write_report(vocab)
    return vocab
