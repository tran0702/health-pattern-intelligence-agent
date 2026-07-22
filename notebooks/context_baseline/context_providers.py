"""
context_providers.py — Task 1 (the missing half): ESTABLISH context from data.

The Context Library (context_library.py) defines WHAT a context is. This module
is the set of PROVIDERS that predict those attributes from the initial dataset,
before the main pipeline runs — realising the meeting's "predicted age / fitness /
location type" and Leelanga's "context is established externally, the pipeline
receives it".

Subject-level providers (predict the cohort descriptor from HR / weather):
  * predict_age_band      — invert Tanaka HRmax from observed peak HR.
  * predict_fitness_from_exertion — the LIVE fitness predictor: HR recovery (HRR60)
    + training volume + exertion peak, all INDEPENDENT of resting HR, so the band is
    not judged by the same signal that set it (circularity fix, PROJECT_STATUS 4-(1)).
  * predict_fitness_level — the older resting-HR chart classifier, kept REPORT-ONLY
    and printed beside the exertion label to expose the gap; never feeds the band.
  * predict_home_climate  — from the weather distribution (temp + humidity),
    grounded in the geocoded home place name when the location_context track has run.
Episode-level context (activity / location_type) is already derived in File 2 /
the enrichment C2 geocoder, so those are wired in, not re-implemented here.

Every provider returns a `FieldEstimate(value, confidence, evidence)` and degrades
to ('unknown', 0.0, why) when its inputs are missing — the library stays robust to
partial data. `build_subject_context` fuses them with any user-provided overrides
and an optional Task-2 GLOBAL prior into a `SubjectContext`, with precedence
user > individual signal > confidence-gated global prior > 'unknown' (step 6,
PROJECT_STATUS 4-(2)). The prior is a plain dict from
global_context.global_prior_from_context, so this module needs no import of the
global track (one-way, artifact-style wiring).

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
            if e.evidence.startswith("user-provided"):
                src = "user"
            elif e.evidence.startswith("from "):     # confidence-gated global prior
                src = "prior"
            else:
                src = "predicted"
            lines.append(f"  {k:22s} = {e.value:<16} conf={e.confidence:.2f}  [{src}] {e.evidence}")
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


# --------------------------------------------------------------------------- #
# Exertion-based fitness (the CIRCULARITY fix — PROJECT_STATUS.md 4-(1)).
# Resting HR must not both SET the fitness band and be JUDGED by it. These signals
# come from workouts/recovery only; none of them read _resting_hr().
# --------------------------------------------------------------------------- #
_HRR_GATE = 130.0        # in-workout peak (bpm) a session must reach before its
                         # recovery is meaningful — a ~100-bpm walk has nothing to
                         # recover from. Absolute, so it needs no (circular) age.


def _nearest_after(t: pd.Series, v: pd.Series, e, target_s: float,
                   tol_s: float = 20.0) -> float | None:
    """HR value of the reading closest to `e + target_s` seconds, within +/- tol_s;
    None if no reading lands in that window."""
    lo = e + pd.Timedelta(seconds=target_s - tol_s)
    hi = e + pd.Timedelta(seconds=target_s + tol_s)
    m = (t >= lo) & (t <= hi)
    if not m.any():
        return None
    idx = (t[m] - (e + pd.Timedelta(seconds=target_s))).abs().idxmin()
    return float(v.loc[idx])


def _hrr60(frames: dict) -> tuple[float | None, int, str]:
    """Median 1-minute heart-rate recovery over *qualifying* workouts.

    HRR60 = (in-workout peak HR) - (HR ~60 s after workout end). A large, fast drop
    is a classic fitness marker and is independent of the absolute resting HR. Only
    workouts whose in-workout peak clears _HRR_GATE count, so low-intensity walks do
    not dilute the estimate. Returns (median_hrr, n_qualifying, note)."""
    wk, hr = frames.get("workouts"), frames.get("hr_raw")
    if wk is None or hr is None or "datetime" not in getattr(hr, "columns", []):
        return None, 0, "no workouts+raw HR for recovery"
    wk = wk.dropna(subset=["start_time", "end_time"])
    if len(wk) == 0:
        return None, 0, "no dated workouts"
    t = hr["datetime"]
    v = pd.to_numeric(hr["value"], errors="coerce")
    hrrs = []
    for s, e in zip(wk["start_time"], wk["end_time"]):
        inw = (t >= s) & (t <= e)
        if int(inw.sum()) < 3:
            continue
        peak = float(v[inw].max())
        if peak < _HRR_GATE:                      # gate: must have really exerted
            continue
        hr60 = _nearest_after(t, v, e, 60.0)
        if hr60 is None:
            continue
        hrrs.append(peak - hr60)
    if not hrrs:
        return None, 0, f"no workout reached peak>={_HRR_GATE:.0f} bpm with a +60s sample"
    med = float(np.median(hrrs))
    return med, len(hrrs), f"HRR60~{med:.0f}bpm/{len(hrrs)} exertion workouts"


def _workout_load(frames: dict) -> tuple[float, float, str]:
    """(sessions_per_week, active_min_per_week, note) from the workouts table — a
    training-habit signal, also independent of resting HR. (0,0,..) if absent."""
    wk = frames.get("workouts")
    if wk is None or "start_time" not in getattr(wk, "columns", []):
        return 0.0, 0.0, "no workouts table"
    w = wk.dropna(subset=["start_time", "end_time"])
    if len(w) < 2:
        return 0.0, 0.0, "too few workouts for a rate"
    span_days = (w["end_time"].max() - w["start_time"].min()).total_seconds() / 86400
    weeks = max(span_days / 7.0, 1.0)
    spw = len(w) / weeks
    dur = pd.to_numeric(w.get("duration", pd.Series(dtype=float)), errors="coerce").dropna()
    mpw = float(dur.sum() / weeks) if len(dur) else float("nan")
    mnote = f", ~{mpw:.0f}min/wk" if mpw == mpw else ""
    return spw, mpw, f"{spw:.1f} sessions/wk{mnote}"


# HRR60 (bpm) -> fitness level (clinical 1-min recovery bands: <12 abnormal).
def _hrr_to_level(hrr: float) -> str:
    return ("athlete" if hrr >= 30 else "trained" if hrr >= 20
            else "recreational" if hrr >= 12 else "sedentary")


_LEVEL_ORDER = {"sedentary": 0, "recreational": 1, "trained": 2, "athlete": 3}
_ORDER_LEVEL = {n: lv for lv, n in _LEVEL_ORDER.items()}


def _volume_ceiling(spw: float) -> str:
    """Highest fitness level a given weekly session frequency can justify: good
    recovery + rare training is 'recreational', not 'athlete'."""
    return ("athlete" if spw >= 4 else "trained" if spw >= 3
            else "recreational" if spw >= 1 else "sedentary")


def predict_fitness_from_exertion(frames: dict, age_band: str = "unknown",
                                  sex: str = "unknown") -> FieldEstimate:
    """Classify fitness from EXERTION, never from resting HR (the circularity fix,
    PROJECT_STATUS.md 4-(1)). Independent signals:
      * HRR60  - 1-min recovery after real-effort workouts (cardiovascular capacity),
      * volume - sessions/week + active min/week (training habit),
      * peak   - whether the subject ever elevates HR at all (via _peak_hr).
    HRR sets the level; training volume caps it; a subject who never reaches an
    exertion peak floors at 'sedentary'. age_band/sex are accepted only for
    signature parity with the chart predictor - HRR needs neither."""
    hrr, n_qual, hnote = _hrr60(frames)
    spw, _mpw, vnote = _workout_load(frames)
    peak, _npk, pnote, from_wk = _peak_hr(frames)

    if hrr is None and not (from_wk and peak is not None) and spw == 0:
        return FieldEstimate("unknown", 0.0, f"no exertion signal ({hnote}; {vnote})")

    if hrr is None:                               # trains but never truly exerts
        conf = round(0.30 + 0.20 * min(1.0, spw / 3.0), 2)
        return FieldEstimate("sedentary", conf,
                             f"never reached peak>={_HRR_GATE:.0f}bpm; {vnote}; {pnote}")

    level = _hrr_to_level(hrr)
    capped = _ORDER_LEVEL[min(_LEVEL_ORDER[level], _LEVEL_ORDER[_volume_ceiling(spw)])]
    conf = round(0.30 + 0.50 * min(1.0, n_qual / 20.0), 2)
    cap = "" if capped == level else f"; volume-capped {level}->{capped}"
    return FieldEstimate(capped, conf, f"{hnote}; {vnote}{cap}")


def predict_fitness_level(frames: dict, age_band: str = "unknown",
                          sex: str = "unknown") -> FieldEstimate:
    """RESTING-BASED fitness — REPORT ONLY, must NOT feed the band (it is the source
    of the circularity: resting HR would both set and be judged by the band). Kept so
    build_subject_context can print it beside the exertion label to expose the gap.
    Classifies resting HR via the age/sex Resting-HR Chart; age_band + sex index the
    chart (pooled if unknown). Output in the 4-level FitnessLevel vocab; the exact
    7-level chart category is reported. Live band prediction uses
    predict_fitness_from_exertion instead."""
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
    "fitness_level": predict_fitness_from_exertion,   # exertion, not resting HR (4-(1))
    "home_climate": predict_home_climate,
}
# fields with no data predictor yet — resolved from `user`, or (for the subset a
# dataset domain can defensibly inform, i.e. heart_health) a confidence-gated GLOBAL
# prior. health_conditions is intentionally NOT prior-eligible (see _PRIOR_FIELDS).
_USER_OR_PRIOR = ("sex", "health_conditions", "goal", "occupation", "sleep", "heart_health")
# fields a Task-2 GLOBAL prior may fall back on when the individual signal is unknown
# (values come from global_context.global_prior_from_context). Kept small + medically
# defensible: a population tier the dataset domain genuinely implies.
_PRIOR_FIELDS = ("fitness_level", "heart_health")
# a population prior is discounted vs a direct individual signal when its confidence
# is reported, so it can never masquerade as individual evidence.
_PRIOR_DISCOUNT = 0.7


def build_subject_context(frames: dict, user: dict | None = None,
                          global_prior: dict | None = None,
                          min_conf: float = 0.25,
                          min_global_conf: float = 0.5) -> ContextEstimation:
    """Fuse data-predicted fields with user overrides (and an optional Task-2 GLOBAL
    prior) into a SubjectContext.

    Precedence per field:
      user value (source of truth, conf 1)
        > individual data prediction (accepted when conf >= min_conf)
        > confidence-gated GLOBAL prior (only for _PRIOR_FIELDS, only when the
          dataset's own classification confidence >= min_global_conf)
        > 'unknown'.

    This is step 6 of Task 2 (PROJECT_STATUS 4-(2)): the GLOBAL context (what kind of
    dataset this is) informs the individual layer without ever overriding a real
    individual signal or a user value. `global_prior` is a plain dict produced by
    global_context.global_prior_from_context, so this module keeps no import of the
    global track — one-way, artifact-style wiring. `min_global_conf` is the gate that
    keeps a low-confidence dataset label from injecting a guess ('unknown' wins).
    """
    user = user or {}
    est: dict[str, FieldEstimate] = {}
    values: dict[str, object] = {}

    def _prior_estimate(field: str) -> FieldEstimate | None:
        """A confidence-gated global prior for `field`, or None if the dataset is not
        confident enough (or offers no prior for this field)."""
        if not global_prior or field not in global_prior:
            return None
        p = global_prior[field]
        if float(p.get("confidence", 0.0)) < min_global_conf:
            return None
        conf = round(_PRIOR_DISCOUNT * float(p["confidence"]), 2)
        return FieldEstimate(p["value"], conf,
                             "from " + p.get("evidence", "global prior")
                             + " (individual signal unknown)")

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
        if not e.known() and f in _PRIOR_FIELDS:      # fall back on the global prior
            pe = _prior_estimate(f)
            if pe is not None:
                e = pe
        est[f] = e
        values[f] = e.value

    # report-only: the RESTING-based fitness label, shown BESIDE the exertion one so
    # the circularity it used to cause stays visible but never feeds the band (4-(1)).
    if "hr_raw" in frames or "hr_features" in frames:
        rf = predict_fitness_level(frames, age_band=values.get("age_band", "unknown"),
                                   sex=user.get("sex", "unknown"))
        est["fitness_resting_report"] = FieldEstimate(
            rf.value, rf.confidence, "REPORT-ONLY (not used for band) - " + rf.evidence)

    for f in _USER_OR_PRIOR:
        if f in user and user[f] not in (None, "unknown"):
            v = user[f]
            est[f] = FieldEstimate(str(v), 1.0, "user-provided")
            values[f] = v
            continue
        pe = _prior_estimate(f) if f in _PRIOR_FIELDS else None
        if pe is not None:
            est[f] = pe
            values[f] = pe.value
        else:
            est[f] = FieldEstimate("unknown", 0.0, "not provided (no data predictor)")

    # exact age (user only) — Tanaka HRmax wants a real number, not a band midpoint.
    if user.get("age_years") is not None:
        try:
            values["age_years"] = float(user["age_years"])
            est["age_years"] = FieldEstimate(str(values["age_years"]), 1.0, "user-provided")
        except (TypeError, ValueError):
            pass

    hc = values.get("health_conditions", "unknown")
    ctx = SubjectContext(
        age_band=values.get("age_band", "unknown"),
        sex=values.get("sex", "unknown"),
        age_years=values.get("age_years"),
        fitness_level=values.get("fitness_level", "unknown"),
        heart_health=values.get("heart_health", "unknown"),
        health_conditions=hc if isinstance(hc, list) else [hc],
        occupation=values.get("occupation", "unknown"),
        sleep=values.get("sleep", "unknown"),
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
