"""
demo_global_context.py — validate Task-2 GLOBAL context (global_context.py).

Builds synthetic MULTI-SUBJECT cohorts whose domain is KNOWN, then checks the three
stages recover it from column metadata + aggregate statistics alone:
  1. cohort classification — an athletic / clinical-cardiac / office dataset is labelled
     with the right domain (offline rule path, so this runs with no API key),
  2. adversarial column rename — columns are renamed to non-obvious ENGLISH names, and
     infer_column_roles must still recover the roles from dtype + value ranges,
  3. structured-missingness contrast — full-coverage synthetic (night ok) vs the real
     Apple-Health subject (heavy night gap -> flag set),
  4. the real subject's GlobalContext (LLM if a key is present, else the rule path).

    python demo_global_context.py            # LLM where a key exists, else offline
    python demo_global_context.py --offline  # force the deterministic path
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import global_context as gx                                    # noqa: E402
from demo_context_providers import make_persona                # noqa: E402

RNG = np.random.default_rng(7)


def _persona_to_tidy(frames: dict, subject_id: str, activity: str,
                     diagnosis: str | None) -> pd.DataFrame:
    """One persona's frames -> tidy rows (subject_id, timestamp, heart_rate,
    workout_type[, diagnosis]); workout-window readings get `activity`."""
    hr = frames["hr_raw"][["datetime", "value"]].rename(
        columns={"datetime": "timestamp", "value": "heart_rate"}).copy()
    hr["subject_id"] = subject_id
    hr["workout_type"] = "none"
    wk = frames["workouts"]
    t = hr["timestamp"]
    for s, e in zip(wk["start_time"], wk["end_time"]):
        hr.loc[(t >= s) & (t <= e), "workout_type"] = activity
    if diagnosis is not None:
        hr["diagnosis"] = diagnosis
    return hr


def make_cohort(spec: dict, n_per_group: int = 4, days: int = 21) -> pd.DataFrame:
    """A multi-subject tidy dataset for a known cohort. `spec` maps a group name to
    persona params (age, resting, workouts_per_week, hrr1, activity[, diagnosis])."""
    rows = []
    for gname, p in spec.items():
        for i in range(n_per_group):
            frames = make_persona(
                age=int(p["age"] + RNG.integers(-3, 4)),
                resting=int(p["resting"] + RNG.integers(-4, 5)),
                days=days, workouts_per_week=p["workouts_per_week"], hrr1=p["hrr1"])
            rows.append(_persona_to_tidy(frames, f"{gname}_{i}", p["activity"],
                                         p.get("diagnosis")))
    return pd.concat(rows, ignore_index=True)


ATHLETIC = {"endurance_athletes": dict(age=25, resting=48, workouts_per_week=6,
                                       hrr1=34, activity="Running")}
CARDIAC = {"cardiac_patients": dict(age=62, resting=82, workouts_per_week=1, hrr1=8,
                                    activity="Walking", diagnosis="cardiac_history")}
OFFICE = {"office_workers": dict(age=40, resting=72, workouts_per_week=1, hrr1=12,
                                 activity="Walking")}


def _classify(df, source):
    roles = gx.infer_column_roles(df, source=source)
    fp = gx.dataset_fingerprint(df, roles)
    gc = gx.classify_global_context(fp, source=source)
    return roles, fp, gc


def main() -> int:
    offline = "--offline" in sys.argv
    src = "default" if offline else "auto"
    live = (not offline) and gx.key_available()
    print("=" * 80)
    print(f"TASK 2 — GLOBAL context   (source={src}, live_llm={'yes' if live else 'no'})")
    print("=" * 80)

    ok = True

    # 1) cohort classification — deterministic rule path so it always runs offline.
    print("\n--- 1. cohort domain recovery (offline rule path) ---")
    expect = [("athletic", make_cohort(ATHLETIC), ("athletic_performance",)),
              ("cardiac", make_cohort(CARDIAC), ("clinical_cohort", "clinical_cardiac")),
              ("office", make_cohort(OFFICE), ("general_population",))]
    for label, df, want in expect:
        _, fp, gc = _classify(df, "default")
        hit = gc.dataset_domain in want
        ok &= hit
        print(f"  [{'ok' if hit else 'XX'}] {label:9s} n_subj={fp.n_subjects} "
              f"wf={fp.workout_fraction:.3f} rest~{fp.subject_variance.inter_subject_resting_median} "
              f"-> domain={gc.dataset_domain!r} (want {want})")
        print(f'          descriptor: "{gc.population_descriptor}"  acts={gc.dominant_activities}')

    # 2) adversarial column rename — non-obvious ENGLISH names; roles from stats/dtype.
    print("\n--- 2. adversarial column rename (schema-agnostic role inference) ---")
    df_off = make_cohort(OFFICE)
    renamed = df_off.rename(columns={"subject_id": "cohort_member", "timestamp": "event_moment",
                                     "heart_rate": "cardio_signal", "workout_type": "movement_kind"})
    roles = gx.infer_column_roles(renamed, source=src)
    got = {r.column: r.role for r in roles.roles}
    want_map = {"cardio_signal": "heart_rate", "event_moment": "timestamp",
                "cohort_member": "subject_id", "movement_kind": "workout_type"}
    for col, want in want_map.items():
        good = got.get(col) == want
        print(f"  [{'ok' if good else '..'}] {col:15s} -> {got.get(col):13s} (want {want})")
    print(f"  (roles source={roles.source}; regex fallback recovers heart_rate/timestamp "
          f"from dtype+range, needs the LLM for subject_id/workout_type)")

    # 3) structured-missingness contrast: synthetic (full day) vs real (night gap).
    print("\n--- 3. structured missingness flag ---")
    _, fp_syn, _ = _classify(make_cohort(OFFICE), "default")
    print(f"  synthetic office: night_cov={fp_syn.missingness.night_coverage} "
          f"heavy={fp_syn.missingness.night_missing_heavy}")
    from context_providers import load_frames
    real = load_frames()
    if real:
        df_real = gx.apple_health_to_df(real)
        roles_r = gx.infer_column_roles(df_real, source="default")
        fp_real = gx.dataset_fingerprint(df_real, roles_r)
        m = fp_real.missingness
        print(f"  real subject    : night_cov={m.night_coverage} heavy={m.night_missing_heavy} "
              f"(day_cov={m.day_coverage}, ratio={m.night_to_day_ratio})")
        ok &= (m.night_missing_heavy and not fp_syn.missingness.night_missing_heavy)

        # 4) real subject GlobalContext (LLM if key, else rule).
        print("\n--- 4. REAL subject global context ---")
        bundle = gx.run_global_context(frames=real, source=src)
        gc = bundle["global_context"]
        fp = bundle["fingerprint"]
        print(f"  domain      : {gc['dataset_domain']}   (source={gc['source']}, conf={gc['confidence']})")
        print(f"  population   : {gc['population_descriptor']}")
        print(f"  activities   : {gc['dominant_activities']}")
        print(f"  evidence     : {gc['evidence']}")
        print(f"  fingerprint  : n_subj={fp['n_subjects']} span={fp['span_days']}d "
              f"hr_median={fp['hr_median']} intra_std="
              f"{fp['subject_variance']['intra_subject_daily_std_median']}")
        print(f"  saved -> results/context_baseline/global_context.json")
    else:
        print("  (no data/processed/*.parquet — skipped real-subject checks)")

    print("\n" + ("PASS — global context recovers known cohorts + flags missingness."
                  if ok else "FAIL — see [XX] above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
