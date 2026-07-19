"""
demo_context_providers.py — validate Task-1 providers.

Generates synthetic HR data for personas whose age/fitness are KNOWN, then checks
the providers recover them from the HR alone (the way compare_cohorts validated the
baseline). Also shows graceful degradation when inputs are missing. If real
data/processed/*.parquet exist, it additionally reports the real subject's context.

    python demo_context_providers.py
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
import context_providers as cp                              # noqa: E402

RNG = np.random.default_rng(3)


def make_persona(age: int, resting: int, days: int = 21, workouts_per_week: int = 4):
    """Synthetic raw HR + workout table for a person of KNOWN age/resting-HR.

    HRmax = 208 - 0.7*age; workouts push HR toward ~0.95*HRmax so the observed
    peak (what the provider sees) is a realistic lower bound on true max.
    """
    hrmax = 208 - 0.7 * age
    rows, wk = [], []
    t0 = pd.Timestamp("2026-01-01 00:00")
    for d in range(days):
        run_days = set(RNG.choice(range(7), size=workouts_per_week, replace=False))
        for minute in range(0, 24 * 60, 5):           # a reading every 5 min
            ts = t0 + pd.Timedelta(days=d, minutes=minute)
            h = ts.hour
            if 0 <= h < 6:
                hr = RNG.normal(resting - 6, 3)        # sleep dip
            else:
                hr = RNG.normal(resting + 8, 6)        # daytime rest
            rows.append((ts, max(35.0, hr)))
        if (d % 7) in run_days:                        # one ~40-min workout that day
            wstart = t0 + pd.Timedelta(days=d, hours=18)
            wend = wstart + pd.Timedelta(minutes=40)
            wk.append(("Running", wstart, wend, 40.0, np.nan))
            for minute in range(0, 40, 5):             # overwrite HR in the workout window
                ts = wstart + pd.Timedelta(minutes=minute)
                peak_frac = RNG.uniform(0.88, 0.97)
                rows.append((ts, peak_frac * hrmax + RNG.normal(0, 3)))
    hr_raw = pd.DataFrame(rows, columns=["datetime", "value"]).sort_values("datetime")
    workouts = pd.DataFrame(wk, columns=["type", "start_time", "end_time", "duration", "distance"])
    return {"hr_raw": hr_raw.reset_index(drop=True), "workouts": workouts}


def run_persona(label, age, resting, **kw):
    frames = make_persona(age, resting, **kw)
    ce = cp.build_subject_context(frames, user={"goal": "endurance"})
    print(f"\n### {label}  (TRUTH: age={age}, resting~{resting})")
    print(ce.report())
    return ce


def main():
    print("=" * 80)
    print("TASK 1 PROVIDERS — recover context from HR alone (personas)")
    print("=" * 80)
    run_persona("Young athlete", age=24, resting=46, workouts_per_week=5)
    run_persona("Recreational adult", age=35, resting=62, workouts_per_week=3)
    run_persona("Older sedentary", age=57, resting=74, workouts_per_week=1)

    # --- graceful degradation: no workouts, then no data at all ---
    print("\n" + "=" * 80)
    print("ROBUSTNESS — missing inputs must yield 'unknown', not a crash")
    print("=" * 80)
    frames = make_persona(35, 62, workouts_per_week=3)
    no_wk = {"hr_raw": frames["hr_raw"]}                 # drop workouts -> weaker peak
    ce = cp.build_subject_context(no_wk)
    print("\n### Same person, workouts table removed:")
    print(ce.report())

    ce = cp.build_subject_context({})                    # nothing at all
    print("\n### No data at all:")
    print(ce.report())
    print(f"\nresulting SubjectContext: {ce.context.model_dump()}")
    print(f"coverage: {ce.context.coverage():.0%}")

    # --- real data if the pipeline has been run ---
    real = cp.load_frames()
    if real:
        print("\n" + "=" * 80)
        print(f"REAL DATA — frames found: {list(real)}")
        print("=" * 80)
        ce = cp.build_subject_context(real, user={"goal": "endurance"})
        print(ce.report())
        print(f"\nSubjectContext: {ce.context.model_dump()}")
    else:
        print("\n(no data/processed/*.parquet yet — run File 1-2 to get the real "
              "subject's predicted context.)")


if __name__ == "__main__":
    main()
