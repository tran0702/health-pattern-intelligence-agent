"""
compare_cohorts.py — evidence for Asara that CONTEXT drives the LLM baseline.

Runs establish_baseline() (Gemini, cached) for several contrasting cohorts and
prints a side-by-side table of the expected-normal ranges. If the ranges move in
the physiologically-expected direction (athlete -> low resting/high HRmax; older
sedentary -> high resting/low HRmax; pregnancy -> raised resting; all-unknown ->
widest), that is direct evidence the context is what shapes the baseline.

    python compare_cohorts.py            # live Gemini (cached), needs GEMINI_API_KEY
    python compare_cohorts.py --default  # offline physiology baseline, for comparison
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_library import SubjectContext          # noqa: E402
import global_baseline as gb                          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

COHORTS: list[tuple[str, SubjectContext]] = [
    ("young athlete (F)",
     SubjectContext(age_band="18_29", sex="female", fitness_level="athlete",
                    goal="endurance", health_conditions=["none"])),
    ("recreational (M)",
     SubjectContext(age_band="30_39", sex="male", fitness_level="recreational",
                    goal="endurance", health_conditions=["none"])),
    ("older sedentary +HTN (M)",
     SubjectContext(age_band="50_59", sex="male", fitness_level="sedentary",
                    goal="general_health", health_conditions=["hypertension"])),
    ("pregnant (F)",
     SubjectContext(age_band="30_39", sex="female", fitness_level="recreational",
                    goal="general_health", health_conditions=["pregnancy"])),
    ("60+ recreational (F)",
     SubjectContext(age_band="60_plus", sex="female", fitness_level="recreational",
                    goal="general_health", health_conditions=["none"])),
    ("all unknown (missing)",
     SubjectContext()),
]


def main() -> None:
    source = "default" if "--default" in sys.argv else "llm"
    rng = lambda r: f"{r.low:.0f}-{r.high:.0f}"

    rows = []
    for label, ctx in COHORTS:
        base = gb.establish_baseline(ctx, source=source)
        rows.append({
            "cohort": label,
            "resting": rng(base.resting_hr),
            "sleep": rng(base.sleep_hr),
            "light": rng(base.light_activity_hr),
            "vigorous": rng(base.vigorous_activity_hr),
            "HRV_sdnn": rng(base.hrv_sdnn_ms),
            "HRmax": f"{base.max_hr_bpm:.0f}",
            "src": base.source,
            "rationale": base.rationale,
            "caveats": base.caveats,
        })

    df = pd.DataFrame(rows)
    show = ["cohort", "resting", "sleep", "light", "vigorous", "HRV_sdnn", "HRmax", "src"]
    print(f"\n=== Gemini global baseline by cohort (source={source}) ===\n")
    print(df[show].to_string(index=False))

    gb.ensure_dirs()
    out = gb.RESULTS_DIR / f"cohort_baselines_{source}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out.relative_to(REPO_ROOT)}")
    # show the reasoning for the two most different cohorts
    print("\n--- rationale (young athlete) ---\n", rows[0]["rationale"])
    print("\n--- rationale (older sedentary +HTN) ---\n", rows[2]["rationale"])
    print("\n--- caveats (all unknown) ---\n", rows[5]["caveats"])


if __name__ == "__main__":
    main()
