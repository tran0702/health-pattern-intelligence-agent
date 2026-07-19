"""
demo_context_baseline.py — end-to-end demo of the approach-(a) design.

Flow (matches the Asara 3 points):
  1. External context is INGESTED (SubjectContext) — the pipeline does not invent it.
  2. The LLM (or offline physiology fallback) ESTABLISHES the cohort global baseline.
  3. Anomaly detection compares the individual's episodes to that baseline and
     TRANSLATES each deviation into a lifestyle/health statement.

Runs fully offline (source="default"). Pass "--llm" to use Gemini if a key is in
.env. Uses data/processed/behavioral_episodes.parquet when present, otherwise a
small synthetic episode set so the demo always runs.

    python demo_context_baseline.py            # offline physiology baseline
    python demo_context_baseline.py --llm      # live Gemini baseline (needs key)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:                                          # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_library import SubjectContext, registry_table            # noqa: E402
import global_baseline as gb                                          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EP_PATH = REPO_ROOT / "data" / "processed" / "behavioral_episodes.parquet"


def load_episodes() -> tuple[pd.DataFrame, str]:
    """Real pipeline episodes if available, else a synthetic stand-in."""
    if EP_PATH.exists():
        ep = pd.read_parquet(EP_PATH)
        ep["hour_of_day"] = pd.to_datetime(ep["datetime"]).dt.hour
        keep = [c for c in ["avg_hr", "activity", "location_type", "hour_of_day"]
                if c in ep.columns]
        return ep[keep].copy(), f"real ({EP_PATH.name}, {len(ep)} windows)"

    rng = np.random.default_rng(0)
    n = 400
    hours = rng.integers(0, 24, n)
    activity = np.where(hours < 6, "sleep",
                        np.where(rng.random(n) < 0.12, "run", "rest"))
    avg_hr = np.where(activity == "sleep", rng.normal(52, 4, n),
                      np.where(activity == "run", rng.normal(150, 12, n),
                               rng.normal(68, 6, n)))
    # inject a few obvious deviations for the demo
    avg_hr[:6] = [95, 98, 92, 44, 190, 41]
    activity[:6] = ["rest", "sleep", "rest", "sleep", "run", "rest"]
    hours[:6] = [14, 3, 22, 2, 18, 15]
    df = pd.DataFrame({"avg_hr": avg_hr.round(1), "activity": activity,
                       "location_type": rng.choice(["home", "gym", "outdoors"], n),
                       "hour_of_day": hours})
    return df, f"synthetic ({n} windows; no processed data yet)"


def main() -> None:
    use_llm = "--llm" in sys.argv
    print("=" * 78)
    print("CONTEXT LIBRARY (Task 1) — external attributes ingested before the pipeline")
    print("=" * 78)
    print(registry_table())

    # 1) External context, WITH a missing field to show graceful degradation.
    ctx = SubjectContext(age_band="30_39", sex="male", fitness_level="recreational",
                         health_conditions=["none"], goal="endurance")  # home_climate unknown
    print(f"\nSubjectContext: {ctx.model_dump()}")
    print(f"context coverage (known cohort fields): {ctx.coverage():.0%}")

    # 2) LLM establishes the cohort global baseline (approach (a)); offline fallback otherwise.
    src = "llm" if use_llm else "default"
    base = gb.establish_baseline(ctx, source=src)
    print("\n" + "=" * 78)
    print(f"GLOBAL BASELINE (Task 2, approach a) — source = {base.source}")
    print("=" * 78)
    for f in ("resting_hr", "sleep_hr", "light_activity_hr", "vigorous_activity_hr", "hrv_sdnn_ms"):
        r = getattr(base, f)
        print(f"  {f:22s} [{r.low:6.1f}, {r.high:6.1f}]")
    print(f"  {'max_hr_bpm':22s} {base.max_hr_bpm:.1f}")
    print(f"  rationale: {base.rationale}")
    print(f"  caveats  : {base.caveats}")

    # 3) Detect deviations vs the baseline + translate their meaning.
    ep, kind = load_episodes()
    scored = gb.detect_against_baseline(ep, base, flag_at=1.0)
    n_flag = int(scored["is_anomaly"].sum())
    print("\n" + "=" * 78)
    print(f"ANOMALY DETECTION vs BASELINE (Task 3) — episodes: {kind}")
    print("=" * 78)
    print(f"flagged {n_flag} / {len(scored)} windows (deviation >= 1 half-width outside band)")
    top = scored[scored["is_anomaly"]].sort_values("deviation", ascending=False).head(6)
    for _, row in top.iterrows():
        msg = gb.translate(row, ctx)
        print(f"\n  hr={row['avg_hr']:.0f}  activity={row['activity']:<7} "
              f"band[{row['sub_range']}]=[{row['band_lo']:.0f},{row['band_hi']:.0f}]  "
              f"dev={row['deviation']:.2f} {row['direction']}")
        print(f"    -> {msg}")

    gb.ensure_dirs()
    out = gb.RESULTS_DIR / "demo_scored_episodes.csv"
    scored.to_csv(out, index=False)
    print(f"\nsaved {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
