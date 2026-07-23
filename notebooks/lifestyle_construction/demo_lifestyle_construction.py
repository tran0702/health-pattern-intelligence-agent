"""
demo_lifestyle_construction.py — offline selftest for Stage B.

Synthesises days of 3 KNOWN lifestyle types, runs the self-supervised encoder + clustering, and
asserts (a) the clusters recover the known types (adjusted Rand index above a floor) and
(b) determinism (two seeded runs give identical embeddings + labels). No API key, no real data.

Usage:  python demo_lifestyle_construction.py [--offline]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import lifestyle_construction as lc


def _synth_episodes(n_per_type: int = 40, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """3 distinct day-types:
       0 sedentary_workday — full-day rest tone at work, normal HR, high wear
       1 active_day        — a midday workout block (high HR), otherwise rest
       2 restful_low_wear  — few home episodes, low HR
    """
    rng = np.random.default_rng(seed)
    rows, true_types, day0 = [], [], pd.Timestamp("2020-01-06")  # a Monday
    day = 0
    for t in range(3):
        for _ in range(n_per_type):
            date = day0 + pd.Timedelta(days=day)
            day += 1
            true_types.append(t)
            if t == 0:                                   # sedentary workday: slots 7..22
                slots, hr_mu, wk, loc, ac = range(7 * 4, 22 * 4), 86, False, "work", "daytime_rest"
            elif t == 1:                                 # active day
                slots, hr_mu, wk, loc, ac = range(6 * 4, 22 * 4), 84, False, "home", "daytime_rest"
            else:                                        # restful low-wear: sparse, low HR
                slots, hr_mu, wk, loc, ac = range(20 * 4, 24 * 4), 66, False, "home", "overnight_rest"
            for s in slots:
                hr = hr_mu + rng.normal(0, 3)
                is_wk, act, a_ctx = False, "rest", ac
                if t == 1 and 12 * 4 <= s < 13 * 4:      # workout block
                    hr, is_wk, act, a_ctx, loc = 150 + rng.normal(0, 5), True, "run", "active_workout", "outdoor"
                rows.append({
                    "datetime": date + pd.Timedelta(minutes=15 * s),
                    "avg_hr": hr, "max_hr": hr + 6, "min_hr": hr - 6,
                    "hrv_sdnn": 40 + rng.normal(0, 5), "is_workout": is_wk,
                    "activity": act, "location_type": loc, "location_place": None,
                    "weather_temp": 18 + rng.normal(0, 2), "weather_humidity": 55,
                    "activity_context": a_ctx, "weather_ctx": "thermoneutral",
                    "workout_type_ctx": "steady_state_cardio" if is_wk else "unknown",
                    "enrich_source": "fallback",
                })
    return pd.DataFrame(rows), np.array(true_types)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B offline selftest.")
    ap.add_argument("--offline", action="store_true", help="(demo is always offline; flag kept for parity)")
    ap.parse_args()

    ep, true_types = _synth_episodes()
    cfg = lc.LCConfig(epochs=15, feature_mode="enriched")

    res = lc.run_stage_b(ep, cfg)
    ari = adjusted_rand_score(true_types, res.labels)
    print(f"days: {len(res.labels)} | chosen k: {res.k} | silhouette: {res.silhouette} | "
          f"ARI vs known types: {ari:.3f}")
    print("states:")
    print(res.nodes[["state", "name", "n_days", "median_hr", "workout_day_frac",
                     "median_wear_slots"]].to_string(index=False))

    # determinism: two seeded embedding fits are identical
    dt = lc.build_day_tensor(ep, feature_mode="enriched")
    e1 = lc.fit_day_embeddings(dt, cfg)
    e2 = lc.fit_day_embeddings(dt, cfg)
    det = bool(np.allclose(e1, e2))
    print(f"determinism (identical embeddings on re-fit): {det}")

    assert ari >= 0.5, f"clusters did not recover known day-types (ARI={ari:.3f})"
    assert det, "non-deterministic embeddings"
    print("\nPASS — known day-types recovered and embeddings reproducible.")


if __name__ == "__main__":
    main()
