"""
demo_location_context.py - end-to-end run of the location_context module on the
REAL project data. Reproducible and isolated.

What it does
------------
1. parse every workout GPX -> GPS points,
2. cluster + reverse-geocode each cluster centroid (a handful of live calls; all
   cached under data/location_context/geocode_cache/ so a rerun is instant),
3. classify each cluster into the controlled location vocab, derive home_climate,
4. re-attach an UPGRADED location_type (+ new location_place) onto
   behavioral_episodes and show the before/after,
5. write results/location_context/{cluster_locations.csv, episodes_with_place.csv,
   home_climate.json}.

Run:
    cd notebooks/location_context
    python demo_location_context.py            # geocodes centroids live (cached after)
    python demo_location_context.py --offline  # cache-only; no network, never crashes
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import location_context as lc   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main(online: bool) -> None:
    lc.ensure_dirs()
    print(f"location_context demo  (online={online})")
    print("=" * 64)

    # --- 1. GPS ---
    df_gps = lc.load_all_gps()
    print(f"[1] GPS points parsed: {len(df_gps):,} "
          f"(from {lc.GPX_DIR.split(os.sep)[-1]})")
    if not len(df_gps):
        print("    no GPS -> everything stays location_type='unknown' (not an error).")

    # --- 2+3. cluster -> geocode -> classify ---
    loc_table = lc.build_location_table(df_gps, online=online)
    print(f"[2] location clusters: {len(loc_table)}")
    if len(loc_table):
        show = loc_table[["cluster", "n_points", "is_home_region",
                          "location_type", "location_place", "osm_category", "osm_type"]]
        print(show.to_string(index=False))

    # --- 4. home_climate (uses cached hourly weather from File 2 if present) ---
    wx_path = os.path.join(lc.PROC_DIR, "weather_hourly.parquet")
    df_weather = pd.read_parquet(wx_path) if os.path.exists(wx_path) else None
    home = lc.derive_home_climate(loc_table, df_weather)
    print("\n[3] home_climate:")
    print(f"    place = {home.place!r}  coord = {home.coord}")
    print(f"    band  = {home.band}  (median {home.temp_median}°C, "
          f"p10 {home.temp_p10}, p90 {home.temp_p90})")

    # --- 5. re-attach to episodes & compare before/after ---
    ep_path = os.path.join(lc.PROC_DIR, "behavioral_episodes.parquet")
    if os.path.exists(ep_path):
        ep = pd.read_parquet(ep_path)
        before = ep["location_type"].value_counts(dropna=False)
        ep2 = lc.attach_location(ep, df_gps, loc_table)
        after = ep2["location_type"].value_counts(dropna=False)
        cmp = pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int)
        print("\n[4] episode location_type  (before = File 2 rule, after = geocoded):")
        print(cmp.to_string())
        named = ep2["location_place"].notna().sum()
        print(f"    episodes now carrying a real place name: {named:,}")

        out_ep = os.path.join(lc.RESULTS_DIR, "episodes_with_place.parquet")
        keep = ["node_id", "datetime", "is_workout", "activity",
                "location_type", "location_place"]
        ep2[[c for c in keep if c in ep2.columns]].to_parquet(out_ep, index=False)
        print(f"    wrote {os.path.relpath(out_ep, lc.REPO_ROOT)}")

    # --- 6. persist the small artefacts ---
    if len(loc_table):
        out_lt = os.path.join(lc.RESULTS_DIR, "cluster_locations.csv")
        loc_table.to_csv(out_lt, index=False)
        print(f"\n[5] wrote {os.path.relpath(out_lt, lc.REPO_ROOT)}")
    out_hc = os.path.join(lc.RESULTS_DIR, "home_climate.json")
    with open(out_hc, "w", encoding="utf-8") as fh:
        json.dump(home.__dict__, fh, indent=2, default=str)
    print(f"    wrote {os.path.relpath(out_hc, lc.REPO_ROOT)}")
    print("\ndone.")


if __name__ == "__main__":
    main(online="--offline" not in sys.argv)
