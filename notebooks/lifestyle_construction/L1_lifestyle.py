"""
L1_lifestyle.py — runner for Stage B on the real (Stage-A enriched) episode timeline.

Runs the self-supervised lifestyle encoder for BOTH feature modes (enriched vs raw — the ablation
that measures whether the LLM/vocab enrichment adds value), clusters each into lifestyle states,
and scores them against derivable proxies plus a hand-crafted daily-aggregate baseline. Saves the
day embeddings, Lifestyle KG, lifestyle map, and an evaluation table.

Usage:
    python L1_lifestyle.py --offline          # names are rule-based; no API key needed
    python L1_lifestyle.py --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import lifestyle_construction as lc

ENRICHED = os.path.join(lc.RESULTS_DIR, "behavioral_episodes_enriched.parquet")
CTX_JSON = os.path.join(lc._REPO_ROOT, "results", "context_baseline", "global_context.json")
CLIMATE_JSON = os.path.join(lc._REPO_ROOT, "results", "location_context", "home_climate.json")


def _load_priors() -> dict:
    priors = {}
    for key, path in [("global_context", CTX_JSON), ("home_climate", CLIMATE_JSON)]:
        if os.path.exists(path):
            try:
                priors[key] = json.load(open(path))
            except Exception:
                pass
    return priors


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage B on real enriched episodes.")
    ap.add_argument("--offline", action="store_true", help="rule-based state names (no LLM)")
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    if not os.path.exists(ENRICHED):
        raise SystemExit(f"missing {ENRICHED} — run L0_enrich.py first")
    ep = pd.read_parquet(ENRICHED)
    priors = _load_priors()
    print(f"episodes: {len(ep):,} | days: {ep['datetime'].dt.date.nunique():,}")

    out = lc.run_and_save(ep, priors=priors, epochs=args.epochs)
    res_e, eval_df = out["primary"], out["eval"]

    print("\n=== proxy-alignment table (higher AMI = recovers more real structure) ===")
    cols = ["arm", "k", "silhouette", "month_ami", "season_ami", "workout_day_ami",
            "weekday_ami_DEPENDENT"]
    print(eval_df[cols].to_string(index=False))
    print("\nlifestyle states (enriched):")
    print(res_e.nodes.to_string(index=False))

    # verification asserts
    assert not np.isnan(res_e.embeddings).any(), "NaN in embeddings"
    print(f"\nOK — wrote day_embeddings / lifestyle_kg_* / lifestyle_map.json / l1_proxy_eval.csv "
          f"to {lc.RESULTS_DIR}")


if __name__ == "__main__":
    main()
