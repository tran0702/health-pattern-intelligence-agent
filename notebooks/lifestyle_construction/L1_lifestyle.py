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

    results = {}
    for mode in ("enriched", "raw"):
        cfg = lc.LCConfig(epochs=args.epochs, feature_mode=mode)
        res = lc.run_stage_b(ep, cfg, priors=priors)
        base_labels = lc.aggregate_baseline(ep, res.extra["aux"], k=res.k)
        base_proxy = lc.proxy_alignment(base_labels, res.extra["aux"])
        results[mode] = (res, base_proxy)
        print(f"\n=== feature_mode={mode} | k={res.k} | silhouette={res.silhouette} ===")
        print("Transformer proxy alignment:", res.proxy)
        print("aggregate baseline proxy   :", base_proxy)

    # primary = enriched: persist embeddings + KG + map
    res_e, base_e = results["enriched"]
    os.makedirs(lc.RESULTS_DIR, exist_ok=True)
    aux = res_e.extra["aux"]
    emb_df = pd.DataFrame(res_e.embeddings,
                          columns=[f"e{i}" for i in range(res_e.embeddings.shape[1])])
    emb_df.insert(0, "date", aux["date"].values)
    emb_df["state"] = res_e.labels
    emb_df["state_name"] = emb_df["state"].map(dict(zip(res_e.nodes["state"], res_e.nodes["name"])))
    emb_df.to_parquet(os.path.join(lc.RESULTS_DIR, "day_embeddings.parquet"), index=False)
    res_e.nodes.to_csv(os.path.join(lc.RESULTS_DIR, "lifestyle_kg_nodes.csv"), index=False)
    res_e.edges.to_csv(os.path.join(lc.RESULTS_DIR, "lifestyle_kg_edges.csv"), index=False)
    json.dump(res_e.lifestyle_map, open(os.path.join(lc.RESULTS_DIR, "lifestyle_map.json"), "w"),
              indent=2, default=str)

    # evaluation table (enriched vs raw vs baseline) — the honest, circularity-safe comparison
    eval_rows = []
    for mode in ("enriched", "raw"):
        res, base_proxy = results[mode]
        eval_rows.append({"arm": f"transformer_{mode}", "k": res.k, "silhouette": res.silhouette,
                          **res.proxy})
        if mode == "enriched":
            eval_rows.append({"arm": "aggregate_baseline", "k": res.k, "silhouette": np.nan,
                              **base_proxy})
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(os.path.join(lc.RESULTS_DIR, "l1_proxy_eval.csv"), index=False)
    json.dump({"config_epochs": args.epochs, "priors_loaded": list(priors),
               "eval": eval_rows}, open(os.path.join(lc.RESULTS_DIR, "l1_meta.json"), "w"),
              indent=2, default=str)

    print("\n=== proxy-alignment table (higher AMI = recovers more real structure) ===")
    cols = ["arm", "k", "silhouette", "month_ami", "season_ami", "workout_day_ami",
            "weekday_ami_DEPENDENT"]
    print(eval_df[cols].to_string(index=False))
    print("\nlifestyle states (enriched):")
    print(res_e.nodes.to_string(index=False))

    # verification asserts
    assert len(emb_df) == aux.shape[0], "day count mismatch"
    assert not np.isnan(res_e.embeddings).any(), "NaN in embeddings"
    print(f"\nOK — wrote day_embeddings / lifestyle_kg_* / lifestyle_map.json / l1_proxy_eval.csv "
          f"to {lc.RESULTS_DIR}")


if __name__ == "__main__":
    main()
