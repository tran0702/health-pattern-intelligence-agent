"""
L0_enrich.py — runner for Stage A (episode_enrichment).

Produces results/lifestyle_construction/behavioral_episodes_enriched.parquet + l0_enrich_meta.json,
then prints a short verification summary. Offline by default is controlled by --offline (no key
needed); with a Gemini key present and without --offline the LLM enriches, falling back
deterministically per signature on any failure.

Usage:
    python L0_enrich.py --offline
    python L0_enrich.py                 # uses Gemini if a key is in .env, else falls back
"""
from __future__ import annotations

import argparse

import episode_enrichment as ee


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A — LLM episode enrichment.")
    ap.add_argument("--offline", action="store_true", help="skip the LLM; deterministic labels only")
    args = ap.parse_args()

    res = ee.run(offline=args.offline)
    m = res.meta
    print(f"vocab source      : {m['vocab_source']}")
    print(f"episodes          : {m['n_episodes']:,}")
    print(f"distinct signatures: {m['n_signatures']}")
    print(f"used LLM          : {m['used_llm']}")
    print(f"enrich source     : {m['enrich_source_episodes']}")
    print(f"personal HR bands : {m['hr_bands']}")
    print(f"location known    : {m['location_semantic_known']:,} episodes")
    print("\nactivity_context distribution:")
    for k, v in m["activity_context_dist"].items():
        print(f"  {k:16s} {v:>7,}")
    print("\nweather_ctx distribution:")
    for k, v in m["weather_ctx_dist"].items():
        print(f"  {k:16s} {v:>7,}")

    # verification asserts
    ep = res.episodes
    assert len(ep) == m["n_episodes"], "row count changed"
    for col in ("activity_context", "weather_ctx", "workout_type_ctx", "location_semantic",
                "enrich_source"):
        assert col in ep.columns, f"missing enriched column {col}"
        assert ep[col].notna().all(), f"NaN in {col}"
    print(f"\nOK — wrote {ee.ENRICHED_OUT}")


if __name__ == "__main__":
    main()
