"""
E6 — Transformer arms (Task 3 / RQ1c): run the self-built Transformer variants across the
official 5-fold user split and save full-test predictions + metadata for E4.

Three variants are compared WITH EACH OTHER (and, in E4, with the ML control + LLM):
  * feature  (A) — FT-Transformer, attention over the 225 features.
  * group    (B) — attention over ~12 sensor-group tokens (cheap on CPU).
  * temporal (C) — attention over TIME (a right-aligned window of recent samples/user).

All three read the SAME feature matrix / gold / folds as E3's ML control, produce one
prediction per sample in the e3_pred_ml schema (uuid, timestamp, pred_<field>, fold), and
are scored the same way. Isolated: reads only data/enrichment_experiment/, writes only
results/enrichment_experiment/e6_*. Does not touch E1-E5 or the 01-04/03b pipeline.

Running is intentionally decoupled from writing — the module (ee_transformer.py) and this
runner are committed first; a later session probes fold-0 timing then runs the full CV.

Usage (examples):
    python E6_transformer.py --variant all
    python E6_transformer.py --variant feature --folds 0            # single-fold timing probe
    python E6_transformer.py --variant temporal --selftest          # determinism check only
    python E6_transformer.py --variant group --train-cap 20000 --epochs 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import ee_common as ee
import ee_transformer as tf

KEY = ["uuid", "timestamp"]
VARIANTS = ("feature", "group", "temporal")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """Feature matrix + gold + folds + feature columns, aligned exactly as in E3."""
    gold = pd.read_parquet(ee.DATA_DIR / "e2_gold_labels.parquet").set_index(KEY).sort_index()
    feat_df = ee.load_all().set_index(KEY).sort_index()
    # same feature set as E3: drop uuid/timestamp (now the index), label_source, and label:*
    features = [c for c in feat_df.columns
                if c != "label_source" and not c.startswith("label:")]
    assert feat_df.index.equals(gold.index), "feature/gold alignment mismatch"
    folds = ee.load_folds()
    return feat_df, gold, folds, features


def macro_f1(pred: pd.DataFrame, gold: pd.DataFrame, idx) -> dict[str, float]:
    """Per-field macro-F1 over eligible rows in `idx` (scored vs real ExtraSensory labels)."""
    sub = gold.loc[idx]
    scores = {}
    for f in ee.FIELDS:
        elig = sub[f"elig_{f}"].to_numpy()
        y = sub.loc[elig, f"gold_{f}"]
        yhat = pred.loc[y.index, f"pred_{f}"]
        scores[f] = round(float(f1_score(y, yhat, average="macro", zero_division=0)), 4)
    return scores


def build_cfg(variant: str, args) -> tf.TFConfig:
    cfg = tf.TFConfig(variant=variant)
    over = {}
    if args.train_cap is not None:
        over["train_cap"] = args.train_cap
    if args.epochs is not None:
        over["epochs"] = args.epochs
    if args.window is not None:
        over["window"] = args.window
    if args.max_gap is not None:
        over["max_gap_s"] = (None if args.max_gap < 0 else args.max_gap)
    from dataclasses import replace
    return replace(cfg, **over) if over else cfg


def run_variant(variant: str, feat_df, gold, folds, features, eval_idx, args) -> None:
    cfg = build_cfg(variant, args)
    which = sorted(folds) if args.folds is None else args.folds
    print(f"\n=== variant={variant}  folds={which}  cfg={asdict(cfg)} ===", flush=True)

    # Per-fold checkpointing: each fold is saved the moment it finishes, so a long run is
    # resumable (re-running skips folds already on disk) and a crash loses at most one fold.
    parts, secs = [], {}
    for k in which:
        ckpt = ee.RESULTS_DIR / f"e6_pred_{variant}_fold{k}.parquet"
        if ckpt.exists() and not args.force:
            print(f"  fold {k}: cached -> {ckpt.name}", flush=True)
            parts.append(pd.read_parquet(ckpt).set_index(KEY).sort_index())
            secs[k] = "cached"
            continue
        t0 = time.time()
        pred_k = tf.run_fold(k, feat_df, gold, folds, features, cfg)
        secs[k] = round(time.time() - t0, 1)
        pred_k.reset_index().to_parquet(ckpt, index=False)   # checkpoint immediately
        parts.append(pred_k)
        print(f"  fold {k}: {secs[k]}s -> saved {ckpt.name}", flush=True)
    pred = pd.concat(parts).sort_index()

    full = macro_f1(pred, gold, pred.index)
    ev = macro_f1(pred, gold, pred.index.intersection(eval_idx)) if eval_idx is not None else None
    meta = {"variant": variant, "config": asdict(cfg), "folds": list(which),
            "seconds_per_fold": secs, "n_pred": int(len(pred)),
            "macro_f1_full_test": full, "macro_f1_eval_sample": ev}
    if variant == "temporal":
        meta["mean_real_window"] = {int(k): round(
            tf.mean_real_window(feat_df, folds, k, cfg), 2) for k in which}

    # only overwrite the canonical prediction file when a FULL 5-fold run was done
    if len(which) == len(folds):
        pred.reset_index().to_parquet(ee.RESULTS_DIR / f"e6_pred_{variant}.parquet", index=False)
        print(f"  saved e6_pred_{variant}.parquet ({len(pred):,} rows)", flush=True)
    else:
        print("  partial-fold run -> predictions NOT saved (probe only)", flush=True)
    json.dump(meta, open(ee.RESULTS_DIR / f"e6_{variant}_meta.json", "w"), indent=2)
    print(f"  macro-F1 full-test: {full}  | eval-sample: {ev}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Transformer variants for E4 (Task 3).")
    ap.add_argument("--variant", default="all", choices=(*VARIANTS, "all"))
    ap.add_argument("--folds", type=lambda s: [int(x) for x in s.split(",")], default=None,
                    help="comma-separated fold ids (default: all 5). Partial runs don't save preds.")
    ap.add_argument("--selftest", action="store_true", help="determinism check on fold 0, then exit")
    ap.add_argument("--train-cap", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--window", type=int, default=None, help="temporal window K")
    ap.add_argument("--max-gap", type=int, default=None, help="temporal max gap seconds (<0 = off)")
    ap.add_argument("--force", action="store_true", help="recompute folds even if a checkpoint exists")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="torch CPU threads")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print(f"torch threads: {torch.get_num_threads()}", flush=True)
    ee.ensure_output_dirs()
    feat_df, gold, folds, features = load_inputs()
    print(f"rows: {len(feat_df):,} | features: {len(features)} | folds: {list(folds)}", flush=True)

    variants = VARIANTS if args.variant == "all" else (args.variant,)

    if args.selftest:
        for v in variants:
            ok = tf.selftest(feat_df, gold, folds, features, variant=v)
            print(f"selftest[{v}]: {'PASS' if ok else 'FAIL'}", flush=True)
        return

    eval_path = ee.RESULTS_DIR / "e3_eval_sample.parquet"
    eval_idx = (pd.read_parquet(eval_path).set_index(KEY).index if eval_path.exists() else None)

    for v in variants:
        run_variant(v, feat_df, gold, folds, features, eval_idx, args)


if __name__ == "__main__":
    main()
