"""
E5 — RQ1b / condition C2: absolute GPS -> reverse-geocoded SEMANTIC location -> LLM.

Question (plan §9.1/§9.5): the LLM was weak at `location`. Is that because it was
starved of a usable location signal, or because it is simply worse than ML? C2
gives the LLM a semantic place cue (a map name + OSM category, e.g. "Cardiff State
Beach Parking (OSM: amenity/parking)") derived from the real coordinate, then asks
the same question. Litmus: does the LLM's `beach` recall move off 0?

Reuses the SAME `e3_eval_sample` so C2 is directly comparable to the frozen C0
baseline (`e3_pred_llm.parquet`). Isolated: reads only data/enrichment_experiment/,
writes only results/enrichment_experiment/e5_*. Does not touch E1-E4 or 01-04/03b.

Note: C2 is deliberately an "LLM + geocoder" hybrid (plan §9.4) — that is the
finding, not a confound.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import ee_common as ee
import ee_enrichers as een
import ee_geocode as geo

ee.ensure_output_dirs()
KEY = ["uuid", "timestamp"]
N_PER_LOC, N_PER_ACT = 2, 1

# --------------------------------------------------------------------------- #
# 1. Load the frozen inputs (gold, features, folds, shared eval sample)
# --------------------------------------------------------------------------- #
gold = pd.read_parquet(ee.DATA_DIR / "e2_gold_labels.parquet").set_index(KEY).sort_index()
feat_df = ee.load_all().set_index(KEY).sort_index()
folds = ee.load_folds()
eval_s = pd.read_parquet(ee.RESULTS_DIR / "e3_eval_sample.parquet")
fold_of = {u: k for k in folds for u in folds[k]["test"]}
eval_s["fold"] = [fold_of[u] for u in eval_s["uuid"]]
eval_sample = eval_s.set_index(KEY).sort_index()
print(f"[1] eval sample: {len(eval_sample)} rows", flush=True)

# --------------------------------------------------------------------------- #
# 2. Coordinates needed = eval rows + the (leakage-free) few-shot example rows
# --------------------------------------------------------------------------- #
absloc = ee.load_all_absolute_location()
pick_idx = set()
for k in folds:
    pick_idx.update(een.fewshot_picks(gold, folds[k]["train"], seed=k,
                                      n_per_loc=N_PER_LOC, n_per_act=N_PER_ACT))
need = eval_sample.index.union(pd.MultiIndex.from_tuples(sorted(pick_idx), names=KEY))
coords_df = absloc.reindex(need)
coords = {idx: (float(r.latitude), float(r.longitude)) for idx, r in coords_df.iterrows()}
n_gps = sum(1 for v in coords.values() if v[0] == v[0])          # not NaN
print(f"[2] coords to resolve: {len(coords)}  (with GPS: {n_gps})", flush=True)

# --------------------------------------------------------------------------- #
# 3. Reverse geocode -> semantic phrase (cached; ~1 req/s to public Nominatim)
# --------------------------------------------------------------------------- #
t0 = time.time()
geo_map = {k: v for k, v in geo.geocode_index(coords).items() if v}
print(f"[3] geocoded in {time.time()-t0:.0f}s; {len(geo_map)} rows carry a place phrase",
      flush=True)

# --------------------------------------------------------------------------- #
# 4. C2 LLM run — few-shot AND query carry the map cue (cached by prompt hash)
# --------------------------------------------------------------------------- #
shots_c2 = een.fewshot_by_fold(gold, feat_df, folds, geo=geo_map,
                               n_per_loc=N_PER_LOC, n_per_act=N_PER_ACT)
enr = een.LLMEnricher(shots=shots_c2, geo=geo_map)
t0 = time.time()
pred_c2 = enr.predict(feat_df.loc[eval_sample.index], eval_sample["fold"])
pred_c2["fold"] = eval_sample["fold"].values
pred_c2.reset_index().to_parquet(ee.RESULTS_DIR / "e5_pred_llm_c2.parquet", index=False)
lat = np.array(enr.latencies)
cost = {"n_eval": int(len(pred_c2)), "n_live_calls": int(len(lat)),
        "mean_latency_s": float(lat.mean()) if len(lat) else 0.0,
        "n_geo_phrase": len(geo_map), "model": ee.GEMINI_MODEL, "mode": "few-shot+C2geo"}
json.dump(cost, open(ee.RESULTS_DIR / "e5_llm_c2_cost.json", "w"), indent=2)
print(f"[4] C2 LLM done in {time.time()-t0:.0f}s: {cost}", flush=True)

# --------------------------------------------------------------------------- #
# 5. Score: per-class location F1, C0 (frozen LLM) vs C2, on eligible eval rows
# --------------------------------------------------------------------------- #
c0 = pd.read_parquet(ee.RESULTS_DIR / "e3_pred_llm.parquet").set_index(KEY).sort_index()
sub = gold.loc[eval_sample.index]
elig = sub["elig_location"].values
y = sub.loc[elig, "gold_location"]
yc0 = c0.loc[y.index, "pred_location"]
yc2 = pred_c2.loc[y.index, "pred_location"]
labels = sorted(y.unique())
support = y.value_counts().reindex(labels).fillna(0).astype(int)


def per_class(yhat, fn):
    return np.round(fn(y, yhat, labels=labels, average=None, zero_division=0), 4)


def n_pred(yhat):
    return yhat.value_counts().reindex(labels).fillna(0).astype(int).values


out = pd.DataFrame({
    "class": labels, "support": support.values,
    "c0_f1": per_class(yc0, f1_score), "c2_f1": per_class(yc2, f1_score),
    "c0_recall": per_class(yc0, recall_score), "c2_recall": per_class(yc2, recall_score),
    "c0_precision": per_class(yc0, precision_score), "c2_precision": per_class(yc2, precision_score),
    "c0_n_pred": n_pred(yc0), "c2_n_pred": n_pred(yc2),
}).sort_values("support", ascending=False).reset_index(drop=True)
out.to_csv(ee.RESULTS_DIR / "e5_c2_location_per_class.csv", index=False)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("\n=== e5_c2_location_per_class (LLM: C0 vs C2) ===")
print(out.to_string(index=False))
mac0 = f1_score(y, yc0, average="macro", zero_division=0)
mac2 = f1_score(y, yc2, average="macro", zero_division=0)
print(f"\nmacro-F1 location  LLM-C0={mac0:.3f}  LLM-C2={mac2:.3f}  (ML ref .344)")
for name, yh in [("C0", yc0), ("C2", yc2)]:
    committed = (yh != "unknown"); gnone = (y == "unknown")
    oc = (yh[gnone.values] != "unknown").mean() if gnone.any() else float("nan")
    print(f"  {name}: coverage={committed.mean():.3f}  over_confidence={oc:.3f}")
print("\n[5] saved e5_c2_location_per_class.csv + e5_pred_llm_c2.parquet", flush=True)
