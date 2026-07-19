"""
E5b — RQ1b / condition C1: raw GPS coordinates (no geocode) -> LLM.

Completes the C0/C1/C2 contrast (plan §9.5). C1 gives the LLM the bare numbers
("latitude 32.87421, longitude -117.22137") in both few-shot and query — NO
semantic place name. Hypothesis: raw coordinates carry little meaning to an LLM,
so C1 should help *far less* than C2. If so, it proves the LLM needs *semantic*
location, not coordinates — the clean publishable contrast.

Reuses the SAME e3_eval_sample. Isolated; writes only results/enrichment_experiment/e5_*.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import ee_common as ee
import ee_enrichers as een

ee.ensure_output_dirs()
KEY = ["uuid", "timestamp"]
N_PER_LOC, N_PER_ACT = 2, 1

gold = pd.read_parquet(ee.DATA_DIR / "e2_gold_labels.parquet").set_index(KEY).sort_index()
feat_df = ee.load_all().set_index(KEY).sort_index()
folds = ee.load_folds()
eval_s = pd.read_parquet(ee.RESULTS_DIR / "e3_eval_sample.parquet")
fold_of = {u: k for k in folds for u in folds[k]["test"]}
eval_s["fold"] = [fold_of[u] for u in eval_s["uuid"]]
eval_sample = eval_s.set_index(KEY).sort_index()
print(f"[1] eval sample: {len(eval_sample)} rows", flush=True)

# --- raw-coordinate cue for eval rows + few-shot picks (no geocoding) ---
absloc = ee.load_all_absolute_location()
pick_idx = set()
for k in folds:
    pick_idx.update(een.fewshot_picks(gold, folds[k]["train"], seed=k,
                                      n_per_loc=N_PER_LOC, n_per_act=N_PER_ACT))
need = eval_sample.index.union(pd.MultiIndex.from_tuples(sorted(pick_idx), names=KEY))
coords_df = absloc.reindex(need)
coord_map = {idx: f"latitude {r.latitude:.5f}, longitude {r.longitude:.5f}"
             for idx, r in coords_df.iterrows() if r.latitude == r.latitude}   # skip NaN
print(f"[2] rows with a raw-coord cue: {len(coord_map)}", flush=True)

# --- C1 LLM run: few-shot AND query carry raw coords (geo_label='GPS coordinates') ---
GEO_LABEL = "GPS coordinates"
shots_c1 = een.fewshot_by_fold(gold, feat_df, folds, geo=coord_map, geo_label=GEO_LABEL,
                               n_per_loc=N_PER_LOC, n_per_act=N_PER_ACT)
enr = een.LLMEnricher(shots=shots_c1, geo=coord_map, geo_label=GEO_LABEL)
t0 = time.time()
pred_c1 = enr.predict(feat_df.loc[eval_sample.index], eval_sample["fold"])
pred_c1["fold"] = eval_sample["fold"].values
pred_c1.reset_index().to_parquet(ee.RESULTS_DIR / "e5_pred_llm_c1.parquet", index=False)
lat = np.array(enr.latencies)
cost = {"n_eval": int(len(pred_c1)), "n_live_calls": int(len(lat)),
        "mean_latency_s": float(lat.mean()) if len(lat) else 0.0,
        "n_coord_cue": len(coord_map), "model": ee.GEMINI_MODEL, "mode": "few-shot+C1rawcoords"}
json.dump(cost, open(ee.RESULTS_DIR / "e5_llm_c1_cost.json", "w"), indent=2)
print(f"[3] C1 LLM done in {time.time()-t0:.0f}s: {cost}", flush=True)

# --- combined C0/C1/C2 per-class location scoring ---
c0 = pd.read_parquet(ee.RESULTS_DIR / "e3_pred_llm.parquet").set_index(KEY).sort_index()
c2 = pd.read_parquet(ee.RESULTS_DIR / "e5_pred_llm_c2.parquet").set_index(KEY).sort_index()
sub = gold.loc[eval_sample.index]
elig = sub["elig_location"].values
y = sub.loc[elig, "gold_location"]
preds = {"c0": c0.loc[y.index, "pred_location"],
         "c1": pred_c1.loc[y.index, "pred_location"],
         "c2": c2.loc[y.index, "pred_location"]}
labels = sorted(y.unique())
support = y.value_counts().reindex(labels).fillna(0).astype(int)

tab = {"class": labels, "support": support.values}
for name, yh in preds.items():
    tab[f"{name}_f1"] = np.round(f1_score(y, yh, labels=labels, average=None, zero_division=0), 4)
    tab[f"{name}_recall"] = np.round(recall_score(y, yh, labels=labels, average=None, zero_division=0), 4)
out = pd.DataFrame(tab).sort_values("support", ascending=False).reset_index(drop=True)
out.to_csv(ee.RESULTS_DIR / "e5_all_conditions_location_per_class.csv", index=False)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("\n=== e5_all_conditions_location_per_class (LLM: C0 / C1 / C2) ===")
print(out.to_string(index=False))
print("\nmacro-F1 location (ML ref .344):")
for name, yh in preds.items():
    mac = f1_score(y, yh, average="macro", zero_division=0)
    committed = (yh != "unknown"); gnone = (y == "unknown")
    oc = (yh[gnone.values] != "unknown").mean() if gnone.any() else float("nan")
    print(f"  LLM-{name.upper()}: macro-F1={mac:.3f}  coverage={committed.mean():.3f}  over_conf={oc:.3f}")
print("\n[4] saved e5_all_conditions_location_per_class.csv + e5_pred_llm_c1.parquet", flush=True)
