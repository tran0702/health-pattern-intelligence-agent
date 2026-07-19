"""
justify_gnn_real.py — the GNN-justification experiment on the subject's REAL
episodes (from the parsed Apple Health data), not synthetic ones.

Real structure (HR distribution, activity mix, temporal gaps) + synthetic anomaly
injection (the only way to get ground truth on unlabeled real data, exactly as
File 4 does). Reuses the pieces from justify_gnn.py; the only change is the episode
source and a descriptive "real anomalies vs baseline" step.

    python justify_gnn_real.py            # baseline from your real context (LLM if key)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import global_baseline as gb                # noqa: E402
import context_providers as cp              # noqa: E402
import justify_gnn as jg                     # noqa: E402  (reuse inject/build_graph/run_gcn/...)

N_RECENT = 3000     # cap for a dense-adjacency GCN on CPU

_ACT = {"Running": "run", "Walking": "walk", "Cycling": "cycle", "Hiking": "walk",
        "Rowing": "other", "TraditionalStrengthTraining": "strength",
        "FunctionalStrengthTraining": "strength", "HighIntensityIntervalTraining": "run"}


VIG = {"run", "cycle", "row", "other", "strength"}   # vigorous / real-exertion activities


def build_real_episodes(frames, n_recent=N_RECENT):
    """Real episodes with is_workout + activity + hour_of_day, choosing the contiguous
    block that has BOTH sleep AND vigorous windows (so the sleep->max-effort transition
    can be tested for real). Prefers File-2 behavioral_episodes; falls back to
    hr_features+workouts if File 2 hasn't run."""
    ep_full = frames.get("episodes")
    if ep_full is not None and "is_workout" in ep_full.columns:
        ep = ep_full.sort_values("datetime").reset_index(drop=True).copy()
        ep["activity"] = ep["activity"].replace({"row": "other"})       # rowing = vigorous
    else:                                                                # fallback: derive
        hf = frames["hr_features"].copy().sort_values("datetime").reset_index(drop=True)
        hf["win_start"] = hf["datetime"]; hf["win_end"] = hf["datetime"] + pd.Timedelta("15min")
        hf["is_workout"] = False
        wk = frames.get("workouts")
        if wk is not None:
            wk = wk.dropna(subset=["start_time", "end_time"])
            for s, e in zip(wk["start_time"], wk["end_time"]):
                hf.loc[(hf["win_start"] < e) & (hf["win_end"] > s), "is_workout"] = True
        hf["hour_of_day2"] = hf["datetime"].dt.hour
        hf["activity"] = np.where((hf["hour_of_day2"] < 6) & (~hf["is_workout"]), "sleep", "rest")
        if wk is not None:
            for typ, s, e in zip(wk["type"], wk["start_time"], wk["end_time"]):
                a = _ACT.get(str(typ).replace("HKWorkoutActivityType", ""), "other")
                hf.loc[(hf["win_start"] < e) & (hf["win_end"] > s), "activity"] = a
        ep = hf

    ep["hour_of_day"] = pd.to_datetime(ep["datetime"]).dt.hour
    # best-both contiguous block: needs sleep AND (rarer) vigorous windows present
    is_s = (ep["activity"] == "sleep").to_numpy().astype(int)
    is_v = ep["activity"].isin(VIG).to_numpy().astype(int)
    if len(ep) > n_recent:
        cs = np.concatenate([[0], np.cumsum(is_s)]); cv = np.concatenate([[0], np.cumsum(is_v)])
        sl = cs[n_recent:] - cs[:-n_recent]; vg = cv[n_recent:] - cv[:-n_recent]
        bs = int(np.argmax(np.minimum(sl, 20 * vg)))                    # require both
        ep = ep.iloc[bs:bs + n_recent].copy().reset_index(drop=True)
    else:
        ep = ep.reset_index(drop=True)
    if getattr(ep["datetime"].dt, "tz", None) is not None:
        ep["datetime"] = ep["datetime"].dt.tz_localize(None)            # graph math wants tz-naive
    ep["day"] = pd.factorize(ep["datetime"].dt.date)[0]
    return ep


def main():
    frames = cp.load_frames()
    if "hr_features" not in frames:
        print("No data/processed/hr_features.parquet — run File 1 first."); return

    # --- real subject context -> real cohort baseline (approach a) ---
    ctx = cp.build_subject_context(frames).context
    base = gb.establish_baseline(ctx, source="auto")
    print(f"real context: {ctx.model_dump()}")
    print(f"baseline source={base.source}  resting={base.resting_hr.low:.0f}-{base.resting_hr.high:.0f} "
          f"sleep={base.sleep_hr.low:.0f}-{base.sleep_hr.high:.0f} vigorous="
          f"{base.vigorous_activity_hr.low:.0f}-{base.vigorous_activity_hr.high:.0f}")

    ep = build_real_episodes(frames)
    span = f"{ep['datetime'].min():%Y-%m-%d} -> {ep['datetime'].max():%Y-%m-%d}"
    print(f"\nreal episodes: {len(ep)} windows ({span})")
    print("activity mix:", ep["activity"].value_counts().to_dict())

    # --- descriptive: REAL windows outside the cohort baseline (no injection) ---
    real_scored = gb.detect_against_baseline(ep, base, flag_at=1.0)
    n_real = int(real_scored["is_anomaly"].sum())
    print(f"\nREAL windows flagged by baseline (deviation>=1): {n_real}/{len(ep)} "
          f"({100*n_real/len(ep):.1f}%)  [descriptive — no ground truth here]")

    # --- inject known anomalies into the REAL stream (File-4 style) ---
    ep2, y_spike, y_phase = jg.inject(ep.copy())
    n = len(ep2)
    print(f"injected: spike={y_spike.sum()}  phase={y_phase.sum()}")

    scored = gb.detect_against_baseline(ep2, base, flag_at=1.0)
    base_node = scored["deviation"].fillna(0.0).to_numpy()

    th = ep2["hour_of_day"].to_numpy()
    act_oh = pd.get_dummies(ep2["activity"]).astype(float).to_numpy()
    feat = np.column_stack([base_node, ep2["avg_hr"].to_numpy(),
                            np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24), act_oh])
    feat = StandardScaler().fit_transform(feat)

    edge_index = jg.build_graph(ep2, feat)
    sleep_nodes = ep2.index[ep2["activity"] == "sleep"].to_numpy()
    # REAL sleep->max-effort transition: high-HR endpoints are actual vigorous windows
    hi_nodes = ep2.index[ep2["activity"].isin(VIG)].to_numpy()
    how_hi = f"vigorous windows ({len(hi_nodes)})"
    if len(hi_nodes) < 8:                                   # fallback if too few workouts
        hi_nodes = ep2.index[ep2["avg_hr"] > ep2["avg_hr"].quantile(0.97)].to_numpy()
        how_hi = "high-HR windows (no vigorous available)"
    n_bad = 150
    print(f"transition endpoints (hi) = {how_hi}")
    bad = np.array([jg.RNG.choice(sleep_nodes, n_bad), jg.RNG.choice(hi_nodes, n_bad)])
    all_edges = np.concatenate([edge_index, bad], axis=1)
    y_edge = np.concatenate([np.zeros(edge_index.shape[1], bool), np.ones(n_bad, bool)])
    print(f"graph edges={edge_index.shape[1]}  injected transition edges={n_bad}")

    z, gnn_node = jg.run_gcn(feat, all_edges)
    gnn_edge = jg.edge_bce_score(z, all_edges)
    base_edge = base_node[all_edges[0]] + base_node[all_edges[1]]

    clean = ~(y_spike | y_phase)
    rows = []
    m = clean | y_spike
    rows.append(jg.scores_row("SPIKE (node)", y_spike[m], base_node[m], gnn_node[m]))
    m = clean | y_phase
    rows.append(jg.scores_row("PHASE (node)", y_phase[m], base_node[m], gnn_node[m]))
    rows.append(jg.scores_row("TRANSITION (edge)", y_edge, base_edge, gnn_edge))

    df = pd.DataFrame(rows)
    print("\n=== REAL episodes: BASELINE (a) vs GCN-DOMINANT, per anomaly type ===\n")
    print(df.to_string(index=False))

    gb.ensure_dirs()
    out = gb.RESULTS_DIR / "gnn_justification_real.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out.relative_to(gb.REPO_ROOT)}")
    print("\n--- VERDICT ---")
    for r in rows:
        who = ("GNN" if r["GNN_PRAUC"] > r["baseline_PRAUC"] + 0.03 else
               "baseline" if r["baseline_PRAUC"] > r["GNN_PRAUC"] + 0.03 else "tie")
        print(f"  {r['anomaly_type']:18s} winner={who:9s} "
              f"(PR-AUC base={r['baseline_PRAUC']} vs GNN={r['GNN_PRAUC']})")


if __name__ == "__main__":
    main()
