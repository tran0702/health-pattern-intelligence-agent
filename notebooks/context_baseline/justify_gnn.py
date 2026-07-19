"""
justify_gnn.py — does the GNN earn its keep on top of approach (a)?

The point-wise baseline detector (approach (a)) compares each window's HR to the
cohort-normative band. The question (Asara's Task-3 logic: don't keep two models
without a reason): is there a class of anomaly the GNN catches that the baseline
misses? If yes -> keep the GNN as the complementary RELATIONAL detector. If no ->
drop it.

Experiment (mirrors File 4's synthetic injection, but pits
BASELINE-DEVIATION (a)  vs  GCN-DOMINANT):
  * synthetic 30-day episode stream whose HR is NORMAL wrt the cohort baseline,
  * a Behavioral Knowledge Graph (temporal + similarity + context edges),
  * three injected anomaly types with known labels:
      - SPIKE (node):      +35..55 bpm on rest windows      -> a MAGNITUDE anomaly
      - PHASE (node):      sleep-hour HR raised to ~70 bpm   -> individually near-normal
                           (inside the resting band) but temporally WRONG
      - TRANSITION (edge): sleep-node -> max-effort-node     -> a RELATIONAL anomaly
  * score both detectors per type (ROC-AUC / PR-AUC / best-F1) on
    {that-type positives} vs {clean-normal} only (isolated, no confound).

Self-contained: a pure-torch GCN (no torch_geometric needed). Reuses the
approach-(a) baseline from global_baseline.py. Writes results/context_baseline/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_library import SubjectContext            # noqa: E402
import global_baseline as gb                            # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(7)
torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# 1. Synthetic normal episode stream (all HR inside the cohort baseline bands)
# --------------------------------------------------------------------------- #
def make_episodes(days=30, step_min=30):
    per_day = 24 * 60 // step_min
    rows = []
    t0 = pd.Timestamp("2026-01-01 00:00")
    for d in range(days):
        run_hour = RNG.integers(17, 20) if RNG.random() < 0.6 else None
        for k in range(per_day):
            ts = t0 + pd.Timedelta(days=d, minutes=k * step_min)
            h = ts.hour
            if run_hour is not None and h == run_hour:
                act, hr = "run", RNG.normal(150, 9)          # within vigorous 145-175
            elif 0 <= h < 6:
                if RNG.random() < 0.25:                       # night coverage gaps
                    continue
                act, hr = "sleep", RNG.normal(55, 3)          # within sleep 45-65
            else:
                act, hr = "rest", RNG.normal(70, 5)           # within resting 60-80
            rows.append((ts, d, h, act, round(float(hr), 1)))
    ep = pd.DataFrame(rows, columns=["datetime", "day", "hour_of_day", "activity", "avg_hr"])
    return ep.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Inject the three anomaly types (with ground-truth labels)
# --------------------------------------------------------------------------- #
def inject(ep):
    n = len(ep)
    y_spike = np.zeros(n, bool)
    y_phase = np.zeros(n, bool)

    rest_idx = ep.index[ep["activity"] == "rest"].to_numpy()
    spike = RNG.choice(rest_idx, max(8, int(0.01 * len(rest_idx))), replace=False)
    ep.loc[spike, "avg_hr"] += RNG.uniform(35, 55, len(spike))          # magnitude anomaly
    y_spike[spike] = True

    days = np.sort(ep["day"].unique())
    start = int(RNG.integers(0, len(days) - 6))
    block = set(days[start:start + 5])
    ph = ep.index[(ep["day"].isin(block)) & (ep["activity"] == "sleep")].to_numpy()
    ep.loc[ph, "avg_hr"] = RNG.normal(70, 2, len(ph))     # inside resting band, wrong for sleep
    y_phase[ph] = True
    return ep, y_spike, y_phase


# --------------------------------------------------------------------------- #
# 3. Behavioral Knowledge Graph (temporal + similarity + context edges)
# --------------------------------------------------------------------------- #
def build_graph(ep, feats):
    n = len(ep)
    src, dst = [], []
    # temporal: consecutive windows <= 60 min apart
    gap = ep["datetime"].diff().dt.total_seconds().fillna(1e9).to_numpy()
    for i in range(1, n):
        if gap[i] <= 3600:
            src += [i - 1]; dst += [i]
    # similarity: kNN (k=8) in standardized feature space
    k = 8
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(feats)
    _, nbr = nn_.kneighbors(feats)
    for i in range(n):
        for j in nbr[i, 1:]:
            src += [i]; dst += [int(j)]
    # context: same activity within +/- 2h (capped fan-out)
    for act, grp in ep.groupby("activity"):
        idx = grp.index.to_numpy(); times = grp["datetime"].to_numpy()
        for a in range(len(idx)):
            near = np.where(np.abs((times - times[a]).astype("timedelta64[s]").astype(float)) <= 7200)[0]
            for b in near[:5]:
                if idx[a] != idx[b]:
                    src += [int(idx[a])]; dst += [int(idx[b])]
    return np.array([src, dst])


# --------------------------------------------------------------------------- #
# 4. Pure-torch GCN-DOMINANT (no torch_geometric): D^-1/2 (A+I) D^-1/2 X W
# --------------------------------------------------------------------------- #
def normalized_adj(edge_index, n):
    A = torch.zeros((n, n))
    s, d = edge_index
    A[s, d] = 1.0; A[d, s] = 1.0        # symmetric
    A.fill_diagonal_(1.0)               # self loops
    dinv = A.sum(1).pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    return dinv.view(-1, 1) * A * dinv.view(1, -1)


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.lin = nn.Linear(i, o, bias=False)

    def forward(self, x, A):
        return A @ self.lin(x)


class DominantLite(nn.Module):
    """GCN encoder + GCN attribute decoder + inner-product structure decoder."""
    def __init__(self, d, hid=32, emb=16):
        super().__init__()
        self.e1 = GCNLayer(d, hid); self.e2 = GCNLayer(hid, emb); self.dec = GCNLayer(emb, d)

    def forward(self, x, A):
        h = F.relu(self.e1(x, A)); z = self.e2(h, A); return z, self.dec(z, A)


def run_gcn(feats, edge_index, epochs=200):
    n = feats.shape[0]
    x = torch.tensor(feats, dtype=torch.float)
    A = normalized_adj(torch.tensor(edge_index, dtype=torch.long), n)
    ei = torch.tensor(edge_index, dtype=torch.long)
    m = DominantLite(x.size(1))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-5)
    s, d = ei
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        z, xh = m(x, A)
        pos = (z[s] * z[d]).sum(1)
        ns = torch.randint(0, n, (s.size(0),)); nd = torch.randint(0, n, (s.size(0),))
        neg = (z[ns] * z[nd]).sum(1)
        loss = 0.7 * F.mse_loss(xh, x) + 0.3 * (
            F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) +
            F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        z, xh = m(x, A)
        node_score = ((x - xh) ** 2).mean(1).numpy()
    return z.detach(), node_score


def edge_bce_score(z, edge_index):
    s, d = torch.tensor(edge_index[0]), torch.tensor(edge_index[1])
    logit = (z[s] * z[d]).sum(1)
    return F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction="none").numpy()


# --------------------------------------------------------------------------- #
# 5. Metrics helper — isolated (positives vs clean-normal only)
# --------------------------------------------------------------------------- #
def best_f1(y, score):
    p, r, thr = precision_recall_curve(y, score)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(np.max(f1))


def scores_row(name, y, base, gnn):
    return {
        "anomaly_type": name, "n_pos": int(y.sum()),
        "baseline_ROC": round(roc_auc_score(y, base), 3),
        "GNN_ROC": round(roc_auc_score(y, gnn), 3),
        "baseline_PRAUC": round(average_precision_score(y, base), 3),
        "GNN_PRAUC": round(average_precision_score(y, gnn), 3),
        "baseline_F1": round(best_f1(y, base), 3),
        "GNN_F1": round(best_f1(y, gnn), 3),
    }


def main():
    ctx = SubjectContext(age_band="30_39", sex="male", fitness_level="recreational",
                         goal="endurance", health_conditions=["none"])
    base = gb.establish_baseline(ctx, source="auto")     # LLM if key (cached), else offline
    print(f"cohort baseline source = {base.source}  resting={base.resting_hr.low:.0f}-"
          f"{base.resting_hr.high:.0f}  sleep={base.sleep_hr.low:.0f}-{base.sleep_hr.high:.0f}")

    ep = make_episodes()
    ep, y_spike, y_phase = inject(ep)
    n = len(ep)
    print(f"episodes={n}  spike={y_spike.sum()}  phase={y_phase.sum()}")

    # --- approach (a) point detector: deviation from the cohort baseline ---
    scored = gb.detect_against_baseline(ep, base, flag_at=1.0)
    base_node = scored["deviation"].fillna(0.0).to_numpy()

    # --- node features for the GCN (include the (a) deviation; NO anomaly labels) ---
    th = ep["hour_of_day"].to_numpy()
    act_oh = pd.get_dummies(ep["activity"]).astype(float).to_numpy()
    feat = np.column_stack([
        base_node, ep["avg_hr"].to_numpy(),
        np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24), act_oh])
    feat = StandardScaler().fit_transform(feat)

    edge_index = build_graph(ep, feat)

    # --- inject impossible-transition edges (sleep -> max-effort) ---
    sleep_nodes = ep.index[ep["activity"] == "sleep"].to_numpy()
    hi_nodes = ep.index[ep["avg_hr"] > ep["avg_hr"].quantile(0.97)].to_numpy()
    n_bad = 150
    bad = np.array([RNG.choice(sleep_nodes, n_bad), RNG.choice(hi_nodes, n_bad)])
    all_edges = np.concatenate([edge_index, bad], axis=1)
    y_edge = np.concatenate([np.zeros(edge_index.shape[1], bool), np.ones(n_bad, bool)])
    print(f"graph edges={edge_index.shape[1]}  injected transition edges={n_bad}")

    # --- train GCN on the injected graph, get node + edge scores ---
    z, gnn_node = run_gcn(feat, all_edges)
    gnn_edge = edge_bce_score(z, all_edges)
    base_edge = base_node[all_edges[0]] + base_node[all_edges[1]]   # fair no-graph proxy

    # --- isolated evaluation: each type's positives vs CLEAN-normal only ---
    clean = ~(y_spike | y_phase)
    rows = []
    m = clean | y_spike
    rows.append(scores_row("SPIKE (node)", y_spike[m], base_node[m], gnn_node[m]))
    m = clean | y_phase
    rows.append(scores_row("PHASE (node)", y_phase[m], base_node[m], gnn_node[m]))
    # edges: injected positives vs original (clean) edges
    rows.append(scores_row("TRANSITION (edge)", y_edge, base_edge, gnn_edge))

    df = pd.DataFrame(rows)
    print("\n=== Justification: BASELINE (a) vs GCN-DOMINANT, per anomaly type ===\n")
    print(df.to_string(index=False))

    # concrete miss counts at the operating threshold
    flagged = scored["is_anomaly"].to_numpy()
    print(f"\nBaseline flags at dev>=1: spike {flagged[y_spike].sum()}/{y_spike.sum()}, "
          f"phase {flagged[y_phase].sum()}/{y_phase.sum()} "
          f"(phase HR sits inside the resting band -> baseline is blind).")

    gb.ensure_dirs()
    out = gb.RESULTS_DIR / "gnn_justification.csv"
    df.to_csv(out, index=False)
    print(f"saved {out.relative_to(REPO_ROOT)}")

    # --- verdict ---
    print("\n--- VERDICT ---")
    for r in rows:
        who = "GNN" if r["GNN_PRAUC"] > r["baseline_PRAUC"] + 0.03 else (
            "baseline" if r["baseline_PRAUC"] > r["GNN_PRAUC"] + 0.03 else "tie")
        print(f"  {r['anomaly_type']:18s} winner={who:9s} "
              f"(PR-AUC base={r['baseline_PRAUC']} vs GNN={r['GNN_PRAUC']})")


if __name__ == "__main__":
    main()
