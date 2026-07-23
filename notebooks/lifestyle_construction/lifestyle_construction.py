"""
lifestyle_construction.py — Stage B of the Lifestyle Construction track (p3/9 bottom half).

Builds the missing "Personal Lifestyle Semantic Construction -> Lifestyle KG -> lifestyle map"
layer from the subject's enriched episode timeline (Stage A output), using a self-supervised
Transformer (no lifestyle labels exist on an n=1 subject).

Pipeline:
  B1  build_day_tensor        one DAY = its episodes in 96 fixed 15-min slots -> per-slot feature
                              vector (continuous z-scored HR/HRV/weather + is_workout + one-hot
                              categoricals). `feature_mode` = 'raw' | 'enriched' for the ablation.
      LifestyleDayEncoder     masked-episode reconstruction (MAE-style, self-supervised) -> a
      fit_day_embeddings      day embedding per day (mean-pooled encoder output).
  B2  cluster_lifestyle_states  KMeans on day embeddings (K by silhouette) -> day-type STATES;
      describe_states           code-measured attributes per state; name_states (rule/LLM).
      build_lifestyle_kg        nodes = states(+attrs), edges = day-to-day transition matrix.
  B3  build_lifestyle_map       state mix + weekly rhythm + temporal drift + subject priors.

Honest evaluation (no ground truth): proxy_alignment (clusters vs DERIVABLE structure never used as
a target — month/season, is_workout-day; weekday reported but flagged dependent) + aggregate_baseline
(does the Transformer beat a hand-crafted daily aggregate?) + the raw-vs-enriched ablation.

Guardrails: descriptive only; seeded/CPU/deterministic; isolated track; English identifiers;
reuses ee_transformer patterns and the frozen Task-1 vocab via Stage A.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

SEED = 0
DEVICE = "cpu"
SLOTS_PER_DAY = 96                      # 24h * 4 (15-min)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
RESULTS_DIR = os.path.join(_REPO_ROOT, "results", "lifestyle_construction")

CONT_COLS = ["avg_hr", "max_hr", "min_hr", "hrv_sdnn", "weather_temp", "weather_humidity"]
# fixed category vocabularies so raw/enriched tensors have stable, comparable dims
CAT_VOCAB = {
    "activity": ["rest", "sleep", "walk", "row", "cycle", "run", "strength", "unknown"],
    "location_type": ["home", "work", "outdoor", "park", "water", "unknown"],
    "activity_context": ["overnight_rest", "daytime_rest", "evening_rest", "light_activity",
                         "active_workout", "recovery", "unknown"],
    "weather_ctx": ["thermoneutral", "cold_stress", "heat_stress", "high_humidity",
                    "extreme_heatwave", "unknown"],
}
CAT_RAW = ["activity", "location_type"]
CAT_ENRICHED = ["activity", "location_type", "activity_context", "weather_ctx"]


@dataclass
class LCConfig:
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 96
    dropout: float = 0.1
    epochs: int = 12
    batch_days: int = 64
    lr: float = 1e-3
    mask_frac: float = 0.15
    emb_dim: int = 16            # final day-embedding size (projection of pooled encoder output)
    seed: int = SEED
    feature_mode: str = "enriched"   # 'raw' | 'enriched'


# --------------------------------------------------------------------------- #
# B1 — day tensor
# --------------------------------------------------------------------------- #
@dataclass
class DayTensor:
    X: np.ndarray                     # (n_days, 96, F)
    mask: np.ndarray                  # (n_days, 96) True = real, False = pad
    dates: np.ndarray                 # (n_days,) date objects
    cont_slices: tuple                # (start, end) of continuous block in F
    cat_layout: list                  # [(name, start, end, n_classes), ...] one-hot blocks
    aux: pd.DataFrame                 # per-day derived proxies (dow, month, workout_day, wear...)
    feature_mode: str


def _cat_list(feature_mode: str) -> list[str]:
    return CAT_ENRICHED if feature_mode == "enriched" else CAT_RAW


def build_day_tensor(ep: pd.DataFrame, feature_mode: str = "enriched") -> DayTensor:
    """Lay each day's episodes into 96 slots and build per-slot feature vectors. Continuous
    columns are z-scored over real episodes; categoricals are one-hot over fixed vocabularies."""
    ep = ep.copy()
    ep["date"] = ep["datetime"].dt.date
    ep["slot"] = ep["datetime"].dt.hour * 4 + ep["datetime"].dt.minute // 15

    cats = _cat_list(feature_mode)
    for c in cats:
        if c not in ep.columns:
            ep[c] = "unknown"
        ep[c] = ep[c].where(ep[c].isin(CAT_VOCAB[c]), "unknown")

    # continuous: median-impute then standardize (fit on all real episodes)
    cont = ep[CONT_COLS].astype(float)
    med = cont.median()
    cont = cont.fillna(med)
    mu, sd = cont.mean(), cont.std().replace(0, 1.0)
    cont_z = ((cont - mu) / sd).to_numpy(dtype=np.float32)
    is_wk = ep["is_workout"].to_numpy(dtype=np.float32)[:, None]

    # one-hot blocks
    cat_arrays, cat_layout, col = [], [], len(CONT_COLS) + 1  # +1 for is_workout
    cont_slices = (0, len(CONT_COLS) + 1)                     # continuous + is_workout treated cont
    for c in cats:
        vocab = CAT_VOCAB[c]
        idx = ep[c].map({v: i for i, v in enumerate(vocab)}).to_numpy()
        oh = np.zeros((len(ep), len(vocab)), dtype=np.float32)
        oh[np.arange(len(ep)), idx] = 1.0
        cat_arrays.append(oh)
        cat_layout.append((c, col, col + len(vocab), len(vocab)))
        col += len(vocab)

    feat = np.concatenate([cont_z, is_wk] + cat_arrays, axis=1).astype(np.float32)
    F = feat.shape[1]

    dates = np.array(sorted(ep["date"].unique()))
    date_to_i = {d: i for i, d in enumerate(dates)}
    X = np.zeros((len(dates), SLOTS_PER_DAY, F), dtype=np.float32)
    mask = np.zeros((len(dates), SLOTS_PER_DAY), dtype=bool)
    di = ep["date"].map(date_to_i).to_numpy()
    sl = ep["slot"].to_numpy()
    for r in range(len(ep)):
        X[di[r], sl[r]] = feat[r]
        mask[di[r], sl[r]] = True

    # per-day derived proxies (NOT fed to the encoder as targets)
    g = ep.groupby("date")
    aux = pd.DataFrame({"date": dates})
    aux["dow"] = pd.to_datetime(aux["date"]).dt.weekday
    aux["is_weekday"] = aux["dow"] < 5
    aux["month"] = pd.to_datetime(aux["date"]).dt.month
    aux["season"] = (aux["month"] % 12) // 3          # 0=summer(DJF) .. Southern hemisphere-agnostic
    aux["workout_day"] = aux["date"].map(g["is_workout"].any()).astype(bool).values
    aux["wear_slots"] = aux["date"].map(g.size()).values
    aux["mean_hr"] = aux["date"].map(g["avg_hr"].mean()).values
    return DayTensor(X=X, mask=mask, dates=dates, cont_slices=cont_slices,
                     cat_layout=cat_layout, aux=aux, feature_mode=feature_mode)


# --------------------------------------------------------------------------- #
# B1 — self-supervised masked-reconstruction encoder
# --------------------------------------------------------------------------- #
class LifestyleDayEncoder(nn.Module):
    def __init__(self, n_features: int, cfg: LCConfig, cont_slices: tuple, cat_layout: list):
        super().__init__()
        self.cont_slices = cont_slices
        self.cat_layout = cat_layout
        self.embed = nn.Linear(n_features, cfg.d_model)
        self.mask_token = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, SLOTS_PER_DAY, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        # reconstruction heads
        n_cont = cont_slices[1] - cont_slices[0]
        self.cont_head = nn.Linear(cfg.d_model, n_cont)
        self.cat_heads = nn.ModuleList([nn.Linear(cfg.d_model, n) for (_, _, _, n) in cat_layout])
        self.project = nn.Linear(cfg.d_model, cfg.emb_dim)

    def encode(self, x, pad_mask, mask_slots=None):
        h = self.embed(x)
        if mask_slots is not None:
            h = torch.where(mask_slots.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        h = h + self.pos
        return self.encoder(h, src_key_padding_mask=~pad_mask)   # pad_mask True=real -> invert

    def forward(self, x, pad_mask, mask_slots):
        h = self.encode(x, pad_mask, mask_slots)
        return self.cont_head(h), [head(h) for head in self.cat_heads]

    def day_embedding(self, x, pad_mask):
        """Mean-pool real-slot encoder outputs (no masking) -> normalized day embedding."""
        h = self.norm(self.encode(x, pad_mask))
        w = pad_mask.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.project(pooled)


def fit_day_embeddings(dt: DayTensor, cfg: LCConfig) -> np.ndarray:
    """Train the masked-reconstruction encoder, then return one embedding per day. Deterministic."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    X = torch.tensor(dt.X)
    M = torch.tensor(dt.mask)
    cs0, cs1 = dt.cont_slices
    cat_targets = [(a, b) for (_, a, b, _) in dt.cat_layout]

    model = LifestyleDayEncoder(dt.X.shape[2], cfg, dt.cont_slices, dt.cat_layout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")
    n = len(X)
    gen = torch.Generator().manual_seed(cfg.seed)

    model.train()
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_days):
            b = perm[i:i + cfg.batch_days]
            xb, mb = X[b], M[b]
            # mask ~mask_frac of REAL slots
            rnd = torch.rand(xb.shape[:2], generator=gen)
            mask_slots = mb & (rnd < cfg.mask_frac)
            if not mask_slots.any():
                continue
            cont_pred, cat_preds = model(xb, mb, mask_slots)
            sel = mask_slots
            # continuous recon (only masked real slots)
            cont_t = xb[..., cs0:cs1]
            loss = mse(cont_pred[sel], cont_t[sel]).mean()
            # categorical recon
            for (a, b2), head_pred in zip(cat_targets, cat_preds):
                tgt = xb[..., a:b2].argmax(-1)
                loss = loss + ce(head_pred[sel], tgt[sel]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, n, 256):
            embs.append(model.day_embedding(X[i:i + 256], M[i:i + 256]).cpu().numpy())
    return np.concatenate(embs).astype(np.float32)


# --------------------------------------------------------------------------- #
# B2 — cluster + describe + KG
# --------------------------------------------------------------------------- #
def cluster_lifestyle_states(emb: np.ndarray, k_range=range(3, 9), seed: int = SEED):
    """KMeans; choose K by silhouette. Returns (labels, k, silhouette)."""
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(emb)
        sil = silhouette_score(emb, km.labels_) if k < len(emb) else -1
        if best is None or sil > best[2]:
            best = (km.labels_, k, sil)
    return best


def describe_states(labels: np.ndarray, aux: pd.DataFrame) -> pd.DataFrame:
    """Code-measured attributes per state (the 'measure' half; naming is separate)."""
    a = aux.copy()
    a["state"] = labels
    rows = []
    for s, grp in a.groupby("state"):
        rows.append({
            "state": int(s), "n_days": int(len(grp)),
            "median_hr": round(float(grp["mean_hr"].median()), 1),
            "weekday_frac": round(float(grp["is_weekday"].mean()), 2),
            "workout_day_frac": round(float(grp["workout_day"].mean()), 2),
            "median_wear_slots": int(grp["wear_slots"].median()),
            "top_month": int(grp["month"].mode().iloc[0]) if len(grp) else -1,
        })
    return pd.DataFrame(rows)


def name_states(desc: pd.DataFrame) -> dict[int, str]:
    """Rule-based descriptor per state (LLM naming can layer on later, offline-safe default)."""
    names = {}
    for _, r in desc.iterrows():
        if r["workout_day_frac"] >= 0.5:
            base = "active"
        elif r["median_wear_slots"] < desc["median_wear_slots"].median():
            base = "low_wear"
        elif r["median_hr"] >= desc["median_hr"].median():
            base = "high_tone"
        else:
            base = "restful"
        day = "weekday" if r["weekday_frac"] >= 0.6 else ("weekend" if r["weekday_frac"] <= 0.4
                                                          else "mixed")
        names[int(r["state"])] = f"{base}_{day}"
    return names


def build_lifestyle_kg(labels: np.ndarray, desc: pd.DataFrame, names: dict) -> tuple:
    """Nodes = states(+attrs+name); edges = day-to-day transition counts between states."""
    nodes = desc.copy()
    nodes["name"] = nodes["state"].map(names)
    trans = {}
    for a, b in zip(labels[:-1], labels[1:]):
        trans[(int(a), int(b))] = trans.get((int(a), int(b)), 0) + 1
    edges = pd.DataFrame([{"src": a, "dst": b, "count": c} for (a, b), c in trans.items()])
    return nodes, edges.sort_values("count", ascending=False).reset_index(drop=True)


def build_lifestyle_map(labels, desc, names, aux, priors: dict | None = None) -> dict:
    """State mix + weekly rhythm + temporal drift + subject priors."""
    a = aux.copy()
    a["state"] = labels
    a["name"] = a["state"].map(names)
    a["year"] = pd.to_datetime(a["date"]).dt.year
    weekly = (a.groupby(["name", "dow"]).size().unstack(fill_value=0))
    drift = (a.groupby(["year", "name"]).size().unstack(fill_value=0))
    return {
        "n_days": int(len(a)),
        "state_distribution": {k: int(v) for k, v in a["name"].value_counts().items()},
        "weekly_rhythm": weekly.to_dict(orient="index"),
        "temporal_drift_by_year": drift.to_dict(orient="index"),
        "subject_priors": priors or {},
    }


# --------------------------------------------------------------------------- #
# Honest evaluation
# --------------------------------------------------------------------------- #
def proxy_alignment(labels: np.ndarray, aux: pd.DataFrame) -> dict:
    """AMI between day-clusters and DERIVABLE proxies never used as a target.
    month/workout_day are clean (not encoder targets); is_weekday reported but flagged dependent
    (time-of-day is implicitly available to the encoder via slot positions)."""
    return {
        "month_ami": round(adjusted_mutual_info_score(aux["month"], labels), 4),
        "season_ami": round(adjusted_mutual_info_score(aux["season"], labels), 4),
        "workout_day_ami": round(adjusted_mutual_info_score(aux["workout_day"], labels), 4),
        "weekday_ami_DEPENDENT": round(adjusted_mutual_info_score(aux["is_weekday"], labels), 4),
    }


def aggregate_baseline(ep: pd.DataFrame, aux: pd.DataFrame, k: int, seed: int = SEED):
    """Cluster a hand-crafted daily aggregate (no Transformer) -> labels for the same proxy test."""
    ep = ep.copy()
    ep["date"] = ep["datetime"].dt.date
    g = ep.groupby("date")

    def _entropy(x):
        h, _ = np.histogram(x.dropna(), bins=10)
        p = h / h.sum() if h.sum() else h
        p = p[p > 0]
        return float(-(p * np.log(p)).sum()) if len(p) else 0.0

    feat = pd.DataFrame({
        "mean_hr": g["avg_hr"].mean(), "std_hr": g["avg_hr"].std().fillna(0),
        "active_frac": g["is_workout"].mean(), "wear": g.size(),
        "mean_temp": g["weather_temp"].mean(), "hr_entropy": g["avg_hr"].apply(_entropy),
    }).reindex(aux["date"]).fillna(0.0)
    z = (feat - feat.mean()) / feat.std().replace(0, 1.0)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(z.to_numpy()).labels_
    return labels


@dataclass
class LifestyleResult:
    embeddings: np.ndarray
    labels: np.ndarray
    k: int
    silhouette: float
    nodes: pd.DataFrame
    edges: pd.DataFrame
    lifestyle_map: dict
    proxy: dict
    feature_mode: str = "enriched"
    extra: dict = field(default_factory=dict)


def run_stage_b(ep: pd.DataFrame, cfg: LCConfig, priors: dict | None = None) -> LifestyleResult:
    """B1->B3 for one feature_mode. Returns embeddings, states, KG, map, proxy alignment."""
    dt = build_day_tensor(ep, feature_mode=cfg.feature_mode)
    emb = fit_day_embeddings(dt, cfg)
    labels, k, sil = cluster_lifestyle_states(emb, seed=cfg.seed)
    desc = describe_states(labels, dt.aux)
    names = name_states(desc)
    nodes, edges = build_lifestyle_kg(labels, desc, names)
    lmap = build_lifestyle_map(labels, desc, names, dt.aux, priors)
    proxy = proxy_alignment(labels, dt.aux)
    return LifestyleResult(embeddings=emb, labels=labels, k=k, silhouette=round(float(sil), 4),
                           nodes=nodes, edges=edges, lifestyle_map=lmap, proxy=proxy,
                           feature_mode=cfg.feature_mode, extra={"aux": dt.aux})


def run_and_save(ep: pd.DataFrame, priors: dict | None = None, epochs: int = 12,
                 results_dir: str = RESULTS_DIR) -> dict:
    """End-to-end Stage B for BOTH feature modes + aggregate baseline; persist the enriched
    embeddings/KG/map + the honest proxy-eval table. Shared by L1_lifestyle.py and File 3.
    Returns {'primary': LifestyleResult(enriched), 'eval': DataFrame, 'baseline_labels': ndarray}."""
    os.makedirs(results_dir, exist_ok=True)
    results = {}
    for mode in ("enriched", "raw"):
        res = run_stage_b(ep, LCConfig(epochs=epochs, feature_mode=mode), priors=priors)
        base_labels = aggregate_baseline(ep, res.extra["aux"], k=res.k)
        results[mode] = (res, base_labels, proxy_alignment(base_labels, res.extra["aux"]))

    res_e, base_labels_e, base_proxy_e = results["enriched"]
    aux = res_e.extra["aux"]
    emb_df = pd.DataFrame(res_e.embeddings,
                          columns=[f"e{i}" for i in range(res_e.embeddings.shape[1])])
    emb_df.insert(0, "date", aux["date"].values)
    emb_df["state"] = res_e.labels
    emb_df["state_name"] = emb_df["state"].map(dict(zip(res_e.nodes["state"], res_e.nodes["name"])))
    emb_df.to_parquet(os.path.join(results_dir, "day_embeddings.parquet"), index=False)
    res_e.nodes.to_csv(os.path.join(results_dir, "lifestyle_kg_nodes.csv"), index=False)
    res_e.edges.to_csv(os.path.join(results_dir, "lifestyle_kg_edges.csv"), index=False)
    json.dump(res_e.lifestyle_map, open(os.path.join(results_dir, "lifestyle_map.json"), "w"),
              indent=2, default=str)

    eval_rows = []
    for mode in ("enriched", "raw"):
        res, _bl, _bp = results[mode]
        eval_rows.append({"arm": f"transformer_{mode}", "k": res.k, "silhouette": res.silhouette,
                          **res.proxy})
        if mode == "enriched":
            eval_rows.append({"arm": "aggregate_baseline", "k": res.k, "silhouette": np.nan,
                              **base_proxy_e})
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(os.path.join(results_dir, "l1_proxy_eval.csv"), index=False)
    json.dump({"config_epochs": epochs, "eval": eval_rows},
              open(os.path.join(results_dir, "l1_meta.json"), "w"), indent=2, default=str)
    return {"primary": res_e, "eval": eval_df, "baseline_labels": base_labels_e}
