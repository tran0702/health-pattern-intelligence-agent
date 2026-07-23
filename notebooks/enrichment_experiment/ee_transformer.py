"""
ee_transformer.py — the Transformer arm for the Context Enrichment Experiment (Task 3).

A self-built torch feature-token Transformer (FT-Transformer style) that classifies the
three context fields (location / activity / companion) DIRECTLY from ExtraSensory's 225
numeric features. It is the third arm, compared head-to-head with the ML control (E3,
SGDClassifier) and the Gemini LLM (ee_enrichers) on the SAME 5-fold user split + gold
labels (scored in E4).

Why numeric features, not the feature->text summary: a text Transformer trained from
scratch on ~40k short strings would learn poor embeddings and would not measure the
Transformer's capability; feature-token self-attention is the established tabular-
Transformer approach and is directly comparable to the ML control (same 225 features).

Three Transformer variants are compared head-to-head with each other (and with the ML
control + LLM), selected by TFConfig.variant:
  * "feature" (A, default): one token per feature (a learned per-feature affine embedding)
    plus a [CLS] token -> F+1 tokens. Full parity with the ML control's feature set.
    Attention runs OVER FEATURES.
  * "group" (B): one token per SENSOR GROUP (columns sharing the name prefix before ':')
    via a small per-group linear embedding -> ~12 tokens. Attention is O(L^2), so this is
    much cheaper on CPU; a coarser view of the same snapshot. Attention OVER SENSOR GROUPS.
  * "temporal" (C): a different axis entirely — attention OVER TIME. Each sample's feature
    vector is a timestep; a right-aligned window of the K most recent samples of the SAME
    user (gap-aware, see _make_windows) forms the sequence and the model predicts the label
    of the last (target) timestep. Still one prediction per sample, so it stays comparable.
Variants A/B share `ContextTransformer` (token_mode); C uses `TemporalTransformer`.

Preprocessing mirrors the ML control (median impute + standardize, fit per fold on train).
Multi-task: one shared encoder, three linear heads; each field's loss is masked to its
eligible rows and class-weighted (balanced). Isolated track: imports only ee_common.

Guardrails: semantics only (no anomaly judgment); scored only vs real ExtraSensory labels;
user-level split (the existing folds); seeded (torch + numpy) for reproducibility; CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import ee_common as ee

SEED = 0
DEVICE = "cpu"


@dataclass
class TFConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    epochs: int = 8
    batch_size: int = 512
    lr: float = 1e-3
    train_cap: int = 60_000            # random cap on train rows/fold (balanced weights handle skew)
    infer_batch: int = 2048
    variant: str = "feature"           # "feature" (A) | "group" (B) | "temporal" (C)
    token_mode: str = "feature"        # internal knob for A/B (set from `variant` by run_fold)
    # temporal (variant C) only:
    window: int = 16                   # K most-recent samples per user in each sequence
    max_gap_s: int | None = 600        # break the window when adjacent samples are >this apart (s)
    seed: int = SEED


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #
class FeatureTokenizer(nn.Module):
    """One token per numeric feature: token_i = x_i * w_i + b_i (learned w,b per feature),
    with a learned [CLS] token prepended. Input x is (B, F) -> (B, F+1, d)."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = x.unsqueeze(-1) * self.weight + self.bias        # (B, F, d)
        cls = self.cls.expand(x.size(0), -1, -1)               # (B, 1, d)
        return torch.cat([cls, tok], dim=1)                    # (B, F+1, d)


class GroupTokenizer(nn.Module):
    """One token per sensor group (a contiguous slice of columns) via a linear embedding
    of that group's values, with a learned [CLS] prepended. Input columns must already be
    ordered by group so the slices line up (see _group_columns)."""

    def __init__(self, group_sizes: list[int], d_model: int):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Linear(s, d_model) for s in group_sizes])
        self.slices, start = [], 0
        for s in group_sizes:
            self.slices.append((start, start + s))
            start += s
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        toks = [emb(x[:, a:b]) for emb, (a, b) in zip(self.embeds, self.slices)]
        tok = torch.stack(toks, dim=1)                         # (B, G, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tok], dim=1)                    # (B, G+1, d)


def _group_columns(features: list[str]) -> tuple[list[str], list[str], list[int]]:
    """Order features by sensor group (the name prefix before ':') and return
    (ordered_features, group_names, group_sizes)."""
    groups: dict[str, list[str]] = {}
    for c in features:
        groups.setdefault(c.split(":")[0], []).append(c)
    names = sorted(groups)
    ordered = [c for g in names for c in groups[g]]
    sizes = [len(groups[g]) for g in names]
    return ordered, names, sizes


# --------------------------------------------------------------------------- #
# Model: shared encoder + one head per field (multi-task)
# --------------------------------------------------------------------------- #
class ContextTransformer(nn.Module):
    def __init__(self, tokenizer: nn.Module, cfg: TFConfig, n_classes: dict[str, int]):
        super().__init__()
        self.tokenizer = tokenizer
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.heads = nn.ModuleDict(
            {f: nn.Linear(cfg.d_model, n_classes[f]) for f in ee.FIELDS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(self.tokenizer(x))
        cls = self.norm(h[:, 0])                               # [CLS] pooled -> (B, d)
        return {f: head(cls) for f, head in self.heads.items()}


# --------------------------------------------------------------------------- #
# Preprocessing (mirrors E3's ml_fit_predict) — fit ONCE per fold on train
# --------------------------------------------------------------------------- #
def _preprocess(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(train_x)
    sc = StandardScaler().fit(imp.transform(train_x))
    ztr = np.nan_to_num(sc.transform(imp.transform(train_x))).astype(np.float32)
    zte = np.nan_to_num(sc.transform(imp.transform(test_x))).astype(np.float32)
    return ztr, zte


def _field_targets(g_tr: pd.DataFrame) -> tuple[dict, dict, dict]:
    """From the (capped) train rows, build for each field: the sorted class list, the
    per-row label index (-1 = ineligible / class unseen in train -> masked out of loss),
    and the balanced class weights. Shared by the feature/group and temporal fit loops."""
    classes: dict[str, list] = {}
    y_idx: dict[str, np.ndarray] = {}
    weights: dict[str, torch.Tensor] = {}
    for f in ee.FIELDS:
        elig = g_tr[f"elig_{f}"].to_numpy()
        y_val = g_tr[f"gold_{f}"].to_numpy()
        cls = sorted(pd.unique(y_val[elig]))
        cls_to_i = {c: i for i, c in enumerate(cls)}
        yi = np.array([cls_to_i.get(v, -1) for v in y_val], dtype=np.int64)
        yi[~elig] = -1
        classes[f] = cls
        y_idx[f] = yi
        w = compute_class_weight("balanced", classes=np.array(cls), y=y_val[elig])
        weights[f] = torch.tensor(w, dtype=torch.float32)
    return classes, y_idx, weights


# --------------------------------------------------------------------------- #
# fit + predict for one fold (mirrors ml_fit_predict; full-test predictions)
# --------------------------------------------------------------------------- #
def tf_fit_predict(fold: int, feat_df: pd.DataFrame, gold: pd.DataFrame,
                   folds: dict, features: list[str], cfg: TFConfig) -> pd.DataFrame:
    """Train the Transformer on `fold`'s train users and predict the FULL test set.
    Returns a frame indexed by (uuid, timestamp) with pred_<field> + fold — the same
    schema as e3_pred_ml.parquet, so E4 consumes it unchanged."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    uid = feat_df.index.get_level_values("uuid")
    tr_idx = feat_df.index[uid.isin(folds[fold]["train"])]
    te_idx = feat_df.index[uid.isin(folds[fold]["test"])]

    if cfg.token_mode == "group":
        ordered, _gnames, gsizes = _group_columns(features)
    else:
        ordered = features

    x_tr = feat_df.loc[tr_idx, ordered].to_numpy(dtype=np.float32)
    x_te = feat_df.loc[te_idx, ordered].to_numpy(dtype=np.float32)
    x_tr, x_te = _preprocess(x_tr, x_te)

    # random train-row cap for speed (identical draw every run via `rng`)
    if len(x_tr) > cfg.train_cap:
        sel = rng.choice(len(x_tr), cfg.train_cap, replace=False)
        x_tr, tr_idx_cap = x_tr[sel], tr_idx[sel]
    else:
        tr_idx_cap = tr_idx

    # per-field class list / label index / balanced weights from the (capped) train rows.
    g_tr = gold.loc[tr_idx_cap]
    classes, y_idx, weights = _field_targets(g_tr)

    tokenizer = (GroupTokenizer(gsizes, cfg.d_model) if cfg.token_mode == "group"
                 else FeatureTokenizer(len(ordered), cfg.d_model))
    model = ContextTransformer(tokenizer, cfg, {f: len(classes[f]) for f in ee.FIELDS}).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ce = {f: nn.CrossEntropyLoss(weight=weights[f]) for f in ee.FIELDS}

    x_tr_t = torch.tensor(x_tr)
    y_tr_t = {f: torch.tensor(y_idx[f]) for f in ee.FIELDS}
    n = len(x_tr_t)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.train()
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            logits = model(x_tr_t[b])
            loss = None
            for f in ee.FIELDS:
                yb = y_tr_t[f][b]
                m = yb >= 0
                if m.any():
                    term = ce[f](logits[f][m], yb[m])
                    loss = term if loss is None else loss + term
            if loss is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

    model.eval()
    x_te_t = torch.tensor(x_te)
    preds: dict[str, list] = {f: [] for f in ee.FIELDS}
    with torch.no_grad():
        for i in range(0, len(x_te_t), cfg.infer_batch):
            logits = model(x_te_t[i:i + cfg.infer_batch])
            for f in ee.FIELDS:
                preds[f].append(logits[f].argmax(1).cpu().numpy())

    out = pd.DataFrame(index=te_idx)
    for f in ee.FIELDS:
        ai = np.concatenate(preds[f]) if preds[f] else np.array([], dtype=int)
        inv = {i: c for i, c in enumerate(classes[f])}
        out[f"pred_{f}"] = [inv[int(i)] for i in ai]
    out["fold"] = fold
    return out


# --------------------------------------------------------------------------- #
# Variant C — Temporal-sequence Transformer (attention OVER TIME)
# --------------------------------------------------------------------------- #
def _make_windows(index: pd.MultiIndex, K: int, max_gap_s: int | None) -> np.ndarray:
    """Right-aligned time windows, one per row. `index` is (uuid, timestamp), already
    sorted. Returns W (N, K) of int64 indices into a PADDED feature array whose row 0 is a
    zero pad-row and whose real rows are 1..N (so W==0 means 'pad'). The last column is
    always the target row itself; earlier columns walk backward through the SAME user's
    preceding samples, stopping at the user boundary or when the gap to the next included
    sample exceeds `max_gap_s` (None disables the gap check). Rows left at 0 are padding
    (masked out of attention), so an isolated sample degrades gracefully to a 1-token seq."""
    uuids = index.get_level_values("uuid").to_numpy()
    ts = index.get_level_values("timestamp").to_numpy().astype(np.int64)
    n = len(index)
    W = np.zeros((n, K), dtype=np.int64)
    start = 0
    for i in range(1, n + 1):
        if i == n or uuids[i] != uuids[start]:            # end of a user's contiguous block
            for p in range(start, i):
                W[p, K - 1] = p + 1                       # target at the last position
                for back in range(1, K):
                    q = p - back
                    if q < start:
                        break
                    if max_gap_s is not None and (ts[q + 1] - ts[q]) > max_gap_s:
                        break                              # temporal gap: stop extending back
                    W[p, K - 1 - back] = q + 1
            start = i
    return W


class TemporalTransformer(nn.Module):
    """Embed each timestep's feature vector to d_model, add a learned positional embedding
    over the K window slots, encode, and read the LAST slot (the target, always real)."""

    def __init__(self, n_features: int, cfg: TFConfig, n_classes: dict[str, int]):
        super().__init__()
        self.embed = nn.Linear(n_features, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.window, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.heads = nn.ModuleDict(
            {f: nn.Linear(cfg.d_model, n_classes[f]) for f in ee.FIELDS})

    def forward(self, w: torch.Tensor, pad_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.embed(w) + self.pos                       # (B, K, d)
        h = self.encoder(h, src_key_padding_mask=pad_mask) # True entries ignored
        cls = self.norm(h[:, -1])                          # last slot = target -> (B, d)
        return {f: head(cls) for f, head in self.heads.items()}


def temporal_fit_predict(fold: int, feat_df: pd.DataFrame, gold: pd.DataFrame,
                         folds: dict, features: list[str], cfg: TFConfig) -> pd.DataFrame:
    """Variant C. Same signature/return as `tf_fit_predict` (frame indexed by (uuid,
    timestamp) with pred_<field> + fold, full test set), so E4/E6 consume it unchanged."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    uid = feat_df.index.get_level_values("uuid")
    tr_df = feat_df[uid.isin(folds[fold]["train"])]
    te_df = feat_df[uid.isin(folds[fold]["test"])]

    x_tr, x_te = _preprocess(tr_df[features].to_numpy(dtype=np.float32),
                             te_df[features].to_numpy(dtype=np.float32))
    W_tr = _make_windows(tr_df.index, cfg.window, cfg.max_gap_s)
    W_te = _make_windows(te_df.index, cfg.window, cfg.max_gap_s)
    n_feat = len(features)
    # row 0 = zero pad-row; real rows shifted to 1..N (matches _make_windows' +1 convention)
    x_tr_pad = np.vstack([np.zeros((1, n_feat), np.float32), x_tr])
    x_te_pad = np.vstack([np.zeros((1, n_feat), np.float32), x_te])

    # cap the number of TARGET windows for training (history still comes from full x_tr_pad)
    n_tr = len(tr_df)
    sel = (rng.choice(n_tr, cfg.train_cap, replace=False) if n_tr > cfg.train_cap
           else np.arange(n_tr))
    W_tr_cap = W_tr[sel]
    g_tr = gold.loc[tr_df.index].iloc[sel]
    classes, y_idx, weights = _field_targets(g_tr)

    model = TemporalTransformer(n_feat, cfg, {f: len(classes[f]) for f in ee.FIELDS}).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ce = {f: nn.CrossEntropyLoss(weight=weights[f]) for f in ee.FIELDS}

    x_tr_pad_t = torch.tensor(x_tr_pad)
    W_tr_t = torch.tensor(W_tr_cap)
    y_tr_t = {f: torch.tensor(y_idx[f]) for f in ee.FIELDS}
    n = len(W_tr_t)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.train()
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            widx = W_tr_t[b]                                # (B, K)
            w = x_tr_pad_t[widx]                            # (B, K, F)
            pad = widx == 0
            logits = model(w, pad)
            loss = None
            for f in ee.FIELDS:
                yb = y_tr_t[f][b]
                m = yb >= 0
                if m.any():
                    term = ce[f](logits[f][m], yb[m])
                    loss = term if loss is None else loss + term
            if loss is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

    model.eval()
    x_te_pad_t = torch.tensor(x_te_pad)
    W_te_t = torch.tensor(W_te)
    preds: dict[str, list] = {f: [] for f in ee.FIELDS}
    with torch.no_grad():
        for i in range(0, len(W_te_t), cfg.infer_batch):
            widx = W_te_t[i:i + cfg.infer_batch]
            logits = model(x_te_pad_t[widx], widx == 0)
            for f in ee.FIELDS:
                preds[f].append(logits[f].argmax(1).cpu().numpy())

    out = pd.DataFrame(index=te_df.index)
    for f in ee.FIELDS:
        ai = np.concatenate(preds[f]) if preds[f] else np.array([], dtype=int)
        inv = {i: c for i, c in enumerate(classes[f])}
        out[f"pred_{f}"] = [inv[int(i)] for i in ai]
    out["fold"] = fold
    return out


def mean_real_window(feat_df: pd.DataFrame, folds: dict, fold: int, cfg: TFConfig) -> float:
    """Diagnostic: average number of real (non-pad) timesteps per TEST window on `fold`.
    ~1.0 means samples are too sparse for the temporal variant to have any history."""
    uid = feat_df.index.get_level_values("uuid")
    te_df = feat_df[uid.isin(folds[fold]["test"])]
    W = _make_windows(te_df.index, cfg.window, cfg.max_gap_s)
    return float((W > 0).sum(axis=1).mean()) if len(W) else 0.0


# --------------------------------------------------------------------------- #
# Dispatcher: one entry point for all three variants (used by E6 + selftest)
# --------------------------------------------------------------------------- #
def run_fold(fold: int, feat_df: pd.DataFrame, gold: pd.DataFrame, folds: dict,
             features: list[str], cfg: TFConfig) -> pd.DataFrame:
    """Run one fold for cfg.variant in {feature (A), group (B), temporal (C)} and return
    full-test predictions in the e3_pred_ml schema."""
    if cfg.variant == "temporal":
        return temporal_fit_predict(fold, feat_df, gold, folds, features, cfg)
    if cfg.variant not in ("feature", "group"):
        raise ValueError(f"unknown variant {cfg.variant!r}")
    return tf_fit_predict(fold, feat_df, gold, folds, features,
                          replace(cfg, token_mode=cfg.variant))


def selftest(feat_df: pd.DataFrame, gold: pd.DataFrame, folds: dict, features: list[str],
             variant: str = "feature", train_cap: int = 3000) -> bool:
    """Determinism check: two seeded runs of one fold must give identical predictions."""
    cfg = TFConfig(variant=variant, epochs=2, train_cap=train_cap)
    a = run_fold(0, feat_df, gold, folds, features, cfg)
    b = run_fold(0, feat_df, gold, folds, features, cfg)
    return all((a[f"pred_{f}"].to_numpy() == b[f"pred_{f}"].to_numpy()).all() for f in ee.FIELDS)
