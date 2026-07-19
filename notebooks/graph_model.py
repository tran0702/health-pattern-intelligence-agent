"""
graph_model.py - the ONE graph model used across the pipeline.

Task 3 (Asara) asked for a single model used throughout. The only graph model in
this project is a DOMINANT autoencoder built on **GCN** (`GCNConv`) layers - "GNN"
is the family, "GCN-DOMINANT" is the concrete member. It was previously defined
twice (File 3 trains it on the real BKG, File 4 retrains it on the injected graph),
with a couple of small differences. This module is the single shared definition so
"one model throughout" is literally true.

It also removes the old PCA-fallback ambiguity: `structural_scores` returns the
method name it actually used ("GCN-DOMINANT" when torch_geometric is available,
"PCA-proxy" otherwise) so a fallback number can never be reported as a real GCN
result.

Node features are structural / physiological + the context-baseline `deviation_z`.
NO LLM anomaly label is ever included (guardrail: the structural view stays
independent of the semantic view).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    TG_AVAILABLE = True
except Exception:                       # torch or torch_geometric missing
    TG_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Shared node-feature construction (identical for File 3 and File 4).
# --------------------------------------------------------------------------- #
def build_node_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Standardized node-feature matrix used by both notebooks.

    Requires columns: datetime, avg_hr, hrv_sdnn, n_samples, deviation_z,
    activity, location_type. If weather_temp / weather_humidity are present and
    populated, they are added as features (enrichment). Returns (X, column_names).
    NO LLM anomaly label.
    """
    t_h = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    base = pd.DataFrame({
        "deviation_z": df["deviation_z"].fillna(0.0).to_numpy(),
        "avg_hr":   df["avg_hr"].fillna(df["avg_hr"].median()).to_numpy(),
        "hrv_sdnn": df["hrv_sdnn"].fillna(df["hrv_sdnn"].median()).to_numpy(),
        "n_samples": df["n_samples"].to_numpy(),
        "hour_sin": np.sin(2 * np.pi * t_h / 24).to_numpy(),
        "hour_cos": np.cos(2 * np.pi * t_h / 24).to_numpy(),
    }, index=df.index)
    # weather features, only when actually populated (episodes may carry all-NaN)
    for col in ("weather_temp", "weather_humidity"):
        if col in df.columns and df[col].notna().any():
            base[col] = df[col].fillna(df[col].median()).to_numpy()
    act_oh = pd.get_dummies(df["activity"], prefix="act")
    loc_oh = pd.get_dummies(df["location_type"], prefix="loc")
    Xdf = pd.concat([base, act_oh.astype(float), loc_oh.astype(float)], axis=1)
    X = StandardScaler().fit_transform(Xdf.values)
    return X, list(Xdf.columns)


# --------------------------------------------------------------------------- #
# The single GCN-DOMINANT model (defined only when torch_geometric is present).
# --------------------------------------------------------------------------- #
if TG_AVAILABLE:

    class DominantLite(nn.Module):
        """Minimal DOMINANT: a shared GCN encoder feeds an attribute decoder;
        the structure is reconstructed via the inner product of the embedding in
        the loss. GCN is chosen over GAT/SAGE for this homogeneous, curated,
        single-subject graph (symmetric normalisation, few params, unsupervised
        reconstruction) - see context_baseline_design.md, Task 3."""

        def __init__(self, in_dim: int, hid: int = 64, emb: int = 32):
            super().__init__()
            self.enc1 = GCNConv(in_dim, hid)
            self.enc2 = GCNConv(hid, emb)
            self.attr = GCNConv(emb, in_dim)     # attribute reconstruction

        def forward(self, x, ei):
            h = F.relu(self.enc1(x, ei))
            z = self.enc2(h, ei)                 # node embedding
            return z, self.attr(z, ei)

    def _struct_loss(z, ei, generator=None):
        s, d = ei
        pos = (z[s] * z[d]).sum(1)
        ns = torch.randint(0, z.size(0), (s.size(0),), device=z.device, generator=generator)
        nd = torch.randint(0, z.size(0), (s.size(0),), device=z.device, generator=generator)
        neg = (z[ns] * z[nd]).sum(1)
        return (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) +
                F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))

    def train_dominant(X, edge_index, epochs: int = 150, alpha: float = 0.7,
                       lr: float = 1e-3, weight_decay: float = 1e-5, seed: int = 0,
                       verbose: bool = False):
        """Train GCN-DOMINANT. alpha weights attribute vs structure reconstruction.
        Returns (model, x_tensor, edge_index_tensor)."""
        torch.manual_seed(seed)
        dev = "cpu"
        x = torch.tensor(np.asarray(X), dtype=torch.float, device=dev)
        ei = torch.as_tensor(np.asarray(edge_index), dtype=torch.long, device=dev)
        model = DominantLite(x.size(1)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            z, x_hat = model(x, ei)
            la = F.mse_loss(x_hat, x)
            ls = _struct_loss(z, ei)
            loss = alpha * la + (1 - alpha) * ls
            loss.backward(); opt.step()
            if verbose and epoch % 30 == 0:
                print(f"epoch {epoch:3d}  attr={la.item():.4f}  struct={ls.item():.4f}")
        return model, x, ei

    def score_dominant(model, x, ei):
        """Per-node attribute-reconstruction error + per-edge reconstruction error.
        A real observed edge the model assigns low probability scores high."""
        model.eval()
        with torch.no_grad():
            z, x_hat = model(x, ei)
            node_score = ((x - x_hat) ** 2).mean(1).cpu().numpy()
            s, d = ei
            edge_logit = (z[s] * z[d]).sum(1)
            edge_score = F.binary_cross_entropy_with_logits(
                edge_logit, torch.ones_like(edge_logit), reduction="none").cpu().numpy()
        return node_score, edge_score


# --------------------------------------------------------------------------- #
# One entry point: run the structural detector and say which method it used.
# --------------------------------------------------------------------------- #
def structural_scores(X, edge_index, **train_kw):
    """Return (node_score, edge_score, method_name).

    method_name is "GCN-DOMINANT" when torch_geometric is available, else
    "PCA-proxy" - so a fallback score is never mistaken for a real GCN result.
    """
    edge_index = np.asarray(edge_index)
    if TG_AVAILABLE:
        model, x, ei = train_dominant(X, edge_index, **train_kw)
        ns, es = score_dominant(model, x, ei)
        return ns, es, "GCN-DOMINANT"

    from sklearn.decomposition import PCA
    X = np.asarray(X)
    pca = PCA(n_components=min(8, X.shape[1]))
    Xrec = pca.inverse_transform(pca.fit_transform(X))
    node_score = ((X - Xrec) ** 2).mean(1)
    s, d = edge_index[0], edge_index[1]
    edge_score = node_score[s] + node_score[d]      # endpoint-sum proxy
    return node_score, edge_score, "PCA-proxy"
