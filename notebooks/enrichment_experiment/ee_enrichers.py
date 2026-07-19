"""
ee_enrichers.py — enricher logic for E3 (importable so LLM runs can be re-done
without retraining ML). Feature->text summary, few-shot builder (leakage-free,
per-fold from train users), and the Gemini LLMEnricher with 503-retry + caching.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

import ee_common as ee


# --------------------------------------------------------------------------- #
# Feature -> interpretable text summary (the only thing the LLM sees)
# --------------------------------------------------------------------------- #
def _motion(std, lo, hi):
    # sensor-specific thresholds: phone accel ~ g units; watch accel is a different, larger scale
    if pd.isna(std):
        return None
    return "still" if std < lo else "light motion" if std < hi else "vigorous motion"


def summarize_features(row, geo_phrase: str | None = None,
                       geo_label: str = "nearby place (map lookup)") -> str:
    b = []
    tod = [c.split(":")[-1].replace("between", "").replace("and", "-")
           for c in ee.TIME_OF_DAY_COLS if row.get(c) == 1]
    if tod:
        b.append("local time ~ " + ", ".join(f"{a}h" for a in tod))
    m = _motion(row.get("raw_acc:magnitude_stats:std"), 0.03, 0.15)
    b += [f"phone {m}"] if m else []
    w = _motion(row.get("watch_acceleration:magnitude_stats:std"), 10, 100)
    b += [f"wrist {w}"] if w else []
    g = row.get("proc_gyro:magnitude_stats:std")
    if pd.notna(g):
        b.append(f"rotation {'high' if g > 0.5 else 'low'} (gyro std {g:.2f})")
    sp = row.get("location:max_speed")
    if pd.notna(sp):
        lvl = ("stationary" if sp < 0.5 else "walking pace" if sp < 2
               else "running/cycling pace" if sp < 8 else "vehicle speed")
        b.append(f"{lvl} (max GPS speed {sp:.1f} m/s)")
    alt = row.get("location:max_altitude", np.nan) - row.get("location:min_altitude", np.nan)
    if pd.notna(alt):
        b.append(f"altitude range {alt:.0f} m")
    d = row.get("location:log_diameter")
    if pd.notna(d):
        b.append(f"GPS spread log-diameter {d:.1f}")
    nu = row.get("location:num_valid_updates")
    if pd.notna(nu):
        b.append(f"{int(nu)} GPS updates")
    a = row.get("audio_properties:max_abs_value")
    if pd.notna(a):
        b.append(f"ambient audio {'loud' if a > 0.3 else 'quiet'}")
    for col, lab in [("discrete:on_the_phone:is_True", "on a phone call"),
                     ("discrete:wifi_status:is_reachable_via_wifi", "on wifi"),
                     ("discrete:battery_state:is_charging", "phone charging")]:
        if row.get(col) == 1:
            b.append(lab)
    s = "; ".join(b) if b else "no reliable sensor signal"
    # RQ1b: append a location cue when provided. geo_label distinguishes conditions —
    # C2 = "nearby place (map lookup)" (semantic), C1 = "GPS coordinates" (raw numbers).
    # geo_phrase=None (default) => byte-identical to the C0 summary used by E1-E4; the
    # default geo_label keeps the already-run C2 prompts hash-identical too.
    if geo_phrase:
        s += f" | {geo_label}: {geo_phrase}"
    return s


PROMPT_SYS = (
    "You label the context of one wearable-sensor sample. Choose exactly one value per field from "
    "the allowed vocabulary. Use 'unknown' (location/activity) or 'alone' (companion) when the signal "
    "is insufficient — do not guess. Semantic labeling only; never infer health anomalies.\n"
)


# --------------------------------------------------------------------------- #
# Few-shot examples — drawn ONLY from a fold's train users (no leakage).
# Cover every location class (incl. 'unknown' -> teaches abstention) + activities.
# --------------------------------------------------------------------------- #
def fewshot_picks(gold, train_uuids, seed=0, n_per_loc=2, n_per_act=1) -> list:
    """Deterministic list of example indices for one few-shot block (no leakage).

    Selection order + RNG usage are unchanged from the original build_fewshot, so
    the C0 examples are identical; exposing the picks lets E5 pre-geocode them.
    """
    uid = gold.index.get_level_values("uuid")
    g = gold.loc[uid.isin(train_uuids)]
    rng = np.random.default_rng(seed)
    picked = set()

    def take(col, elig_col, k):
        gg = g[g[elig_col]]
        for _, sub in gg.groupby(col):
            idxs = list(sub.index)
            rng.shuffle(idxs)
            for i in idxs[:k]:
                picked.add(i)

    take("gold_location", "elig_location", n_per_loc)
    take("gold_activity", "elig_activity", n_per_act)
    return sorted(picked, key=lambda x: (str(x[0]), x[1]))


def build_fewshot(gold, feat_df, train_uuids, seed=0, n_per_loc=2, n_per_act=1, geo=None,
                  geo_label="nearby place (map lookup)") -> str:
    """Few-shot text block. `geo` (dict index->phrase) appends a location cue per
    example (C2 semantic place / C1 raw coords via geo_label); geo=None (default)
    reproduces the exact C0 block used by E3/E4."""
    lines = ["Examples (sensor summary => label):"]
    for i in fewshot_picks(gold, train_uuids, seed, n_per_loc, n_per_act):
        row = feat_df.loc[i]
        lab = {"location": gold.loc[i, "gold_location"],
               "activity": gold.loc[i, "gold_activity"],
               "companion": gold.loc[i, "gold_companion"]}
        gp = geo.get(i) if geo is not None else None
        lines.append(f"- {summarize_features(row, gp, geo_label)} => {json.dumps(lab)}")
    return "\n".join(lines)


def fewshot_by_fold(gold, feat_df, folds, geo=None, **kw) -> dict:
    """One few-shot block per fold, built from that fold's train users.
    Pass geo (dict index->phrase) for the C2 condition; omit it for C0."""
    return {k: build_fewshot(gold, feat_df, folds[k]["train"], seed=k, geo=geo, **kw)
            for k in folds}


# --------------------------------------------------------------------------- #
# LLM enricher
# --------------------------------------------------------------------------- #
_TRANSIENT = ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "deadline", "overloaded")


class LLMEnricher:
    def __init__(self, model=ee.GEMINI_MODEL, shots: dict | None = None, max_retries=4,
                 geo: dict | None = None, geo_label: str = "nearby place (map lookup)"):
        self.model = model
        self.shots = shots or {}                 # {fold: fewshot_text}; {} -> zero-shot
        self.max_retries = max_retries
        self.geo = geo or {}                     # {index: phrase} for C1/C2; {} -> C0 (no cue)
        self.geo_label = geo_label               # "nearby place (map lookup)" C2 / "GPS coordinates" C1
        self._client = None
        self.latencies: list[float] = []

    def _prompt(self, row, fold) -> str:
        shot = self.shots.get(fold, "")
        block = (shot + "\n\n") if shot else ""
        gp = self.geo.get(row.name) if self.geo else None   # row.name == (uuid, timestamp)
        return f"{PROMPT_SYS}{block}Now label this sample:\n{summarize_features(row, gp, self.geo_label)}"

    def _cache_path(self, prompt):
        h = hashlib.sha256((self.model + "\n" + prompt).encode()).hexdigest()[:32]
        return ee.LLM_CACHE_DIR / f"{h}.json"

    def _generate(self, prompt):
        if self._client is None:
            from google.genai import types
            self._client, self._types = ee.get_gemini_client(), types
        cfg = self._types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json",
            response_schema=ee.ContextLabel,
            thinking_config=self._types.ThinkingConfig(thinking_budget=0))
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg)
                return r.parsed.model_dump()
            except Exception as e:                       # retry only transient errors
                last = str(e)
                if attempt == self.max_retries - 1 or not any(t in last for t in _TRANSIENT):
                    break
                time.sleep(1.5 * (2 ** attempt))
        return {"location": "unknown", "activity": "unknown",
                "companion": "unknown", "_error": (last or "unknown")[:120]}

    def _call(self, prompt):
        cp = self._cache_path(prompt)
        if cp.exists():
            return json.loads(cp.read_text())
        out = self._generate(prompt)
        cp.write_text(json.dumps(out))
        return out

    def predict(self, df, fold_series):
        from tqdm.auto import tqdm
        rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM"):
            prompt = self._prompt(row, int(fold_series.loc[idx]))
            cached = self._cache_path(prompt).exists()
            t = time.time()
            out = self._call(prompt)
            if not cached:
                self.latencies.append(time.time() - t)
            rows.append(out)
        return pd.DataFrame(rows, index=df.index).rename(
            columns={f: f"pred_{f}" for f in ee.FIELDS})
