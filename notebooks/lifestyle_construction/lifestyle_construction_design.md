# Lifestyle Construction track — design

Isolated track that builds the **bottom half of Asara's p3/9 diagram** — *Personal Lifestyle
Semantic Construction → Lifestyle KG → lifestyle map* — which the main pipeline (File 1–4) never
built. It also closes a gap the user pointed out in File 2: its "context definition" step only
**normalizes** labels, it does not **enrich** them.

Two stages, combined (user-chosen direction, 2026-07-23):

```
behavioral_episodes.parquet  ──Stage A (LLM enrich)──►  behavioral_episodes_enriched.parquet
                                                             │
                                     Stage B (self-supervised Transformer) ──► day embeddings
                                                             │                   ──► Lifestyle KG
                                                             └───────────────────► lifestyle map
```

Everything is **isolated** (reads `data/processed/` + `results/location_context/` read-only, writes
only `results/lifestyle_construction/`), **offline-first**, **seeded/CPU/deterministic**, and
**descriptive only** (no anomaly or health claims). Core pattern throughout: **code measures the
numbers, the LLM names them** — the LLM never ingests raw rows.

---

## Stage A — LLM episode enrichment (`episode_enrichment.py`)

File 2's Step 4 runs Gemini only to map a tiny `(activity_hint, location_type, is_workout)` combo
set onto an enum — with no HR values, coordinates, or place names in the prompt, so given
`location_type="unknown"` it can only echo `unknown`. Stage A gives the LLM a real fingerprint.

- **Signature (code):** each episode → `(is_workout, activity, has_place, hour_bucket, weekday,
  hr_band vs personal baseline, weather_ctx)`. The 104,565 episodes collapse to **326 distinct
  signatures**, so the LLM is called 326× (temp=0, disk-cached) — not per row.
- **Label (LLM):** Gemini maps each signature → `activity_context` (a small per-episode SITUATION
  enum defined here, since the Task-1 vocab is subject-level) + `workout_type` (from the frozen
  Task-1 `WorkoutTypeVocab`). `weather_ctx` is taken directly from the frozen `WeatherVocab`
  (deterministic from temp/humidity — the LLM is not needed for it).
- **Provenance & fallback:** enum-constrained output; every episode carries `enrich_source ∈
  {llm, fallback}`; a deterministic rule labels everything when no key is present. Runs fully
  offline.
- **Output:** `behavioral_episodes_enriched.parquet` (+ `activity_context`, `weather_ctx`,
  `workout_type_ctx`, `location_semantic`, `enrich_source`) + `l0_enrich_meta.json`.

**Observation (real run):** the LLM used a **cleaner, more consistent** scheme than the rule
fallback — it labelled by time-of-day + workout status (`daytime_rest` 61.8k / `evening_rest` 35.7k /
`overnight_rest` 6.1k / `active_workout` 0.9k) and dropped the fallback's HR-band-driven
`light_activity`. That consistency turns out to help Stage B (below).

---

## Stage B — self-supervised lifestyle Transformer (`lifestyle_construction.py`)

No lifestyle labels exist on an n=1 subject, so the encoder is **self-supervised**.

- **B1 day tensor + encoder.** One DAY = its episodes in 96 fixed 15-min slots; per-slot feature =
  `[z(avg/max/min_hr, hrv, temp, humidity), is_workout, one-hot categoricals]`. `feature_mode ∈
  {raw, enriched}` (raw = activity+location_type; enriched additionally = activity_context +
  weather_ctx) — this is the ablation. `LifestyleDayEncoder` (adapts `ee_transformer.TemporalTransformer`)
  is trained by **masked-episode reconstruction** (mask ~15% real slots, reconstruct continuous MSE
  + categorical CE); the day embedding is the mean-pooled encoder output. Seeded/deterministic.
- **B2 Lifestyle KG.** KMeans on day embeddings (K by silhouette) → day-type **states**; code
  measures each state's attributes (median HR, weekday/workout fraction, wear slots, peak months,
  mean temp); the **LLM names** each state from that profile (`name_states_llm`, cached + rule
  fallback), told to label by what actually distinguishes a cluster (usually season) not by ~1 bpm HR
  gaps. Nodes = states(+attrs+name), edges = day-to-day transition counts. *Code measures → LLM names.*
  On the real subject the LLM produced `chilly_winter_rest` / `mild_autumn_active` / `hot_summer_peak`
  (vs the old rule's thin, duplicate `high_tone_weekday` ×2).
- **B3 lifestyle map.** State distribution + weekly rhythm (state × weekday) + temporal drift by
  year + subject priors (`global_context.json`, `home_climate.json`).
- **B4 regime shifts (rendered in File 4).** `segment_regimes` merges consecutive same-state days into
  sustained regimes (≥5 days; brief flicker absorbed, adjacent same-state regimes merged);
  `regime_breaks` scores each boundary with **code-measured** deltas (HR tone, temp, wear, workout
  frequency) plus `p_transition` and `season_share` (how usual that state is in those calendar months —
  `<OFF_SEASON_SHARE=0.15` = an **off-season departure**); `explain_breaks_llm` has the LLM write one
  descriptive sentence per shift (cached, rule fallback). **Lifestyle deviation, not anomaly
  detection** — "when did this person's own routine shift?", never a health verdict.
- **Output:** `day_embeddings.parquet`, `lifestyle_kg_nodes.csv`, `lifestyle_kg_edges.csv`,
  `lifestyle_map.json`, `l1_proxy_eval.csv`, `l1_meta.json`.

---

## Honest, circularity-safe evaluation

No ground truth exists, so validate that day-clusters recover **derivable structure never used as a
target**, and beat a trivial baseline. Metric = adjusted mutual information (AMI) of clusters vs a
proxy.

- **Clean proxies:** `month` / `season` and `workout_day` (not encoder targets). `weekday` is
  reported but flagged **dependent** (time-of-day is implicitly available via slot positions).
- **Ablation:** run B1 for `raw` and `enriched` — the real test of whether the LLM layer adds value.
- **Baseline:** cluster a hand-crafted daily aggregate (mean HR, %active, wear, HR entropy, temp).

### Results (real subject, 1,761 days, single seed)

| arm | month AMI | season AMI | workout_day AMI |
|---|---|---|---|
| transformer_enriched (LLM) | **0.167** | **0.192** | 0.008 |
| transformer_enriched (rule fallback) | 0.108 | 0.123 | 0.004 |
| transformer_raw | 0.117 | 0.136 | 0.052 |
| aggregate_baseline | 0.028 | 0.023 | **0.317** |

**Findings (honest):**
1. **LLM enrichment helps** — LLM-enriched embeddings recover seasonal lifestyle structure best
   (season 0.192 > raw 0.136 > fallback 0.123 > baseline 0.023). The LLM's consistent labelling
   beats both raw features and rule labels. (Offline/fallback enrichment did *not* help — it is a
   coarsened function of raw — so the LLM version is what makes the enrichment worthwhile.)
2. **Both Transformers ≫ aggregate baseline on month/season** — the sequence model captures seasonal
   rhythm the trivial aggregate misses.
3. **Aggregate baseline wins on `workout_day`** (0.317) — day-pooling washes out the ~900 rare
   workout episodes; a simple "% active" feature separates workout days better. Not hidden.

**Limitations to state to Asara:** single seed (direction is consistent across month *and* season
but multi-seed mean±std would firm it up); non-workout `activity_context` is an LLM **inference**
(the ablation shows it is a *useful* inference, not verified truth); "help" is on the **seasonal**
axis, not rare events; and — as everywhere on this subject — outside workouts the only signal is HR +
time + weather. The method transfers directly to a richer, tagged dataset.

---

## Files & how to run

```
notebooks/lifestyle_construction/
  episode_enrichment.py            # Stage A module
  lifestyle_construction.py        # Stage B module
  L0_enrich.py                     # Stage A runner  ->  behavioral_episodes_enriched.parquet
  L1_lifestyle.py                  # Stage B runner  ->  embeddings / KG / map / proxy eval
  demo_lifestyle_construction.py   # offline selftest (synthetic known day-types + determinism)
  lifestyle_construction_design.md # this file
```

- `python demo_lifestyle_construction.py --offline` → synthetic day-types recovered (ARI 1.000),
  embeddings reproducible.
- `python L0_enrich.py [--offline]` → enriched episodes (offline = rule labels; online = Gemini).
- `python L1_lifestyle.py --offline` → Stage B end-to-end + the enriched-vs-raw-vs-baseline table.

## Reuse

`ee_transformer.py` (TemporalTransformer, preprocessing, seeded CPU loop); File 2's Gemini pattern
(`EpisodeContext`, google-genai structured output, temp=0, `.env` from repo root, deterministic
fallback); Task-1 `context_vocab.generated_vocab` (frozen vocab, one-way import); `location_context`
place names; `context_baseline`/`location_context` JSON priors — all via files, no cross-track
runtime coupling.

## Not done (opt-in / follow-up)

- Wiring an additive "Step 4b" enrichment cell into File 2 (as `location_context` was wired) — needs
  explicit approval since it edits the main pipeline.
- Multi-seed robustness for the ablation; optional Gemini naming of lifestyle states.
