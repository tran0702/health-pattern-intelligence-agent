# Bước 0 (plan §9.3) — Location per-class pre-check

**Data:** `e2_gold_labels.parquet` × `e3_pred_ml.parquet` × `e3_pred_llm.parquet`,
restricted to the shared `e3_eval_sample` (1,522 location-eligible rows). Same scoring
rule as E4 (eligible rows only). macro-F1 reproduces E4 exactly: ML .344 / LLM .248.

| class      | support | ml_f1 | llm_f1 | ml_recall | llm_recall | note |
|------------|--------:|------:|-------:|----------:|-----------:|------|
| transit    | 200 | .545 | **.677** | .53 | .56 | speed signal — both good, LLM edges ahead |
| home       | 200 | .452 | .432 | .64 | .87 | routine/time signal — both good |
| work       | 200 | **.533** | .313 | .64 | .30 | ML uses signal LLM misses |
| restaurant | 200 | **.438** | .059 | .35 | .035 | ML clearly better; LLM barely predicts it |
| beach      | 130 | **.290** | **.000** | .22 | **.00** | ML works; **LLM never predicts beach** |
| outdoors   | 200 | .275 | .343 | .24 | .43 | LLM slightly better |
| gym        | 200 | .037 | .077 | .025 | .045 | **both ≈ 0 — signal ceiling** |
| unknown    | 192 | .184 | .081 |  |  | LLM over-commits, loses the abstain class |

## Assessment (one paragraph)

The location ceiling is **class-dependent, not one story**. `gym` is a genuine signal
ceiling: even ML with the full 225 features scores ~0 (f1 .037, recall .025) — the
*relative*-location features simply cannot separate a gym from any other indoor static
place, so neither model's fault and GPS is the only plausible unlock. But `beach` and
`restaurant` break the clean "LLM is just starved of signal" hypothesis: from the **same**
relative features ML reaches f1 .29 / .44 while the LLM scores ~0 and **never predicts
`beach` at all** — usable signal *does* exist in those features, the LLM just isn't
extracting it from the text-summarized prompt. So the pre-check gives a mixed verdict:
`gym` (+ the collapsed rare classes) is signal-limited, while `beach`/`restaurant`/`work`
are an **LLM-specific extraction weakness** where the signal is present. Where signal is
unambiguous (`transit` speed, `home` routine) both models do fine and the LLM even wins.

## Decision (§9.3 ⏸ — is real GPS worth downloading?)

**Conditionally yes, but only for the C2 (semantic geocode) variant.** Rationale:
- The sharpest LLM failures — beach recall 0, gym recall ~0, restaurant recall ~0 — are
  exactly the "static place" classes that absolute coordinates + reverse-geocoded POI
  descriptions (C2) are designed to rescue. §9.5's litmus test ("does LLM `beach` recall
  jump from 0 to >0?") is directly testable.
- **Caveat (§9.2):** C1 (raw lat/long) makes location a *person-specific, near-leakage*
  problem — ML would memorize each person's fixed home/gym coordinate; gains won't
  generalize and won't fairly test the LLM. So C1 alone would just reproduce what we
  already see (helps ML, not LLM) and would **not** answer "is the LLM dumb or starved".
- Therefore: download `ExtraSensory.per_uuid_absolute_location.zip` only if we commit to
  building **C2** (offline reverse-geocoder + cache) and reading the result on
  beach/gym/restaurant recall — not just bolting raw coords onto the prompt.

If there is no appetite for the C2 build, Bước 0 already answers enough: the LLM's
location gap is part signal-ceiling (gym) and part extraction-weakness (beach/restaurant),
and raw GPS would not cleanly separate those.
