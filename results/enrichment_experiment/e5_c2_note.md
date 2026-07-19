# RQ1b / C2 — does semantic location unlock the LLM? (plan §9.4–§9.5)

**Setup.** Same `e3_eval_sample` (1,522 location-eligible rows) as C0, so numbers are
directly comparable. C2 = the frozen few-shot LLM **plus** a reverse-geocoded place
cue for each sample's real coordinate (e.g. `"Cardiff State Beach Parking, South Coast
Highway 101 (OSM: amenity/parking)"`), added to both the few-shot examples and the
query. Geocoder = OSM Nominatim (offline-cached, ~900 unique coords). LLM =
`gemini-3.1-flash-lite`, 1,504 live calls, 0.83 s/call. Artifacts:
`e5_c2_location_per_class.csv`, `e5_pred_llm_c2.parquet`, `e5_llm_c2_cost.json`.

## Location per-class F1 — LLM C0 (sensor only) vs C2 (+ map cue)

| class      | support | C0 F1 | **C2 F1** | C0 recall | **C2 recall** | C2 precision | note |
|------------|--------:|------:|----------:|----------:|--------------:|-------------:|------|
| **beach**  | 130 | .000 | **.426** | **.000** | **.277** | .923 | **litmus PASSED: 0 → predicts beach, almost always right** |
| **gym**    | 200 | .077 | **.589** | .045 | **.445** | .873 | huge lift despite patchy OSM gym tags |
| restaurant | 200 | .059 | **.470** | .035 | **.410** | .550 | huge lift |
| work       | 200 | .313 | **.593** | .300 | **.675** | .529 | big lift |
| home       | 200 | .432 | .494 | .870 | .775 | .362 | small lift |
| outdoors   | 200 | .343 | .311 | .430 | .320 | .302 | slight drop (diffuse class → geocode names a nearby POI) |
| transit    | 200 | .677 | .665 | .560 | .670 | .660 | flat — already speed-driven, doesn't need the cue |
| unknown    | 192 | .081 | .092 | .089 | .078 | .112 | flat |

**macro-F1 location:  LLM-C0 .248  →  LLM-C2 .455.**  For reference, ML on the full
225 relative features (C0) is **.344** — so with a semantic location cue the LLM
(.455) now **overtakes** the ML baseline. Over-confidence barely moves (.911 → .922)
and the rescued classes are **high-precision** (beach .92, gym .87), so the gains are
real commitments, not spray-and-pray.

## Answer to RQ1b

**The LLM's location weakness was a missing-signal problem, not a reasoning problem.**
The moment it is given a semantic "where am I" cue, `beach` recall jumps from **0 → .28**,
`gym`/`restaurant` go from ~0 to ~.5 F1, and macro-F1 nearly doubles (.248 → .455),
passing ML. This confirms the §9.5 reading: *"LLM bị bỏ đói tín hiệu, không phải kém."*
It also matches Bước 0: `gym` was a signal ceiling for the *relative* features (ML also
~0), but absolute location + geocoding lifts it — the ceiling was the input, not the model.

## Honest caveats (do not oversell)

1. **C2 is an LLM + geocoder hybrid** (plan §9.4). A large share of the lift is the
   geocoder *naming* the place (a coordinate near "…State Beach…" makes `beach` easy).
   The finding is "the LLM correctly *consumes* semantic location," **not** "the LLM
   alone is brilliant." That is exactly the RQ1b question, and it is answered.
2. **Not an apples-to-apples LLM-vs-ML win.** We compared LLM-C2 against ML-C0 (relative
   features only). We did **not** give ML the coordinates (C1). ML-with-coords would also
   jump — but that is a *person-specific, near-leakage* setup (memorizing each person's
   fixed home/gym coordinate), which is why the source paper excluded absolute location
   (§9.2). So read this as **"semantic location unlocks the LLM,"** not "LLM > ML in general."
3. Significance not yet quantified here (Bước 0-style per-class table only). A bootstrap
   CI by user on the C0→C2 gap can be added if the professor wants a p-value.
4. `outdoors` slipped slightly — reverse-geocoding an open-air point often returns a
   nearby named POI, nudging it off `outdoors`. Minor, expected.

## Suggested next steps (optional)
- **C1 (raw coords, no geocode) for the LLM** — cheap add; expected to help *little*
  (numbers are not semantic to an LLM). If C1 ≪ C2, it proves the LLM needs *semantic*
  location, not coordinates — a clean, publishable contrast.
- **Bootstrap CI by user** on the C0→C2 location gap for statistical rigor.
