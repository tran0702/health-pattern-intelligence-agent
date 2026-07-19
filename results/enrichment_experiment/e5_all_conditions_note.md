# RQ1b — full C0 / C1 / C2 contrast (plan §9.5)

Same `e3_eval_sample` (1,522 location-eligible rows), same few-shot, only the location
cue changes. All three are the SAME `gemini-3.1-flash-lite` few-shot enricher:
- **C0** — sensor summary only (frozen baseline).
- **C1** — sensor summary **+ raw coordinates** ("latitude 32.87421, longitude -117.22137"), no geocode.
- **C2** — sensor summary **+ reverse-geocoded semantic place** ("Cardiff State Beach Parking (OSM: amenity/parking)").

Artifacts: `e5_all_conditions_location_per_class.csv`, `e5_pred_llm_c1.parquet` /
`e5_pred_llm_c2.parquet`, `e5_llm_c1_cost.json` / `e5_llm_c2_cost.json`.

## Location per-class F1 (recall in parens)

| class      | support | C0 (sensor) | C1 (+raw coords) | C2 (+semantic) |
|------------|--------:|------------:|-----------------:|---------------:|
| **beach**  | 130 | .000 (.00) | **.015 (.01)** | **.426 (.28)** |
| **gym**    | 200 | .077 (.05) | .149 (.09) | **.589 (.45)** |
| restaurant | 200 | .059 (.04) | .112 (.07) | **.470 (.41)** |
| work       | 200 | .313 (.30) | .549 (.64) | .593 (.68) |
| home       | 200 | .432 (.87) | .493 (.83) | .494 (.78) |
| outdoors   | 200 | .343 (.43) | .344 (.45) | .311 (.32) |
| transit    | 200 | .677 (.56) | .683 (.57) | .665 (.67) |
| unknown    | 192 | .081 | .111 | .092 |

**macro-F1 location:  C0 .248  →  C1 .307  →  C2 .455.**  (ML-on-relative-features ref: **.344**.)
Over-confidence stays flat across all three (.911 / .875 / .922). C1 = 1,508 live calls, 0.92 s/call.

## Reading (the clean contrast §9.5 asked for)

- **Raw coordinates (C1) barely help, and NOT where it matters.** macro-F1 moves only
  +.06 (.248→.307), still **below ML (.344)**. On the hard, semantically-distinct classes
  it does almost nothing: **beach recall .00 → .008** (1 hit in 130), gym/restaurant crawl
  up but stay <.15 F1. The bump C1 *does* give is concentrated in the common, spatially
  clustered classes — **work** (.31→.55) and **home** (.43→.49) — consistent with the LLM
  doing crude coordinate-proximity matching to nearby few-shot examples (a place-specific
  crutch), not semantic understanding.
- **Semantic location (C2) is transformative on exactly the classes C1 can't touch:**
  beach .00→.43, gym→.59, restaurant→.47; macro-F1 .455, **overtaking ML**.
- **Therefore:** the LLM needs **semantic** location, not coordinates. Coordinates alone
  give a small, uneven, place-specific lift that leaves the LLM's real failure modes
  (beach/gym/restaurant) unsolved; only *naming the place* fixes them. This separates
  "needs coordinates" from "needs semantics" — decisively the latter.

## Bottom line for RQ1b
The LLM's location weakness (RQ1) was a **missing-semantic-signal** problem, not poor
reasoning and not merely a missing-number problem. Feed it a human-readable place and it
uses it correctly (high precision) and beats the ML baseline; feed it bare coordinates and
it mostly can't. Caveat unchanged (§9.4): C2 is an **LLM + geocoder hybrid**, so read this
as "semantic location unlocks the LLM," not "LLM > ML in general" — ML was never given the
coordinates (a person-specific / near-leakage setup the source paper excluded, §9.2).

Remaining optional rigor: bootstrap-CI by user on the C0→C2 gap for a p-value.
