"""
demo_vocab.py — build & verify the AI-generated context vocabulary (Task 1).

    python demo_vocab.py            # LLM if GEMINI_API_KEY present, else offline seed
    python demo_vocab.py --offline  # force the deterministic seed (no network)

It generates the vocabulary for the defined profile, freezes vocabulary.json, code-gens
generated_vocab.py (Pydantic Literals), writes the markdown report, then verifies:
  * every dimension of the spine is covered and carries an 'unknown' value,
  * generated_vocab.py imports and its Literals match the frozen JSON,
  * free-text normalization folds messy strings onto controlled values.
"""
from __future__ import annotations

import importlib
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import context_profile as cp                                  # noqa: E402
import vocab_generator as vg                                  # noqa: E402


def main() -> int:
    source = "default" if "--offline" in sys.argv else "auto"
    live = source != "default" and vg.key_available()
    print("=" * 78)
    print(f"TASK 1 — AI-generated context vocabulary   (source={source}, "
          f"live_llm={'yes' if live else 'no -> offline seed'})")
    print("=" * 78)

    vocab = vg.build_all(source=source)
    m = vocab["meta"]
    print(f"\nGenerated: {m['n_dimensions']} dimensions, {m['n_terms']} terms "
          f"(source={m['source']}).")
    print(f"  frozen  -> {vg.VOCAB_JSON}")
    print(f"  literals-> {vg.LITERALS_PY}")
    print(f"  report  -> {vg.REPORT_MD}")

    # --- per-dimension summary ---
    print("\n--- dimensions ---")
    for name, dv in vocab["dimensions"].items():
        vals = [t["value"] for t in dv["terms"]]
        print(f"  {name:20s} ({len(vals):2d}) {', '.join(vals[:8])}"
              + (" ..." if len(vals) > 8 else ""))

    # --- checks ---
    print("\n--- verification ---")
    ok = True

    spine = {d.name for d in cp.CONTEXT_PROFILE}
    covered = set(vocab["dimensions"])
    missing = spine - covered
    print(f"  [{'ok' if not missing else 'XX'}] all {len(spine)} spine dimensions covered"
          + (f" (MISSING {missing})" if missing else ""))
    ok &= not missing

    no_unknown = [n for n, dv in vocab["dimensions"].items()
                  if "unknown" not in [t["value"] for t in dv["terms"]]]
    print(f"  [{'ok' if not no_unknown else 'XX'}] every dimension has an 'unknown' "
          f"value" + (f" (MISSING in {no_unknown})" if no_unknown else ""))
    ok &= not no_unknown

    # generated_vocab.py imports and matches the frozen JSON
    try:
        gv = importlib.import_module("generated_vocab")
        importlib.reload(gv)
        import typing
        gv_ok = True
        for name, dv in vocab["dimensions"].items():
            lit = getattr(gv, vg._camel(name))
            if set(typing.get_args(lit)) != {t["value"] for t in dv["terms"]}:
                gv_ok = False
        print(f"  [{'ok' if gv_ok else 'XX'}] generated_vocab.py Literals match the "
              f"frozen JSON")
        ok &= gv_ok
    except Exception as e:
        print(f"  [XX] generated_vocab.py failed to import: {type(e).__name__}: {e}")
        ok = False

    # normalization folds free text onto controlled values
    print("\n--- normalization demo (free text -> controlled value) ---")
    cases = [("workout_type", "jog"), ("workout_type", "swimming"),
             ("weather", "muggy"), ("occupation", "desk job"),
             ("heart_health", "very fit"), ("sleep", "restless"),
             ("workout_type", "quidditch")]
    for dim, text in cases:
        val = vg.normalize(dim, text, vocab)
        print(f"  {dim:16s} {text!r:14s} -> {val}")

    # round-trip reload
    reloaded = vg.load_vocabulary()
    rt = reloaded["meta"]["n_terms"] == m["n_terms"]
    print(f"\n  [{'ok' if rt else 'XX'}] vocabulary.json round-trips ({reloaded['meta']['n_terms']} terms)")
    ok &= rt

    print("\n" + ("PASS — Task 1 vocabulary built and verified."
                  if ok else "FAIL — see [XX] above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
