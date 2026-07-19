"""
ee_geocode.py — reverse-geocode absolute coordinates into a short SEMANTIC place
phrase for the RQ1b C2 condition (E5). Turns a raw (lat, lon) into text like
    'Cardiff State Beach Parking, South Coast Highway 101 (OSM: amenity/parking)'
which is what the LLM sees — so we test whether *semantic* location (not raw
numbers) unlocks the enricher.

ISOLATED. Reads/writes only under data/enrichment_experiment/. Uses the public
OSM Nominatim reverse endpoint, respecting its usage policy:
  - <= 1 request/second (we sleep >= _MIN_INTERVAL between *uncached* calls),
  - a real User-Agent identifying the app,
  - on-disk cache keyed by rounded coord so we never re-hit a place twice
    (the run is fully reproducible from the cache once populated).

Design note (scientific honesty): reverse geocoding makes the C2 system an
"LLM + geocoder" hybrid. That is intentional and is itself the finding the plan
asks for (§9.4/§9.5): does giving the LLM a semantic location cue move its
beach/gym/restaurant recall off ~0? We feed the map's own place name + OSM
category/type and let the LLM do the mapping to the fixed schema — we do NOT
hard-code coord->schema rules (that would make the geocoder, not the LLM, the
thing under test).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import ee_common as ee

_BASE = "https://nominatim.openstreetmap.org/reverse"
_UA = "ee-enrichment-experiment/0.1 (academic research; contact tminhtan18@gmail.com)"
_MIN_INTERVAL = 1.1          # seconds between uncached requests (policy: <= 1 req/s)
_ROUND = 4                   # coord rounding for the cache key (~11 m); dedups repeats
_ZOOM = 18                   # building/POI granularity
_last_call = [0.0]           # module-level throttle clock


def round_key(lat: float, lon: float, nd: int = _ROUND) -> tuple[float, float]:
    return (round(float(lat), nd), round(float(lon), nd))


def _cache_path(latr: float, lonr: float):
    return ee.GEOCODE_CACHE_DIR / f"{latr:.4f}_{lonr:.4f}.json"


def _throttle():
    dt = time.time() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.time()


def _fetch(latr: float, lonr: float) -> dict:
    """Live Nominatim reverse call for one rounded coord (throttled)."""
    q = urllib.parse.urlencode({"lat": latr, "lon": lonr, "format": "jsonv2",
                                "zoom": _ZOOM, "addressdetails": 1})
    req = urllib.request.Request(f"{_BASE}?{q}", headers={"User-Agent": _UA})
    _throttle()
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def reverse(lat: float, lon: float) -> dict:
    """
    Return the parsed reverse-geocode result for (lat, lon), cached by rounded
    coord. Cache stores the fields we need + the phrase, so re-parsing is free.
    Returns {} for missing coords (NaN) or on a hard failure.
    """
    if lat is None or lon is None or lat != lat or lon != lon:   # NaN guard
        return {}
    latr, lonr = round_key(lat, lon)
    cp = _cache_path(latr, lonr)
    if cp.exists():
        rec = json.loads(cp.read_text())
        if not rec.get("error"):                 # re-derive phrase from cached raw fields
            rec["phrase"] = phrase_from_result(rec)  # (lets phrasing evolve w/o refetch)
        return rec
    try:
        raw = _fetch(latr, lonr)
    except Exception as e:                       # network/HTTP error -> cache the miss
        rec = {"error": f"{type(e).__name__}: {str(e)[:120]}", "phrase": None}
        cp.write_text(json.dumps(rec))
        return rec
    rec = {
        "name": raw.get("name") or "",
        "category": raw.get("category"),
        "type": raw.get("type"),
        "addresstype": raw.get("addresstype"),
        "display_name": raw.get("display_name", ""),
        "address": raw.get("address", {}),
    }
    rec["phrase"] = phrase_from_result(rec)
    cp.write_text(json.dumps(rec))
    return rec


# Address tags that carry a usable semantic place cue (priority high -> low).
_SEMANTIC_TAGS = ["natural", "leisure", "tourism", "amenity", "shop", "historic",
                  "landuse", "building"]


def phrase_from_result(rec: dict) -> str | None:
    """
    Compact human phrase from a reverse-geocode record. Combines the map's place
    NAME (often the strongest cue, e.g. '... State Beach ...') + OSM category/type
    + a semantic address tag, WITHOUT collapsing to the target vocabulary.
    """
    if not rec or rec.get("error"):
        return None
    dn = rec.get("display_name", "") or ""
    comps = [c.strip() for c in dn.split(",")]
    comps = [c for c in comps if c and not c.replace(" ", "").isdigit()]
    head = ", ".join(comps[:3])

    cat, typ = rec.get("category"), rec.get("type")
    bits = []
    if typ and typ != "yes":
        bits.append(f"{cat}/{typ}" if cat else str(typ))
    elif cat and cat not in ("building", "place", "highway", "boundary"):
        bits.append(str(cat))

    addr = rec.get("address", {}) or {}
    head_lc = head.lower()
    for k in _SEMANTIC_TAGS:                      # add one extra semantic tag if informative
        v = addr.get(k)
        if not v or v == "yes":
            continue
        vs = str(v)
        if vs.lower() == str(typ).lower() or vs.lower() in head_lc:
            continue                              # already conveyed by type or the name/head
        bits.append(f"{k}: {vs}")
        break

    tag = f" (OSM: {'; '.join(bits)})" if bits else ""
    phrase = (head + tag).strip(" ,")
    return phrase or None


def geocode_index(coords: dict) -> dict:
    """
    coords: {index_key: (lat, lon)}  ->  {index_key: phrase_or_None}.
    Progress is printed every 50 live calls; everything is cached so a rerun
    resumes instantly. index_key is opaque (we use the (uuid, timestamp) tuple).
    """
    ee.ensure_output_dirs()
    out = {}
    live = 0
    n = len(coords)
    for i, (k, (lat, lon)) in enumerate(coords.items(), 1):
        latr, lonr = (None, None)
        if lat == lat and lon == lon:            # not NaN
            latr, lonr = round_key(lat, lon)
        cached = latr is not None and _cache_path(latr, lonr).exists()
        rec = reverse(lat, lon)
        out[k] = rec.get("phrase")
        if not cached and latr is not None:
            live += 1
            if live % 50 == 0:
                print(f"  geocoded {i}/{n} rows ({live} live calls)…", flush=True)
    print(f"  geocode done: {n} rows, {live} live calls, "
          f"{sum(v is not None for v in out.values())} with a phrase.", flush=True)
    return out
