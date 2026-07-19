"""
location_context.py - shared, isolated module that turns workout GPS into
map-grounded PLACE context.

Why this exists
---------------
File 2 already clusters workout GPS (DBSCAN/haversine) and labels each cluster
with a hard-coded time-of-day rule (`home` = cluster with most 00-05h points,
`work` = 09-17h weekday, else `outdoor`). That rule is fragile and gives no real
place identity, so >99% of episodes stay `location_type='unknown'` and the few
that don't are coarse. It is flagged `TODO(user-input)` in File 2 itself.

This module upgrades exactly that step, WITHOUT touching File 1-4:
  1. cluster the GPS the same way File 2 does (so cluster ids stay comparable),
  2. reverse-geocode each cluster CENTROID once (cached Nominatim; a handful of
     calls, not one per point) into a real place name + OSM category/type,
  3. classify that OSM tag into a richer, controlled `location_type` vocabulary
     (adds `park` and `water` - the latter matters here: this subject rows a lot,
     which happens on water), keeping the raw place phrase in `location_place`,
  4. derive a `home_climate` descriptor (home suburb + its weather band) for
     `context_baseline.SubjectContext`, which currently has no geocode source.

Design is ported from `enrichment_experiment/ee_geocode.py` (same OSM usage
policy: <=1 req/s, real User-Agent, on-disk cache keyed by rounded coord) but is
SELF-CONTAINED so this track has no cross-dependency on the enrichment track.

Isolation contract (same spirit as context_baseline/):
  - reads  : data/apple_health_export/workout-routes/*.gpx, data/processed/*.parquet
  - writes : data/location_context/geocode_cache/, results/location_context/
  - never imports or edits notebooks 01-04 / 03b / enrichment_experiment.

Robust when offline / no GPS (project guardrail): every function degrades to the
File-2 heuristic or to `unknown` and NEVER crashes.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & constants (self-contained; no ee_common dependency).
# --------------------------------------------------------------------------- #
TZ = "Australia/Adelaide"
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
GPX_DIR = os.path.join(REPO_ROOT, "data", "apple_health_export", "workout-routes")
PROC_DIR = os.path.join(REPO_ROOT, "data", "processed")
CACHE_DIR = os.path.join(REPO_ROOT, "data", "location_context", "geocode_cache")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "location_context")

ADELAIDE_CBD = (-34.9285, 138.6007)   # fallback coordinate when there is no GPS
EARTH_M = 6_371_000.0

# Nominatim reverse-geocode client settings (OSM usage policy).
_BASE = "https://nominatim.openstreetmap.org/reverse"
_UA = "location-context/0.1 (health-pattern-intelligence; contact tminhtan18@gmail.com)"
_MIN_INTERVAL = 1.1     # seconds between uncached requests (policy: <= 1 req/s)
_ROUND = 4              # coord rounding for cache key (~11 m) - dedups nearby centroids
_ZOOM = 16             # area granularity: returns the enclosing park/beach/suburb
                        # (not the nearest bench) -> far better for classification
_last_call = [0.0]      # module-level throttle clock


def ensure_dirs() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. GPX parsing (subsampled - clustering only needs the shape of the tracks).
# --------------------------------------------------------------------------- #
def parse_gpx(path: str, every_s: float = 60.0) -> pd.DataFrame:
    """Return [time(tz-aware), lat, lon] for one GPX file, subsampled to ~1 pt /
    `every_s` seconds. GPX time is UTC 'Z'; convert to local TZ."""
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    rows = []
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return pd.DataFrame(columns=["time", "lat", "lon"])
    last_t = None
    for pt in root.findall(".//gpx:trkpt", ns):
        lat, lon = pt.get("lat"), pt.get("lon")
        te = pt.find("gpx:time", ns)
        if not (lat and lon and te is not None and te.text):
            continue
        t = pd.to_datetime(te.text, utc=True)
        if last_t is not None and (t - last_t).total_seconds() < every_s:
            continue
        last_t = t
        rows.append((t, float(lat), float(lon)))
    d = pd.DataFrame(rows, columns=["time", "lat", "lon"])
    if len(d):
        d["time"] = d["time"].dt.tz_convert(TZ)
    return d


def load_all_gps(gpx_dir: str = GPX_DIR, every_s: float = 60.0) -> pd.DataFrame:
    """Parse every GPX file under `gpx_dir` into one [time, lat, lon] frame."""
    files = sorted(glob.glob(os.path.join(gpx_dir, "*.gpx")))
    parts = [parse_gpx(f, every_s) for f in files]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["time", "lat", "lon"])
    return pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Clustering (identical method to File 2 so cluster semantics stay aligned).
# --------------------------------------------------------------------------- #
def cluster_gps(df_gps: pd.DataFrame, eps_m: float = 100.0, min_samples: int = 5
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DBSCAN on radians of (lat, lon) with the haversine metric (same as File 2).
    Returns (df_gps_with_cluster, centroids[cluster,lat,lon,n_points]). `-1` is
    noise. Falls back to a single empty result when there is no GPS."""
    from sklearn.cluster import DBSCAN
    df = df_gps.copy()
    if not len(df):
        df["cluster"] = pd.Series(dtype=int)
        return df, pd.DataFrame(columns=["cluster", "lat", "lon", "n_points"])
    coords_rad = np.radians(df[["lat", "lon"]].values)
    df["cluster"] = DBSCAN(eps=eps_m / EARTH_M, min_samples=min_samples,
                           metric="haversine").fit_predict(coords_rad)
    real = df[df["cluster"] != -1]
    cent = (real.groupby("cluster")[["lat", "lon"]].mean()
            .assign(n_points=real.groupby("cluster").size())
            .reset_index())
    cent = cent.sort_values("n_points", ascending=False).reset_index(drop=True)
    return df, cent


# --------------------------------------------------------------------------- #
# 3. Cached Nominatim reverse geocoder (self-contained; OSM policy respected).
# --------------------------------------------------------------------------- #
def _round_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(float(lat), _ROUND), round(float(lon), _ROUND))


def _cache_path(latr: float, lonr: float) -> str:
    # zoom is part of the key: results differ by zoom, so caches must not collide.
    return os.path.join(CACHE_DIR, f"{latr:.4f}_{lonr:.4f}_z{_ZOOM}.json")


def _throttle() -> None:
    dt = time.time() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.time()


def reverse_geocode(lat: float, lon: float, online: bool = True) -> dict:
    """Reverse-geocode one coord, cached by rounded coord. Returns {} for NaN.
    `online=False` (or any network error) returns a cached hit if present, else
    {} - so the whole pipeline still runs offline, just without place names."""
    if lat is None or lon is None or lat != lat or lon != lon:   # NaN guard
        return {}
    ensure_dirs()
    latr, lonr = _round_key(lat, lon)
    cp = _cache_path(latr, lonr)
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            return json.load(fh)
    if not online:
        return {}
    q = urllib.parse.urlencode({"lat": latr, "lon": lonr, "format": "jsonv2",
                                "zoom": _ZOOM, "addressdetails": 1})
    req = urllib.request.Request(f"{_BASE}?{q}", headers={"User-Agent": _UA})
    try:
        _throttle()
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = json.loads(r.read())
    except Exception as e:                       # network/HTTP error -> cache the miss
        rec = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh)
        return rec
    rec = {
        "name": raw.get("name") or "",
        "category": raw.get("category"),
        "type": raw.get("type"),
        "addresstype": raw.get("addresstype"),
        "display_name": raw.get("display_name", ""),
        "address": raw.get("address", {}),
    }
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return rec


def place_phrase(rec: dict) -> str | None:
    """Compact human place name from a reverse-geocode record (name + up to two
    address components), e.g. 'West Lakes, Adelaide'. None when unavailable."""
    if not rec or rec.get("error"):
        return None
    dn = rec.get("display_name", "") or ""
    comps = [c.strip() for c in dn.split(",")]
    comps = [c for c in comps if c and not c.replace(" ", "").isdigit()]
    name = (rec.get("name") or "").strip()
    head = comps[:2]
    if name and (not head or name.lower() != head[0].lower()):
        head = [name] + [c for c in head if c.lower() != name.lower()][:1]
    phrase = ", ".join(head).strip(" ,")
    return phrase or None


# --------------------------------------------------------------------------- #
# 4. OSM tag -> controlled location_type vocabulary (transparent, auditable).
# --------------------------------------------------------------------------- #
# Controlled vocab (extends File 2's home/work/outdoor/gym/unknown, backward
# compatible - `park`/`water` are new and only ever ADD detail):
LOCATION_VOCAB = ("home", "work", "gym", "park", "water", "outdoor", "unknown")


# Curated whole-word text cues, applied to name + display_name. These survive the
# fact that Nominatim often returns a micro-feature category (a bench, a car park)
# even at zoom 16, while the enclosing name still says "... Beach"/"... Falls".
# Kept deliberately specific to avoid false positives (e.g. NOT "walk"/"gardens",
# which appear in street/suburb names like "Parkers Walk"/"Colonel Light Gardens").
_WATER_WORDS = ("beach", "bay", "lake", "river", "creek", "marina", "esplanade",
                "waterfront", "wharf", "jetty", "reservoir", "lagoon", "foreshore")
_PARK_WORDS = ("conservation park", "national park", "nature reserve", "reserve",
               "falls", "trail", "summit", "lookout")
_GYM_WORDS = ("gym", "fitness", "aquatic centre", "aquatic center", "leisure centre",
              "sports centre", "stadium", "velodrome")


def _word(text: str, kws) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in kws)


def classify_place(rec: dict) -> str:
    """Map an OSM reverse-geocode record to a controlled `location_type`.
    Reads the top-level OSM category/type (not the address dict), backed by a
    curated text cue over the place name. Roads default to `outdoor`; any geocoded
    workout spot that matches nothing is `outdoor` (workout GPS is outdoors by
    definition). No geocode -> `unknown`."""
    if not rec or rec.get("error"):
        return "unknown"
    cat = (rec.get("category") or "").lower()
    typ = (rec.get("type") or "").lower()
    addr = {k: str(v).lower() for k, v in (rec.get("address") or {}).items()}
    text = ((rec.get("name") or "") + " " + (rec.get("display_name") or "")).lower()

    # --- strong text cues first (robust to micro-feature category noise) ---
    if _word(text, _WATER_WORDS):
        return "water"
    if _word(text, _GYM_WORDS):
        return "gym"
    if _word(text, _PARK_WORDS):
        return "park"

    # --- authoritative OSM category/type (top-level fields) ---
    if cat == "natural" and typ in {"water", "bay", "beach", "wetland", "coastline", "strait"}:
        return "water"
    if cat == "leisure" and typ in {"marina", "swimming_area"}:
        return "water"
    if (cat == "leisure" and typ in {"sports_centre", "fitness_centre", "fitness_station",
                                     "swimming_pool", "stadium"}) or (cat == "amenity" and typ == "gym"):
        return "gym"
    if (cat == "leisure" and typ in {"park", "pitch", "garden", "playground",
                                     "recreation_ground", "golf_course", "nature_reserve"}) \
            or (cat == "natural" and typ in {"wood", "grassland", "heath", "scrub"}) \
            or addr.get("leisure") in {"park", "pitch", "garden"}:
        return "park"
    if cat == "office" or typ in {"commercial", "office", "industrial", "university",
                                  "college", "school", "hospital"} \
            or addr.get("landuse") in {"industrial", "commercial", "retail"}:
        return "work"
    # residential building/area -> home (but a residential ROAD is 'highway' -> outdoor)
    if (cat == "building" and typ in {"house", "residential", "apartments", "detached",
                                      "bungalow", "terrace", "semidetached_house"}) \
            or (cat == "place" and typ in {"house", "residential", "neighbourhood", "suburb"}) \
            or (cat != "highway" and addr.get("landuse") == "residential"):
        return "home"
    if cat == "highway" or typ in {"path", "footway", "cycleway", "pedestrian", "track"}:
        return "outdoor"
    return "outdoor"


# --------------------------------------------------------------------------- #
# 5. Build the per-cluster location table.
# --------------------------------------------------------------------------- #
def build_location_table(df_gps: pd.DataFrame, online: bool = True,
                         eps_m: float = 100.0) -> pd.DataFrame:
    """Cluster GPS, geocode each centroid once, classify it. Returns one row per
    cluster: [cluster, lat, lon, n_points, is_home_region, location_type,
    location_place, osm_category, osm_type]. Empty frame when there is no GPS."""
    _, cent = cluster_gps(df_gps, eps_m=eps_m)
    if not len(cent):
        return pd.DataFrame(columns=["cluster", "lat", "lon", "n_points",
                                     "is_home_region", "location_type",
                                     "location_place", "osm_category", "osm_type"])
    rows = []
    for i, c in cent.iterrows():
        rec = reverse_geocode(c["lat"], c["lon"], online=online)
        rows.append({
            "cluster": int(c["cluster"]),
            "lat": round(float(c["lat"]), 5),
            "lon": round(float(c["lon"]), 5),
            "n_points": int(c["n_points"]),
            "is_home_region": bool(i == 0),          # densest cluster = home region
            "location_type": classify_place(rec),
            "location_place": place_phrase(rec),
            "osm_category": rec.get("category"),
            "osm_type": rec.get("type"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. home_climate descriptor for context_baseline.SubjectContext.
# --------------------------------------------------------------------------- #
@dataclass
class HomeClimate:
    place: str | None                 # geocoded home-region name (or None offline)
    coord: tuple[float, float]        # (lat, lon) of the home region / fallback
    band: str                         # 'hot' | 'temperate' | 'cold' | 'unknown'
    temp_median: float | None = None
    temp_p10: float | None = None
    temp_p90: float | None = None
    source: str = "geocode+weather"


def derive_home_climate(loc_table: pd.DataFrame,
                        df_weather: pd.DataFrame | None = None) -> HomeClimate:
    """Home region = densest cluster (matches File 2's weather anchor). Its
    geocoded name gives the place; the weather distribution gives a coarse climate
    band. Degrades to Adelaide CBD + 'unknown' band when there is no GPS/weather."""
    if len(loc_table):
        home = loc_table.loc[loc_table["is_home_region"]].iloc[0]
        place, coord = home["location_place"], (float(home["lat"]), float(home["lon"]))
    else:
        place, coord = None, ADELAIDE_CBD

    band, med, p10, p90 = "unknown", None, None, None
    if df_weather is not None and len(df_weather) and "weather_temp" in df_weather:
        t = pd.to_numeric(df_weather["weather_temp"], errors="coerce").dropna()
        if len(t):
            med, p10, p90 = float(t.median()), float(t.quantile(.1)), float(t.quantile(.9))
            # coarse, non-diagnostic banding on the median daily temperature
            band = "hot" if med >= 22 else "cold" if med <= 12 else "temperate"
    return HomeClimate(place=place, coord=coord, band=band,
                       temp_median=med, temp_p10=p10, temp_p90=p90)


# --------------------------------------------------------------------------- #
# 7. Attach upgraded location to episodes (drop-in for File 2 Step 2b/3).
# --------------------------------------------------------------------------- #
# Only these geocoded types override an existing label. home/work/outdoor come
# from a behavioural (time-of-day) signal that geocoding a workout route cannot
# see, so we never overwrite them - geocoding only ADDS place identity.
RECREATION_TYPES = frozenset({"water", "park", "gym"})


def attach_location(ep: pd.DataFrame, df_gps: pd.DataFrame, loc_table: pd.DataFrame,
                    win: str = "15min") -> pd.DataFrame:
    """ADDITIVE upgrade of `ep`: keep the existing `location_type` (File 2's
    home/work/outdoor time-of-day labels) as the base, override ONLY with a
    geocoded recreation type (park/water/gym), and attach `location_place`
    wherever a window's GPS resolves to a cluster. Windows with no GPS overlap
    keep their base label (expected: no GPS off-workout). If `ep` has no prior
    `location_type`, the geocoded type is used as the base instead."""
    ep = ep.copy()
    dcol = "datetime" if "datetime" in ep.columns else "win_start"
    ep["_win"] = pd.to_datetime(ep[dcol]).dt.floor(win)

    ep["_cluster"] = -1
    df, _ = cluster_gps(df_gps)
    if len(df) and (df["cluster"] != -1).any():
        g = df[df["cluster"] != -1].copy()
        g["_win"] = g["time"].dt.floor(win)
        modal = g.groupby("_win")["cluster"].agg(lambda s: s.value_counts().idxmax())
        ep["_cluster"] = ep["_win"].map(modal).fillna(-1).astype(int)

    lut_type = dict(zip(loc_table["cluster"], loc_table["location_type"])) if len(loc_table) else {}
    lut_place = dict(zip(loc_table["cluster"], loc_table["location_place"])) if len(loc_table) else {}

    geo = ep["_cluster"].map(lambda c: lut_type.get(int(c)))          # geocoded type or None
    base = (ep["location_type"] if "location_type" in ep.columns
            else geo.fillna("unknown"))
    is_rec = geo.isin(RECREATION_TYPES)                               # only these override
    ep["location_type"] = base.where(~is_rec, geo)
    ep["location_place"] = ep["_cluster"].map(lambda c: lut_place.get(int(c)))
    return ep.drop(columns=["_win", "_cluster"])
