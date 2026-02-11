# app/services/region_genres.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------
# IMPORTANT:
# regions_results.json is expected to contain keys like:
#   "italian", "french", "german", ... and maybe also clusters
#   like "latin", "african", "mena", "usa".
#
# Each key maps to an object like:
# {
#   "seed_genres": [...],
#   "expanded_genres": {"genre": count, ...},   # dict OR list
#   "n_seed_authors": 1234,
#   ...
# }
# ------------------------------------------------------------

# ISO2 -> results.json key (adjective keys, NOT country names)
ISO2_TO_KEY: Dict[str, str] = {
    "IT": "italian",
    "FR": "french",
    "DE": "german",
    "ES": "spanish",
    "PT": "portuguese",
    "GB": "uk", #british?
    "IE": "irish",
    "US": "american",
    "CA": "canadian",
    "CH": "swiss",
    "AT": "austrian",
    "NL": "dutch",
    "BE": "belgian",
    "SE": "swedish",
    "NO": "norwegian",
    "DK": "danish",
    "FI": "finnish",
    "IS": "icelandic",
    "PL": "polish",
    "CZ": "czech",
    "HU": "hungarian",
    "RO": "romanian",
    "BG": "bulgarian",
    "GR": "greek",
    "TR": "turkish",
    "AL": "albanian",
    "AU": "australian",
    "IN": "indian",
    "JP": "japanese",
    "CN": "chinese",
    "MX": "mexican",
    "AR": "argentine",
    "BR": "brazilian",
    "CL": "chilean",
    "CO": "colombian",
    "PE": "peruvian",
    "VE": "venezuelan",
}

# Nice labels (UI)
KEY_TO_LABEL: Dict[str, str] = {
    "italian": "Italy",
    "french": "France",
    "german": "Germany",
    "spanish": "Spain",
    "portuguese": "Portugal",
    "uk": "United Kingdom",
    "irish": "Ireland",
    "american": "United States",
    "canadian": "Canada",
    "swiss": "Switzerland",
    "austrian": "Austria",
    "dutch": "Netherlands",
    "belgian": "Belgium",
    "swedish": "Sweden",
    "norwegian": "Norway",
    "danish": "Denmark",
    "finnish": "Finland",
    "icelandic": "Iceland",
    "polish": "Poland",
    "czech": "Czech Republic",
    "hungarian": "Hungary",
    "romanian": "Romania",
    "bulgarian": "Bulgaria",
    "greek": "Greece",
    "turkish": "Turkey",
    "albanian": "Albania",
    "australian": "Australia",
    "indian": "India",
    "japanese": "Japan",
    "chinese": "China",
    "mexican": "Mexico",
    "argentine": "Argentina",
    "brazilian": "Brazil",
    "chilean": "Chile",
    "colombian": "Colombia",
    "peruvian": "Peru",
    "venezuelan": "Venezuela",
    # clusters (se li vuoi in UI, altrimenti ignorali)
    "latin": "Latin America (cluster)",
    "african": "Africa (cluster)",
    "mena": "MENA (cluster)",
    "usa": "USA (cluster)",
}

_RESULTS_CACHE: Optional[dict] = None


def _load_results_json() -> dict:
    """
    Load cached results generated offline (notebook/script).
    Expected path: project_root/data/regions_results.json
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "data", "regions_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"regions_results.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_expanded_genres(expanded: Any) -> Dict[str, int]:
    """
    expanded_genres can be:
      - dict {genre: count}
      - list of [genre, count] pairs
      - anything else -> {}
    """
    if not expanded:
        return {}
    if isinstance(expanded, dict):
        out: Dict[str, int] = {}
        for k, v in expanded.items():
            g = str(k).strip()
            if not g:
                continue
            try:
                out[g] = int(v)
            except Exception:
                continue
        return out
    if isinstance(expanded, list):
        out = {}
        for item in expanded:
            try:
                g, c = item
                g = str(g).strip()
                if not g:
                    continue
                out[g] = int(c)
            except Exception:
                continue
        return out
    return {}


def get_region_payload(*, iso: str | None = None, key: str | None = None, top_n: int = 120) -> dict:
    """
    Returns:
      {
        ok: bool,
        iso: "IT",
        key: "italian",
        label: "Italy",
        n_seed_authors: int,
        genres: [ ... ]   # expanded first, then seeds (dedup)
      }
    """
    global _RESULTS_CACHE
    if _RESULTS_CACHE is None:
        _RESULTS_CACHE = _load_results_json()

    # resolve key from iso if provided
    if iso:
        iso = iso.upper().strip()
        key = ISO2_TO_KEY.get(iso)

    key = (key or "").strip().lower()
    if not key:
        return {"ok": False, "error": "Region not recognized", "iso": iso, "key": key}

    data = _RESULTS_CACHE.get(key)
    if not data:
        # helpful debug: show similar keys
        avail = sorted(list(_RESULTS_CACHE.keys()))
        return {
            "ok": False,
            "error": f"No data for key='{key}'",
            "iso": iso,
            "key": key,
            "available_keys_sample": avail[:30],
        }

    seed_genres = data.get("seed_genres", []) or []
    expanded_raw = data.get("expanded_genres", {}) or {}
    expanded = _coerce_expanded_genres(expanded_raw)

    expanded_sorted = sorted(expanded.items(), key=lambda kv: kv[1], reverse=True)
    expanded_top = [g for g, _c in expanded_sorted[: int(top_n)]]

    # union preserving order: expanded first, then seeds not already present
    seen = set()
    genres: List[str] = []
    for g in expanded_top + list(seed_genres):
        gg = (g or "").strip()
        if not gg or gg in seen:
            continue
        seen.add(gg)
        genres.append(gg)

    return {
        "ok": True,
        "iso": iso,
        "key": key,
        "label": KEY_TO_LABEL.get(key, key),
        "n_seed_authors": int(data.get("n_seed_authors", 0) or 0),
        "genres": genres,
    }
