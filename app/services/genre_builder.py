# app/services/genre_builder.py
from __future__ import annotations

import ast
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import pandas as pd

# ----------------------------
# Parsing helpers
# ----------------------------

def parse_list(x) -> List[str]:
    """
    Parse a Python-list-like string from CSV (e.g. "['a','b']") into a list.
    Returns [] on any failure.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, list):
            return [str(v) for v in out if str(v).strip()]
        return []
    except Exception:
        return []


# ----------------------------
# Genre enrichment (existing)
# ----------------------------

def build_artist_genre_lookup(
    artists_df: pd.DataFrame,
    artist_id_col: str = "id",
    genres_col: str = "genres",
) -> Dict[str, List[str]]:
    """
    Build dict: artist_id -> list of genres (possibly empty).
    """
    if artist_id_col not in artists_df.columns:
        raise ValueError(f"artists_df missing column '{artist_id_col}'")
    if genres_col not in artists_df.columns:
        raise ValueError(f"artists_df missing column '{genres_col}'")

    tmp = artists_df[[artist_id_col, genres_col]].copy()
    tmp[artist_id_col] = tmp[artist_id_col].astype(str)
    tmp["genres_list"] = tmp[genres_col].apply(parse_list)

    return dict(zip(tmp[artist_id_col], tmp["genres_list"]))


def _merge_track_genres(
    artist_ids: List[str],
    artist_to_genres: Dict[str, List[str]],
) -> List[str]:
    """
    For a list of artist IDs, return unique genres union (order stable-ish).
    """
    seen = set()
    out: List[str] = []
    for aid in artist_ids:
        for g in artist_to_genres.get(str(aid), []):
            if g not in seen:
                seen.add(g)
                out.append(g)
    return out


def enrich_tracks_with_genres(
    tracks_df: pd.DataFrame,
    artist_to_genres: Dict[str, List[str]],
    track_artist_ids_col: str = "id_artists",
    add_genres_list_col: str = "genres_list",
    add_genres_str_col: str = "genres_str",
    sep: str = "|",
) -> pd.DataFrame:
    """
    Add genres_list + genres_str to tracks_df using artist_to_genres lookup.
    - tracks_df[track_artist_ids_col] is expected to be a list-like string: "['id1','id2']"
    """
    if track_artist_ids_col not in tracks_df.columns:
        raise ValueError(f"tracks_df missing column '{track_artist_ids_col}'")

    df = tracks_df.copy()
    df["_artist_ids_list"] = df[track_artist_ids_col].apply(parse_list)

    df[add_genres_list_col] = df["_artist_ids_list"].apply(
        lambda ids: _merge_track_genres(ids, artist_to_genres)
    )

    # cache string for fast contains filtering
    df[add_genres_str_col] = df[add_genres_list_col].apply(
        lambda gs: sep.join(gs) if gs else ""
    )

    df.drop(columns=["_artist_ids_list"], inplace=True, errors="ignore")
    return df


def compute_ui_genres(
    tracks_df: pd.DataFrame,
    genres_list_col: str = "genres_list",
    min_count: int = 50,
    top_k: int = 200,
) -> Tuple[List[str], Counter]:
    """
    Compute the list of genres to show in the UI dropdown.
    Rule:
      - keep genres with count >= min_count
      - then take top_k by frequency
    Returns:
      (ui_genres, genre_counter)
    """
    if genres_list_col not in tracks_df.columns:
        raise ValueError(f"tracks_df missing column '{genres_list_col}'")

    c: Counter = Counter()
    for gs in tracks_df[genres_list_col]:
        if not gs:
            continue
        c.update(gs)

    kept = [g for g, n in c.items() if n >= int(min_count)]
    kept_sorted = sorted(kept, key=lambda g: c[g], reverse=True)[: int(top_k)]
    return kept_sorted, c


def build_enriched_tracks_and_ui_genres(
    tracks_csv_path: str,
    artists_csv_path: str,
    *,
    track_artist_ids_col: str = "id_artists",
    artist_id_col: str = "id",
    artist_genres_col: str = "genres",
    min_count: int = 50,
    top_k: int = 200,
) -> Tuple[pd.DataFrame, List[str], Counter]:
    """
    Convenience one-shot:
      - read CSVs
      - build lookup
      - enrich tracks with genres_list/genres_str
      - compute ui_genres
    """
    tracks = pd.read_csv(tracks_csv_path)
    artists = pd.read_csv(artists_csv_path)

    artist_to_genres = build_artist_genre_lookup(
        artists, artist_id_col=artist_id_col, genres_col=artist_genres_col
    )

    tracks_enriched = enrich_tracks_with_genres(
        tracks, artist_to_genres, track_artist_ids_col=track_artist_ids_col
    )

    ui_genres, counter = compute_ui_genres(
        tracks_enriched, min_count=min_count, top_k=top_k
    )

    return tracks_enriched, ui_genres, counter


# ----------------------------
# NEW: robust genre search (Task #2)
# ----------------------------

_TOKEN_SPLIT = re.compile(r"[\s\-_\/&]+")

def _norm_text(s: str) -> str:
    """
    Lowercase + strip + remove diacritics (accents).
    """
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower().strip()


def _tokens(q: str) -> List[str]:
    qn = _norm_text(q)
    if not qn:
        return []
    return [t for t in _TOKEN_SPLIT.split(qn) if t]


def search_genres(
    all_genres: Iterable[str],
    query: str,
    *,
    limit: int = 200,
) -> List[str]:
    """
    Genre search that supports:
      - substring matching (e.g. "italia" -> "pop italiano")
      - multi-token order-free matching (e.g. "brit pop" -> "britpop", "pop brit", ...)
    Ranking: exact > startswith(full query) > startswith(any token) > token-match > substring.
    """
    ts = _tokens(query)
    if not ts:
        return []

    qn = _norm_text(query)

    scored: List[Tuple[int, str]] = []

    for g in all_genres:
        gs = "" if g is None else str(g)
        gn = _norm_text(gs)
        if not gn:
            continue

        if len(ts) == 1:
            ok = ts[0] in gn
        else:
            ok = all(t in gn for t in ts)

        if not ok:
            continue

        # lower score = better
        if gn == qn:
            score = 0
        elif qn and gn.startswith(qn):
            score = 5
        elif any(gn.startswith(t) for t in ts):
            score = 10
        elif len(ts) > 1:
            score = 20
        else:
            score = 30

        scored.append((score, gs))

    scored.sort(key=lambda x: (x[0], x[1].lower()))
    return [g for _, g in scored[: int(limit)]]
