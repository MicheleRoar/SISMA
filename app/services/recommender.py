# app/services/recommender.py
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_store import DataStore


# ----------------------------
# Helpers: robust parsing
# ----------------------------
def _parse_listish(x) -> List[str]:
    """
    Accepts:
      - real Python lists
      - stringified lists like "['Rihanna', 'JAY-Z']"
    Returns list of cleaned strings, or [].
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, float) and pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, list):
                return [str(v).strip() for v in out if str(v).strip()]
        except Exception:
            return []
    return []


def normalize_artists_field(x) -> str:
    """
    Normalizes artists into a semicolon-separated string for display:
      ["Rihanna","JAY-Z"] -> "Rihanna; JAY-Z"
    If it's not list-ish, returns the raw string trimmed.
    """
    lst = _parse_listish(x)
    if lst:
        return "; ".join(lst)
    s = "" if x is None else str(x)
    return s.strip()


def artists_list_from_field(x) -> List[str]:
    """
    Canonical token list for matching:
      - prefers list-ish artists
      - otherwise splits on common separators
    Output: lowercase trimmed tokens (dedup, order preserved).
    """
    lst = _parse_listish(x)
    if lst:
        toks = [str(a).strip().lower() for a in lst if str(a).strip()]
        seen, out = set(), []
        for t in toks:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    s = "" if x is None else str(x)
    s = s.strip()
    if not s:
        return []

    # split on typical separators if we don't have a list:
    parts = re.split(
        r"\s*;\s*|\s*,\s*|\s*/\s*|\s*&\s*|\s+feat\.?\s+|\s+featuring\s+",
        s,
        flags=re.IGNORECASE,
    )
    toks = [p.strip().lower() for p in parts if p and p.strip()]

    seen, out = set(), []
    for t in toks:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def primary_artist_from_field(x) -> str:
    """
    Primary artist (lowercase) used for max_per_artist constraint.
    - if list-ish: first element
    - else: first before ';'
    """
    lst = _parse_listish(x)
    if lst:
        return str(lst[0]).strip().lower()

    s = normalize_artists_field(x)
    if not s:
        return ""
    if ";" in s:
        return s.split(";", 1)[0].strip().lower()
    return s.strip().lower()


def two_genres_from_row(row) -> str:
    gs = row.get("genres_list", None)
    if isinstance(gs, list) and len(gs) > 0:
        return ", ".join([str(x) for x in gs[:2]])

    s = str(row.get("genres_str", "")).strip()
    if s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return ", ".join(parts[:2])

    return str(row.get("track_genre", "")).strip()


# ----------------------------
# Config
# ----------------------------
@dataclass
class RecommenderConfig:
    k: int = 50
    max_per_artist: int = 2

    weights: Dict[str, float] = None
    key_cyclic: bool = True

    # expansion steps (raw units per feature)
    expand_steps: Dict[str, Tuple[float, ...]] = None

    # pool sizing
    pool_multiplier: int = 10
    min_pool: int = 500

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "danceability": 1.4,
                "energy": 1.6,
                "valence": 1.4,
                "tempo": 1.2,
                "acousticness": 1.0,
                "instrumentalness": 1.0,
                "speechiness": 0.9,
                "loudness": 0.7,
                "liveness": 0.6,
                "mode": 0.25,
                "time_signature": 0.15,
                "key": 0.20,
            }

        if self.expand_steps is None:
            z01 = (0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.28)
            self.expand_steps = {
                "danceability": z01,
                "energy": z01,
                "valence": z01,
                "acousticness": z01,
                "instrumentalness": z01,
                "speechiness": (0.0, 0.02, 0.04, 0.07, 0.10, 0.15),
                "liveness": z01,
                "tempo": (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 28.0, 36.0),
                "loudness": (0.0, 1.5, 3.0, 5.0, 7.0, 9.0),
            }


class PlaylistRecommender:
    """
    Weighted-distance recommender with:
    - Range constraints (hard -> soft via controlled fallback)
    - Genre-universe expansion FIRST (token/substring widen)  [optional]
    - Feature expansion NEXT (non-tempo first, tempo last)    [optional]
    - Optional universe builder + big style pool builder (for Planner)

    IMPORTANT: for Planner, you typically want:
      - strict_semantics=True  (no genre widening / no semantic drift)
      - lock_tempo=True        (tempo is HARD, never expanded)
    """


    def _build_candidate_index(
        self,
        *,
        genre: str | None = None,
        genre_buckets: list[str] | None = None,
        ranges: dict | None = None,
        exclude_artists: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        strict_semantics: bool = False,
        lock_tempo: bool = False,
    ) -> np.ndarray:
        """
        HARD pre-filter stage for Discovery.
        This defines the candidate universe BEFORE scoring.

        Rules:
        - If lock_tempo: NEVER exit tempo range
        - If strict_semantics: NEVER exit genre universe
        - Exclusions are always HARD
        """

        df = self.df
        idx = np.arange(len(df), dtype=np.int64)

        # -----------------------------
        # HARD tempo filter
        # -----------------------------
        if lock_tempo and ranges and "tempo" in ranges:
            tmin, tmax = ranges["tempo"]
            if tmin is not None:
                idx = idx[df.iloc[idx]["tempo"].values >= tmin]
            if tmax is not None:
                idx = idx[df.iloc[idx]["tempo"].values <= tmax]

        # -----------------------------
        # HARD genre universe
        # -----------------------------
        if strict_semantics:
            genres = []

            if genre:
                genres = [genre]
            elif genre_buckets:
                genres = list(genre_buckets)

            if genres:
                genre_idx = self.store.get_row_indices_by_genres(genres)
                idx = np.intersect1d(idx, np.asarray(genre_idx, dtype=np.int64))

        # -----------------------------
        # HARD exclusions
        # -----------------------------
        if exclude_artists:
            mask = ~df.iloc[idx]["artists"].isin(exclude_artists)
            idx = idx[mask.values]

        if exclude_genres:
            mask = ~df.iloc[idx]["track_genre"].isin(exclude_genres)
            idx = idx[mask.values]

        return idx

    def __init__(self, store: DataStore, config: Optional[RecommenderConfig] = None):
        self.store = store
        self.config = config or RecommenderConfig()
        self.feature_names = self.store.get_feature_names()
        self._feat_idx = {f: i for i, f in enumerate(self.feature_names)}

        self.w = self._build_weight_vector(self.config.weights).astype(np.float32)
        self.X = self.store.get_X(scaled=True)  # (N, D)
        self.df = self.store.get_df()

        # --- PATCH: precompute artist tokens for exact membership matching ---
        if "artists" in self.df.columns:
            raw = self.df["artists"]
            self._artists_tokens = raw.apply(artists_list_from_field)          # Series[list[str]]
            self._primary_artist = raw.apply(primary_artist_from_field)        # Series[str]
            # fast contains via delimiter trick (no regex)
            self._artists_tokens_str = self._artists_tokens.apply(lambda xs: "|" + "|".join(xs) + "|")
        else:
            self._artists_tokens = pd.Series([[]] * len(self.df))
            self._primary_artist = pd.Series([""] * len(self.df))
            self._artists_tokens_str = pd.Series(["||"] * len(self.df))

    def _build_weight_vector(self, weights: Dict[str, float]) -> np.ndarray:
        w = np.ones(len(self.feature_names), dtype=np.float32)
        for f, val in (weights or {}).items():
            if f in self._feat_idx:
                w[self._feat_idx[f]] = float(val)
        return w

    # -----------------------------
    # Public API
    # -----------------------------
    def recommend(
        self,
        user_input: Dict[str, float],
        genre: str = "",
        k: Optional[int] = None,
        max_per_artist: Optional[int] = None,
        exclude_track_ids: Optional[set] = None,
        shuffle_within_top: bool = False,
        random_state: int = 42,
        allow_explicit: bool = False,
        dontcare: Optional[Dict[str, bool]] = None,
        weight_overrides: Optional[Dict[str, float]] = None,
        ranges: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        exclude_artists: Optional[List[str]] = None,
        exclude_genres: Optional[List[str]] = None,
        *,
        # NEW: Planner controls
        strict_semantics: bool = False,
        lock_tempo: bool = False,
    ) -> pd.DataFrame:
        """
        strict_semantics=True:
          - no genre widening; range fallback never changes the semantic universe
        lock_tempo=True:
          - tempo is HARD; fallback never expands tempo range
        """
        k = int(k or self.config.k)
        max_per_artist = int(max_per_artist or self.config.max_per_artist)
        exclude_track_ids = exclude_track_ids or set()
        ranges = ranges or {}

        # 1) HARD candidate universe (Discovery)
        idx = self._build_candidate_index(
        genre=genre,
        ranges=ranges,
        exclude_artists=exclude_artists,
        exclude_genres=exclude_genres,
        strict_semantics=strict_semantics,
        lock_tempo=lock_tempo,
        )

        # 2) explicit filter (still HARD, but applied ONCE)
        idx = self._filter_explicit_indices(idx, allow_explicit=allow_explicit)

        # 2) range engine with CONTROLLED fallback:
        target_ranges = self._normalize_ranges(ranges)
        expanded_ranges_finite = None
        if target_ranges:
            if strict_semantics:
                # no genre widening; only feature expansion (and maybe tempo locked)
                idx, expanded_ranges_finite = self._filter_with_controlled_fallback_no_genre(
                    idx=idx,
                    target_ranges=target_ranges,
                    k=k,
                    lock_tempo=lock_tempo,
                )
            else:
                # legacy behavior: may widen genre query + expand features
                idx, expanded_ranges_finite = self._filter_with_controlled_fallback(
                    idx=idx,
                    genre_query=genre,
                    target_ranges=target_ranges,
                    k=k,
                    allow_explicit=allow_explicit,
                    lock_tempo=lock_tempo,
                )

        # 3) distance target: for ranged features use midpoint
        user_input = dict(user_input or {})
        for f, (mn, mx) in target_ranges.items():
            if mn is not None and mx is not None:
                user_input[f] = float(0.5 * (mn + mx))

        u_scaled = self._user_vector_scaled(user_input)

        # per-request weights
        w = self.w.copy()
        if weight_overrides:
            for f, val in weight_overrides.items():
                if f in self._feat_idx:
                    w[self._feat_idx[f]] = float(val)
        if dontcare:
            for f, flag in dontcare.items():
                if flag and f in self._feat_idx:
                    w[self._feat_idx[f]] = 0.0

        # distances
        d = self._weighted_distance(self.X[idx], u_scaled, w)

        if shuffle_within_top:
            rng = np.random.default_rng(random_state)
            d = d + rng.normal(loc=0.0, scale=1e-4, size=d.shape).astype(np.float32)

        pool_size = min(len(idx), max(k * self.config.pool_multiplier, self.config.min_pool))
        top_pool_local = self._topk_indices(d, pool_size)
        top_pool_global = idx[top_pool_local]
        if shuffle_within_top:
            rng = np.random.default_rng(random_state)
            rng.shuffle(top_pool_global)

        selected = self._apply_constraints(
            top_pool_global,
            k=k,
            max_per_artist=max_per_artist,
            exclude_track_ids=exclude_track_ids,
            artist_caps=None,
        )

        cols = []
        for c in ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity", "genres_list", "genres_str"]:
            if c in self.df.columns:
                cols.append(c)

        out = self.df.iloc[selected][cols].copy()
        if "artists" in out.columns:
            out["artists"] = out["artists"].apply(normalize_artists_field)
        out["track_genre"] = out.apply(two_genres_from_row, axis=1)

        # --- BPM (integer) ---
        if "tempo" in self.df.columns:
            sel = np.asarray(selected, dtype=np.int64)
            out["bpm"] = (
                pd.to_numeric(self.df.iloc[sel]["tempo"], errors="coerce")
                .round()
                .astype("Int64")
                .to_numpy()
            )

        # --- MATCH ---
        sel_local_mask = np.isin(top_pool_global, selected)
        sel_d = d[top_pool_local][sel_local_mask]
        p10, p90 = np.percentile(d[top_pool_local], [10, 90])
        denom = (p90 - p10) if (p90 - p10) > 1e-9 else 1.0
        match_dist = 100.0 * (1.0 - np.clip((sel_d - p10) / denom, 0.0, 1.0))

        if target_ranges and expanded_ranges_finite is not None and len(selected) > 0:
            match_range = self._range_score_playlist_100(selected, target_ranges, expanded_ranges_finite)
            match = 0.75 * match_range + 0.25 * match_dist
        else:
            match = match_dist

        out["match"] = np.round(match, 2).astype(float)

        if "popularity" in out.columns:
            out["popularity"] = pd.to_numeric(out["popularity"], errors="coerce").fillna(0)
            out = out.sort_values(["popularity", "match"], ascending=[False, False])
        else:
            out = out.sort_values("match", ascending=False)

        return out.reset_index(drop=True)

    def recommend_bucketed(
        self,
        user_input: Dict[str, float],
        *,
        artist_buckets: Optional[List[str]] = None,
        genre_buckets: Optional[List[str]] = None,
        artist_weights: Optional[List[float]] = None,
        genre_weights: Optional[List[float]] = None,
        k: int = 50,
        allow_explicit: bool = False,
        max_per_artist: int = 2,
        exclude_track_ids: Optional[set] = None,
        shuffle_within_top: bool = False,
        random_state: int = 42,
        weight_overrides: Optional[Dict[str, float]] = None,
        dontcare: Optional[Dict[str, bool]] = None,
        fallback: bool = True,
        ranges: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        exclude_artists: Optional[List[str]] = None,
        exclude_genres: Optional[List[str]] = None,
        # NEW: Planner controls
        strict_semantics: bool = False,
        lock_tempo: bool = False,
    ) -> pd.DataFrame:
        """
        NOTE:
          - bucketed already uses explicit buckets; no genre widening is performed here.
          - lock_tempo=True => tempo is HARD (never expanded) when ranges are used.
        """

        exclude_track_ids = exclude_track_ids or set()
        artist_buckets = [str(a).strip() for a in (artist_buckets or []) if str(a).strip()]
        genre_buckets = [str(g).strip() for g in (genre_buckets or []) if str(g).strip()]
        k = int(k)
        max_per_artist = int(max_per_artist)


        # --- DISCOVERY SAFETY BELT (bucketed) ---
        # If the user selected genre buckets, NEVER leave that genre universe.
        allowed_genre_idx = None
        if strict_semantics and genre_buckets:
            parts = []
            for g in genre_buckets:
                gi = self.store.get_row_indices_by_genre(g)
                if gi is not None and len(gi) > 0:
                    parts.append(np.asarray(gi, dtype=np.int64))
            if parts:
                allowed_genre_idx = np.unique(np.concatenate(parts))

        if artist_buckets:
            if not artist_weights or len(artist_weights) != len(artist_buckets):
                artist_weights = [1.0] * len(artist_buckets)
        else:
            artist_weights = []

        if genre_buckets:
            if not genre_weights or len(genre_weights) != len(genre_buckets):
                genre_weights = [1.0] * len(genre_buckets)
        else:
            genre_weights = []

        ranges = ranges or {}
        target_ranges = self._normalize_ranges(ranges)

        user_input = dict(user_input or {})
        for f, (mn, mx) in target_ranges.items():
            if mn is not None and mx is not None:
                user_input[f] = float(0.5 * (mn + mx))
        u_scaled = self._user_vector_scaled(user_input)

        w_req = self.w.copy()
        if weight_overrides:
            for f, val in weight_overrides.items():
                if f in self._feat_idx:
                    w_req[self._feat_idx[f]] = float(val)
        if dontcare:
            for f, flag in dontcare.items():
                if flag and f in self._feat_idx:
                    w_req[self._feat_idx[f]] = 0.0

        selected_global: List[int] = []
        selected_tid: set = set(exclude_track_ids)

        def pick_from_candidates(
            candidate_idx: np.ndarray,
            need: int,
            *,
            artist_caps: Optional[Dict[str, int]] = None,
        ) -> Tuple[List[int], Optional[Dict[str, Tuple[float, float]]]]:
            candidate_idx = self._filter_explicit_indices(candidate_idx, allow_explicit=allow_explicit)
            candidate_idx = self._apply_exclusions(
                candidate_idx,
                exclude_artists=exclude_artists,
                exclude_genres=exclude_genres,
            )

            # --- SAFETY BELT: NEVER leave genre universe in strict mode ---
            if allowed_genre_idx is not None:
                candidate_idx = np.intersect1d(
                    np.asarray(candidate_idx, dtype=np.int64),
                    allowed_genre_idx,
                )

            if need <= 0 or candidate_idx.size == 0:
                return [], None

            expanded_ranges_finite = None
            if target_ranges:
                candidate_idx, expanded_ranges_finite = self._filter_with_controlled_fallback_no_genre(
                    idx=candidate_idx,
                    target_ranges=target_ranges,
                    k=need,
                    lock_tempo=lock_tempo,
                )

            d = self._weighted_distance(self.X[candidate_idx], u_scaled, w_req)
            if shuffle_within_top:
                rng = np.random.default_rng(random_state)
                d = d + rng.normal(0.0, 1e-4, size=d.shape).astype(np.float32)

            pool_size = min(len(candidate_idx), max(need * 15, self.config.min_pool))
            top_local = self._topk_indices(d, pool_size)
            top_global = candidate_idx[top_local]

            picked = self._apply_constraints(
                top_global,
                k=need,
                max_per_artist=max_per_artist,
                exclude_track_ids=selected_tid,
                artist_caps=artist_caps,
            )
            return picked, expanded_ranges_finite

        buckets: List[Dict[str, str]] = []
        weights: List[float] = []

        for a, w in zip(artist_buckets, artist_weights):
            buckets.append({"type": "artist", "name": a})
            weights.append(float(w))

        for g, w in zip(genre_buckets, genre_weights):
            buckets.append({"type": "genre", "name": g})
            weights.append(float(w))

        last_expanded = None

        if not buckets:
            cand = np.arange(len(self.df), dtype=np.int64)
            picked, last_expanded = pick_from_candidates(cand, k, artist_caps=None)
            selected_global.extend(picked)
        else:
            quotas = self._allocate_quotas(k, weights)
            for b, q in zip(buckets, quotas):
                if q <= 0:
                    continue

                if b["type"] == "artist":
                    cand = self._indices_by_artist_contains(b["name"])
                    caps = {b["name"].strip().lower(): int(q)}
                    picked, last_expanded = pick_from_candidates(cand, q, artist_caps=caps)
                else:
                    cand = self._indices_by_genre_contains(b["name"])
                    picked, last_expanded = pick_from_candidates(cand, q, artist_caps=None)

                for ii in picked:
                    tid = str(self.df.iloc[int(ii)].get("track_id", "")).strip()
                    if tid:
                        selected_tid.add(tid)
                selected_global.extend(picked)

            remaining = k - len(selected_global)
            if remaining > 0 and fallback:
                if allowed_genre_idx is not None:
                    cand = allowed_genre_idx
                else:
                    cand = np.arange(len(self.df), dtype=np.int64)

                picked, last_expanded = pick_from_candidates(
                    cand,
                    remaining,
                    artist_caps=None,
                )
                selected_global.extend(picked)

        cols = []
        for c in ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity", "genres_list", "genres_str"]:
            if c in self.df.columns:
                cols.append(c)

        out = self.df.iloc[selected_global][cols].copy()
        if "artists" in out.columns:
            out["artists"] = out["artists"].apply(normalize_artists_field)
        out["track_genre"] = out.apply(two_genres_from_row, axis=1)

        # --- BPM (integer) ---
        if "tempo" in self.df.columns:
            sel = np.asarray(selected_global, dtype=np.int64)
            out["bpm"] = (
                pd.to_numeric(self.df.iloc[sel]["tempo"], errors="coerce")
                .round()
                .astype("Int64")
                .to_numpy()
            )

        if target_ranges and last_expanded is not None and len(selected_global) > 0:
            match_range = self._range_score_playlist_100(selected_global, target_ranges, last_expanded)
            sel_idx = np.array(selected_global, dtype=np.int64)
            d_sel = self._weighted_distance(self.X[sel_idx], u_scaled, w_req)
            dmin = float(np.min(d_sel)) if len(d_sel) else 0.0
            dmax = float(np.max(d_sel)) if len(d_sel) else 1.0
            denom = (dmax - dmin) if (dmax - dmin) > 1e-9 else 1.0
            match_dist = 100.0 * (1.0 - (d_sel - dmin) / denom)
            out["match"] = np.round(0.75 * match_range + 0.25 * match_dist, 2)
        else:
            out["match"] = 0.0

        if "popularity" in out.columns:
            out["popularity"] = pd.to_numeric(out["popularity"], errors="coerce").fillna(0)
            out = out.sort_values(["popularity", "match"], ascending=[False, False])
        else:
            out = out.sort_values("match", ascending=False)

        return out.head(k).reset_index(drop=True)

    # -----------------------------
    # Range engine + Controlled fallback (core)
    # -----------------------------
    @staticmethod
    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _normalize_ranges(
        self,
        ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for f, (mn, mx) in (ranges or {}).items():
            if f not in self._feat_idx:
                continue

            a = None if mn is None else float(mn)
            b = None if mx is None else float(mx)

            if (a is not None) and (b is not None) and (a > b):
                a, b = b, a

            if f in {"danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"}:
                if a is not None:
                    a = self._clamp01(a)
                if b is not None:
                    b = self._clamp01(b)

            out[f] = (a, b)

        out = {f: (a, b) for f, (a, b) in out.items() if (a is not None) or (b is not None)}
        return out

    def _expanded_range(
        self,
        f: str,
        target: Tuple[Optional[float], Optional[float]],
        step: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        mn, mx = target
        if mn is None and mx is None:
            return (None, None)

        if mn is None:
            return (None, mx + step if mx is not None else None)
        if mx is None:
            return (mn - step if mn is not None else None, None)

        a = mn - step
        b = mx + step

        if f in {"danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"}:
            a = self._clamp01(a)
            b = self._clamp01(b)

        return (a, b)

    def _filter_by_ranges(
        self,
        idx: np.ndarray,
        ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    ) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0 or not ranges:
            return idx

        mask = np.ones(idx.size, dtype=bool)
        for f, (mn, mx) in ranges.items():
            if f not in self.df.columns:
                continue

            vals = pd.to_numeric(self.df.iloc[idx][f], errors="coerce").to_numpy(dtype=float)

            if mn is not None:
                mask &= (vals >= float(mn))
            if mx is not None:
                mask &= (vals <= float(mx))

            if not mask.any():
                return idx[:0]

        return idx[mask]

    def _make_finite_bounds(self, f: str, mn: Optional[float], mx: Optional[float]) -> Tuple[float, float]:
        col = f if f in self.df.columns else None
        if col is None:
            return (-np.inf, np.inf)

        vals = pd.to_numeric(self.df[col], errors="coerce").dropna()
        if vals.empty:
            return (-np.inf, np.inf)

        lo = float(vals.min()) if mn is None else float(mn)
        hi = float(vals.max()) if mx is None else float(mx)

        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    def _range_score_playlist_100(
        self,
        selected_indices: List[int],
        target_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
        expanded_ranges_finite: Dict[str, Tuple[float, float]],
    ) -> np.ndarray:
        sel = np.asarray(selected_indices, dtype=np.int64)
        if sel.size == 0 or not target_ranges:
            return np.zeros(sel.size, dtype=np.float32)

        feats = [f for f in target_ranges.keys() if f in self._feat_idx and f in self.df.columns]
        if not feats:
            return np.zeros(sel.size, dtype=np.float32)

        wsum = 0.0
        score = np.zeros(sel.size, dtype=np.float32)

        for f in feats:
            w = float(self.config.weights.get(f, 1.0))
            if w <= 0:
                continue

            mn_t, mx_t = target_ranges[f]
            lo_e, hi_e = expanded_ranges_finite.get(f, (-np.inf, np.inf))

            x = pd.to_numeric(self.df.iloc[sel][f], errors="coerce").fillna(np.nan).to_numpy(dtype=float)

            s_f = np.zeros(sel.size, dtype=np.float32)

            inside = np.ones(sel.size, dtype=bool)
            if mn_t is not None:
                inside &= (x >= float(mn_t))
            if mx_t is not None:
                inside &= (x <= float(mx_t))
            s_f[inside] = 1.0

            outside = ~inside & ~np.isnan(x)
            if outside.any():
                dist = np.zeros(sel.size, dtype=float)
                if mn_t is not None:
                    dist = np.where(x < float(mn_t), float(mn_t) - x, dist)
                if mx_t is not None:
                    dist = np.where(x > float(mx_t), x - float(mx_t), dist)

                room = np.zeros(sel.size, dtype=float)
                if mn_t is not None:
                    room = np.where(x < float(mn_t), float(mn_t) - float(lo_e), room)
                if mx_t is not None:
                    room = np.where(x > float(mx_t), float(hi_e) - float(mx_t), room)

                room = np.maximum(room, 1e-9)
                ratio = np.clip(dist / room, 0.0, 1.0)
                s_f = np.where(outside, (1.0 - ratio).astype(np.float32), s_f)

            score += (w * s_f)
            wsum += w

        if wsum <= 1e-9:
            return np.zeros(sel.size, dtype=np.float32)

        return (100.0 * (score / float(wsum))).astype(np.float32)

    # ---- genre widening (legacy) ----
    @staticmethod
    def _tokenize_genre_query(q: str) -> List[str]:
        s = (q or "").strip().lower()
        if not s:
            return []
        parts = [p.strip() for p in s.replace("/", " ").replace("-", " ").split() if p.strip()]
        return [p for p in parts if len(p) >= 3]

    def _genre_variants(self, q: str) -> List[str]:
        base = (q or "").strip()
        if not base:
            return [""]

        toks = self._tokenize_genre_query(base)
        out: List[str] = [base]

        if len(toks) >= 2:
            for i in range(len(toks)):
                v = " ".join([toks[j] for j in range(len(toks)) if j != i]).strip()
                if v:
                    out.append(v)

        for t in toks:
            out.append(t)

        seen = set()
        uniq: List[str] = []
        for x in out:
            xx = str(x).strip().lower()
            if xx and xx not in seen:
                seen.add(xx)
                uniq.append(str(x).strip())
        return uniq

    def _filter_with_controlled_fallback(
        self,
        *,
        idx: np.ndarray,
        genre_query: str,
        target_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
        k: int,
        allow_explicit: bool,
        lock_tempo: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """
        LEGACY path:
          1) strict ranges on current idx
          2) genre widening via variants
          3) feature expansion (tempo last, unless lock_tempo=True)
        """
        strict = self._filter_by_ranges(idx, target_ranges)
        if strict.size >= int(k):
            exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in target_ranges.items()}
            return strict, exp_for_score

        widened = np.asarray(idx, dtype=np.int64)
        for gv in self._genre_variants(genre_query):
            gi = self.store.get_row_indices_by_genre(gv)
            gi = self._filter_explicit_indices(gi, allow_explicit=allow_explicit)
            widened = np.unique(np.concatenate([widened, gi]).astype(np.int64))
            strict2 = self._filter_by_ranges(widened, target_ranges)
            if strict2.size >= int(k):
                exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in target_ranges.items()}
                return strict2, exp_for_score

        expanded_idx, exp_for_score = self._filter_with_controlled_feature_expansion(
            idx=widened,
            target_ranges=target_ranges,
            k=k,
            lock_tempo=lock_tempo,
        )
        return expanded_idx, exp_for_score

    def _filter_with_controlled_fallback_no_genre(
        self,
        *,
        idx: np.ndarray,
        target_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
        k: int,
        lock_tempo: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """
        Planner-safe path:
          1) strict ranges on current idx
          2) ONLY feature expansion (tempo expansion disabled if lock_tempo=True)
        """
        strict = self._filter_by_ranges(idx, target_ranges)
        if strict.size >= int(k):
            exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in target_ranges.items()}
            return strict, exp_for_score

        return self._filter_with_controlled_feature_expansion(
            idx=idx,
            target_ranges=target_ranges,
            k=k,
            lock_tempo=lock_tempo,
        )

    def _filter_with_controlled_feature_expansion(
        self,
        *,
        idx: np.ndarray,
        target_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
        k: int,
        lock_tempo: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, float]]]:
        """
        Expand non-tempo first; tempo last (unless lock_tempo=True).
        If lock_tempo=True and tempo is too tight => you may still end up with <k results (by design).
        """
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in target_ranges.items()}
            return idx, exp_for_score

        non_tempo = [f for f in target_ranges.keys() if f != "tempo"]
        tempo_in = "tempo" in target_ranges

        def step_for(f: str, i: int) -> float:
            steps = self.config.expand_steps.get(f, (0.0,))
            if i < len(steps):
                return float(steps[i])
            return float(steps[-1])

        base_ranges = dict(target_ranges)
        chosen_ranges = dict(base_ranges)

        max_len_non = 1
        for f in non_tempo:
            max_len_non = max(max_len_non, len(self.config.expand_steps.get(f, (0.0,))))

        best_non = None
        for i in range(max_len_non):
            expanded = dict(base_ranges)
            for f in non_tempo:
                expanded[f] = self._expanded_range(f, base_ranges[f], step_for(f, i))
            filtered = self._filter_by_ranges(idx, expanded)
            best_non = expanded
            if filtered.size >= int(k):
                exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in expanded.items()}
                return filtered, exp_for_score

        if best_non is not None:
            chosen_ranges = best_non

        # If tempo not requested => done
        if not tempo_in:
            filtered = self._filter_by_ranges(idx, chosen_ranges)
            exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in chosen_ranges.items()}
            return filtered, exp_for_score

        # If tempo is requested but locked => never expand tempo
        if lock_tempo:
            filtered = self._filter_by_ranges(idx, chosen_ranges)  # tempo remains original bounds
            exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in chosen_ranges.items()}
            return filtered, exp_for_score

        # Otherwise expand tempo as last resort (legacy)
        max_len_t = len(self.config.expand_steps.get("tempo", (0.0,)))
        best_all = None
        for j in range(max_len_t):
            expanded2 = dict(chosen_ranges)
            expanded2["tempo"] = self._expanded_range("tempo", target_ranges["tempo"], step_for("tempo", j))
            filtered2 = self._filter_by_ranges(idx, expanded2)
            best_all = expanded2
            if filtered2.size >= int(k):
                exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in expanded2.items()}
                return filtered2, exp_for_score

        if best_all is None:
            best_all = dict(chosen_ranges)
        filtered = self._filter_by_ranges(idx, best_all)
        exp_for_score = {f: self._make_finite_bounds(f, a, b) for f, (a, b) in best_all.items()}
        return filtered, exp_for_score

    # -----------------------------
    # Bucket helpers (PATCHED ARTIST MATCH)
    # -----------------------------
    def _indices_by_artist_contains(self, artist_query: str) -> np.ndarray:
        """
        PATCH:
        - Exact-token match across ALL artists (so "rihanna" matches duets/features)
        - NO false positives like Mandrake / Nick Drake / Pete Drake
        - Soft fallback contains() only if exact-token yields zero rows
        """
        q = (artist_query or "").strip().lower()
        if not q:
            return np.arange(len(self.df), dtype=np.int64)

        needle = f"|{q}|"
        mask_any = self._artists_tokens_str.astype(str).str.contains(needle, regex=False)
        idx = np.flatnonzero(mask_any.to_numpy())

        if idx.size == 0 and "artists" in self.df.columns:
            s = self.df["artists"].astype(str).str.lower()
            idx = np.flatnonzero(s.str.contains(q, regex=False).to_numpy())

        return idx.astype(np.int64)

    def _indices_by_genre_contains(self, genre_query: str) -> np.ndarray:
        return self.store.get_row_indices_by_genre(genre_query)

    @staticmethod
    def _allocate_quotas(k: int, weights: List[float]) -> List[int]:
        w = np.array([max(0.0, float(x)) for x in (weights or [])], dtype=np.float64)
        if w.size == 0:
            return []
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()

        raw = w * int(k)
        base = np.floor(raw).astype(int)
        rem = int(k) - int(base.sum())
        if rem <= 0:
            return base.tolist()

        frac = raw - base
        order = np.argsort(-frac)
        for i in range(rem):
            base[int(order[i % len(order)])] += 1

        return base.tolist()

    def _filter_explicit_indices(self, idx: np.ndarray, allow_explicit: bool) -> np.ndarray:
        if allow_explicit:
            return idx
        if "explicit" not in self.df.columns:
            return idx

        exp = pd.to_numeric(self.df["explicit"], errors="coerce").fillna(0).astype(int).to_numpy()
        idx = np.asarray(idx, dtype=np.int64)
        return idx[(exp[idx] == 0)]

    def _apply_exclusions(
        self,
        idx: np.ndarray,
        *,
        exclude_artists: Optional[List[str]] = None,
        exclude_genres: Optional[List[str]] = None,
    ) -> np.ndarray:
        exclude_artists = [str(x).strip() for x in (exclude_artists or []) if str(x).strip()]
        exclude_genres = [str(x).strip() for x in (exclude_genres or []) if str(x).strip()]

        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return idx

        # --- artist ban ---
        if exclude_artists:
            parts = [self._indices_by_artist_contains(a) for a in exclude_artists]
            ex_idx = np.unique(np.concatenate(parts).astype(np.int64)) if parts else np.array([], dtype=np.int64)
            if ex_idx.size:
                idx = idx[~np.isin(idx, ex_idx)]

        # --- genre ban ---
        if exclude_genres:
            parts = [self._indices_by_genre_contains(g) for g in exclude_genres]
            ex_idx = np.unique(np.concatenate(parts).astype(np.int64)) if parts else np.array([], dtype=np.int64)
            if ex_idx.size:
                idx = idx[~np.isin(idx, ex_idx)]

        return idx.astype(np.int64)

    # -----------------------------
    # Internals: scaling + distance + constraints
    # -----------------------------
    def _user_vector_scaled(self, user_input: Dict[str, float]) -> np.ndarray:
        stdizer = self.store.standardizer
        mean = stdizer.mean.astype(np.float32)
        std = stdizer.std.astype(np.float32)

        raw = mean.copy()
        for f, v in (user_input or {}).items():
            if f in self._feat_idx and v is not None:
                raw[self._feat_idx[f]] = float(v)

        u = (raw - mean) / std
        return u.astype(np.float32)

    def _weighted_distance(self, Xcand: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        diff = Xcand - u

        if self.config.key_cyclic and "key" in self._feat_idx:
            ki = self._feat_idx["key"]
            stdizer = self.store.standardizer
            mean = float(stdizer.mean[ki])
            std = float(stdizer.std[ki])

            key_raw_cand = (Xcand[:, ki] * std + mean)
            key_raw_u = (u[ki] * std + mean)

            unknown_mask = (key_raw_cand < 0) | (key_raw_u < 0)

            a = np.mod(key_raw_cand, 12.0)
            b = np.mod(key_raw_u, 12.0)
            cyc = np.minimum(np.abs(a - b), 12.0 - np.abs(a - b))
            cyc_norm = (cyc / 6.0).astype(np.float32)
            diff[:, ki] = np.where(unknown_mask, 0.0, cyc_norm).astype(np.float32)

        diff_w = diff * w
        return np.sqrt(np.sum(diff_w * diff_w, axis=1)).astype(np.float32)

    @staticmethod
    def _topk_indices(d: np.ndarray, k: int) -> np.ndarray:
        if k >= len(d):
            return np.argsort(d)
        part = np.argpartition(d, kth=k - 1)[:k]
        return part[np.argsort(d[part])]

    def _apply_constraints(
        self,
        candidate_indices: np.ndarray,
        k: int,
        max_per_artist: int,
        exclude_track_ids: set,
        artist_caps: Optional[Dict[str, int]] = None,
    ) -> List[int]:
        selected: List[int] = []
        artist_counts: Dict[str, int] = {}

        tid_col = "track_id" if "track_id" in self.df.columns else ("id" if "id" in self.df.columns else None)

        for idx in candidate_indices:
            row = self.df.iloc[int(idx)]

            if tid_col is not None:
                tid = str(row.get(tid_col, "")).strip()
                if tid and tid in exclude_track_ids:
                    continue

            a = primary_artist_from_field(row.get("artists", ""))
            if a:
                cap = int(max_per_artist)
                if artist_caps and a in artist_caps:
                    cap = int(artist_caps[a])

                c = artist_counts.get(a, 0)
                if c >= cap:
                    continue
                artist_counts[a] = c + 1

            selected.append(int(idx))
            if len(selected) >= int(k):
                break

        return selected

    # -----------------------------
    # Universe builder + Pool builder (Planner foundation)
    # -----------------------------
    def build_universe_indices(
        self,
        *,
        include_artists: Optional[List[str]] = None,
        include_genres: Optional[List[str]] = None,
        exclude_artists: Optional[List[str]] = None,
        exclude_genres: Optional[List[str]] = None,
    ) -> np.ndarray:
        include_artists = [str(x).strip() for x in (include_artists or []) if str(x).strip()]
        include_genres = [str(x).strip() for x in (include_genres or []) if str(x).strip()]
        exclude_artists = [str(x).strip() for x in (exclude_artists or []) if str(x).strip()]
        exclude_genres = [str(x).strip() for x in (exclude_genres or []) if str(x).strip()]

        if not include_artists and not include_genres:
            universe = np.arange(len(self.df), dtype=np.int64)
        else:
            parts = []
            for a in include_artists:
                parts.append(self._indices_by_artist_contains(a))
            for g in include_genres:
                parts.append(self._indices_by_genre_contains(g))
            universe = (
                np.unique(np.concatenate(parts).astype(np.int64))
                if parts
                else np.arange(len(self.df), dtype=np.int64)
            )

        if exclude_artists:
            ex_parts = [self._indices_by_artist_contains(a) for a in exclude_artists]
            ex_idx = np.unique(np.concatenate(ex_parts).astype(np.int64)) if ex_parts else np.array([], dtype=np.int64)
            if ex_idx.size:
                universe = universe[~np.isin(universe, ex_idx)]

        if exclude_genres:
            ex_parts = [self._indices_by_genre_contains(g) for g in exclude_genres]
            ex_idx = np.unique(np.concatenate(ex_parts).astype(np.int64)) if ex_parts else np.array([], dtype=np.int64)
            if ex_idx.size:
                universe = universe[~np.isin(universe, ex_idx)]

        return universe.astype(np.int64)

    def build_pool(
        self,
        *,
        user_input: Dict[str, float],
        universe_idx: Optional[np.ndarray] = None,
        pool_size: int = 10000,
        allow_explicit: bool = False,
        exclude_track_ids: Optional[set] = None,
        shuffle_within_top: bool = True,
        random_state: int = 42,
        dontcare: Optional[Dict[str, bool]] = None,
        weight_overrides: Optional[Dict[str, float]] = None,
        ranges: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        # NEW: Planner controls
        lock_tempo: bool = False,
    ) -> pd.DataFrame:
        exclude_track_ids = exclude_track_ids or set()
        pool_size = int(max(50, pool_size))

        idx = (
            np.asarray(universe_idx, dtype=np.int64)
            if universe_idx is not None
            else np.arange(len(self.df), dtype=np.int64)
        )
        idx = self._filter_explicit_indices(idx, allow_explicit=allow_explicit)

        ranges = ranges or {}
        target_ranges = self._normalize_ranges(ranges)

        expanded_ranges_finite = None
        if target_ranges:
            idx, expanded_ranges_finite = self._filter_with_controlled_fallback_no_genre(
                idx=idx,
                target_ranges=target_ranges,
                k=pool_size,
                lock_tempo=lock_tempo,
            )

        user_input = dict(user_input or {})
        for f, (mn, mx) in target_ranges.items():
            if mn is not None and mx is not None:
                user_input[f] = float(0.5 * (mn + mx))

        u_scaled = self._user_vector_scaled(user_input)

        w = self.w.copy()
        if weight_overrides:
            for f, val in weight_overrides.items():
                if f in self._feat_idx:
                    w[self._feat_idx[f]] = float(val)
        if dontcare:
            for f, flag in dontcare.items():
                if flag and f in self._feat_idx:
                    w[self._feat_idx[f]] = 0.0

        d = self._weighted_distance(self.X[idx], u_scaled, w)

        if shuffle_within_top:
            rng = np.random.default_rng(random_state)
            d = d + rng.normal(loc=0.0, scale=1e-4, size=d.shape).astype(np.float32)

        top_local = self._topk_indices(d, min(pool_size, len(d)))
        top_global = idx[top_local]

        if exclude_track_ids:
            tid_col = "track_id" if "track_id" in self.df.columns else ("id" if "id" in self.df.columns else None)
            if tid_col:
                tids = self.df.iloc[top_global][tid_col].astype(str).str.strip().to_numpy()
                keep = ~np.isin(tids, list(exclude_track_ids))
                top_global = top_global[keep]

        cols = [c for c in ["track_id", "track_name", "artists", "album_name", "popularity", "genres_list", "genres_str"] if c in self.df.columns]
        out = self.df.iloc[top_global][cols].copy()
        if "artists" in out.columns:
            out["artists"] = out["artists"].apply(normalize_artists_field)
        out["track_genre"] = out.apply(two_genres_from_row, axis=1)

        # --- BPM (integer) ---
        if "tempo" in self.df.columns:
            sel = np.asarray(top_global, dtype=np.int64)
            out["bpm"] = (
                pd.to_numeric(self.df.iloc[sel]["tempo"], errors="coerce")
                .round()
                .astype("Int64")
                .to_numpy()
            )

        if target_ranges and expanded_ranges_finite is not None and len(top_global) > 0:
            match_range = self._range_score_playlist_100(list(top_global), target_ranges, expanded_ranges_finite)
            d_sel = d[top_local][: len(match_range)]
            dmin = float(np.min(d_sel)) if len(d_sel) else 0.0
            dmax = float(np.max(d_sel)) if len(d_sel) else 1.0
            denom = (dmax - dmin) if (dmax - dmin) > 1e-9 else 1.0
            match_dist = 100.0 * (1.0 - (d_sel - dmin) / denom)
            out["match"] = np.round(0.75 * match_range + 0.25 * match_dist, 2)
        else:
            out["match"] = 0.0

        if "popularity" in out.columns:
            out["popularity"] = pd.to_numeric(out["popularity"], errors="coerce").fillna(0)
            out = out.sort_values(["match", "popularity"], ascending=[False, False])
        else:
            out = out.sort_values("match", ascending=False)

        return out.reset_index(drop=True)


# -------------------------------------------------------------------
# Legacy “stable API” (kept as-is; not used by PlaylistRecommender)
# -------------------------------------------------------------------
def _coerce_range_for_column(df: pd.DataFrame, col: str, v: float) -> float:
    """
    Se la colonna è in [0,1] ma il preset usa 0..100, convertiamo.
    Se la colonna è in scala 'normale', lasciamo stare.
    """
    if col not in df.columns:
        return v
    s = df[col]
    try:
        mx = float(np.nanmax(s.values))
    except Exception:
        return v

    # tipico: danceability/energy/valence ecc sono 0..1
    if mx <= 1.5 and v > 1.5:
        return v / 100.0
    return v


def _apply_minmax(df: pd.DataFrame, col: str, vmin: Optional[float], vmax: Optional[float]) -> pd.DataFrame:
    if col not in df.columns:
        return df
    if vmin is not None:
        df = df[df[col] >= vmin]
    if vmax is not None:
        df = df[df[col] <= vmax]
    return df


def recommend_tracks(store, params: Dict[str, Any], k: int = 50, seed: Optional[int] = 7) -> List[Dict[str, Any]]:
    """
    API "stabile" per Planner (legacy):
    - store: DataStore con df_clean (o df)
    - params: dict tipo PRESETS[preset] con chiavi *_min / *_max (tempo_min, energy_min, ...)
    - k: numero brani
    Ritorna lista di dict con almeno track_name/artists/tempo/track_genre/track_id se disponibili.
    """
    df = None
    if hasattr(store, "df_clean"):
        df = store.df_clean
    elif hasattr(store, "df"):
        df = store.df
    else:
        raise RuntimeError("DataStore non espone df_clean/df")

    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    x = df.copy()

    key2col = {
        "tempo": "tempo",
        "energy": "energy",
        "danceability": "danceability",
        "valence": "valence",
        "instrumentalness": "instrumentalness",
        "acousticness": "acousticness",
        "liveness": "liveness",
        "speechiness": "speechiness",
        "loudness": "loudness",
        "popularity": "popularity",
    }

    for key, col in key2col.items():
        vmin = params.get(f"{key}_min", None)
        vmax = params.get(f"{key}_max", None)

        if vmin is not None:
            vmin = _coerce_range_for_column(x, col, float(vmin))
        if vmax is not None:
            vmax = _coerce_range_for_column(x, col, float(vmax))

        x = _apply_minmax(x, col, vmin, vmax)

    if x.empty:
        x = df.copy()

    rng = np.random.default_rng(seed)
    if "popularity" in x.columns:
        noise = rng.normal(0, 1, size=len(x))
        x = x.assign(_score=x["popularity"].fillna(0).astype(float) + noise)
        x = x.sort_values("_score", ascending=False)
    else:
        x = x.sample(frac=1.0, random_state=seed)

    out_df = x.head(int(k)).copy()

    def pick(row, *names, default=""):
        for n in names:
            if n in row and pd.notna(row[n]):
                return row[n]
        return default

    out: List[Dict[str, Any]] = []
    for _, row in out_df.iterrows():
        out.append({
            "track_id": pick(row, "track_id", "id", default=None),
            "track_name": pick(row, "track_name", "title", default=""),
            "artists": pick(row, "artists", "artist", default=""),
            "tempo": float(pick(row, "tempo", default=np.nan)) if "tempo" in out_df.columns else None,
            "track_genre": pick(row, "track_genre", "genre", default=""),
            "popularity": float(pick(row, "popularity", default=np.nan)) if "popularity" in out_df.columns else None,
            "duration_ms": float(pick(row, "duration_ms", default=np.nan)) if "duration_ms" in out_df.columns else None,
        })
    return out


def build_playlist(store, params: Dict[str, Any], k: int = 50, seed: Optional[int] = 7):
    return recommend_tracks(store, params=params, k=k, seed=seed)