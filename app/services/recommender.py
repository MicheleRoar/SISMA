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

    IMPORTANT: for Planner:
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
        # HARD exclusions (robust, token-safe)
        # -----------------------------
        idx = self._apply_exclusions(
            idx,
            exclude_artists=exclude_artists,
            exclude_genres=exclude_genres,
        )

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
    # NEW: pool -> include/exclude -> recommend
    # -----------------------------

    def recommend_from_pool(
        self,
        *,
        user_input: Dict[str, float],
        pool_idx: np.ndarray,
        k: int = 50,
        max_per_artist: int = 2,
        exclude_track_ids: Optional[set] = None,
        allow_explicit: bool = False,
        shuffle_within_top: bool = True,
        random_state: int = 42,
        weight_overrides: Optional[Dict[str, float]] = None,
        dontcare: Optional[Dict[str, bool]] = None,
        include_artists: Optional[List[str]] = None,
        include_genres: Optional[List[str]] = None,
        include_mode: str = "prefer",   # "must" | "prefer"
        prefer_strength: float = 0.18,  # (non usato nel two-pass, tenuto per compatibilità)
        exclude_artists: Optional[List[str]] = None,
        exclude_genres: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        exclude_track_ids = exclude_track_ids or set()
        include_artists = [str(x).strip() for x in (include_artists or []) if str(x).strip()]
        include_genres = [str(x).strip() for x in (include_genres or []) if str(x).strip()]
        exclude_artists = [str(x).strip() for x in (exclude_artists or []) if str(x).strip()]
        exclude_genres = [str(x).strip() for x in (exclude_genres or []) if str(x).strip()]

        idx = np.asarray(pool_idx, dtype=np.int64)
        if idx.size == 0:
            return pd.DataFrame()

        # HARD: explicit + exclusions
        idx = self._filter_explicit_indices(idx, allow_explicit=allow_explicit)
        idx = self._apply_exclusions(idx, exclude_artists=exclude_artists, exclude_genres=exclude_genres)
        if idx.size == 0:
            return pd.DataFrame()

        # helper: union indices for included artists/genres (global indices)
        def _union_artist_idx(names: List[str]) -> np.ndarray:
            if not names:
                return np.array([], dtype=np.int64)
            parts = [self._indices_by_artist_contains(a) for a in names]
            return np.unique(np.concatenate(parts).astype(np.int64)) if parts else np.array([], dtype=np.int64)

        def _union_genre_idx(names: List[str]) -> np.ndarray:
            if not names:
                return np.array([], dtype=np.int64)
            parts = [self._indices_by_genre_contains(g) for g in names]
            return np.unique(np.concatenate(parts).astype(np.int64)) if parts else np.array([], dtype=np.int64)

        inc_mode = (include_mode or "prefer").strip().lower()
        if inc_mode not in {"must", "prefer"}:
            inc_mode = "prefer"

        inc_artist_idx = _union_artist_idx(include_artists)
        inc_genre_idx = _union_genre_idx(include_genres)

        # include_mode="must": HARD filter inside pool
        if inc_mode == "must":
            if include_artists:
                idx = np.intersect1d(idx, inc_artist_idx)
                if idx.size == 0:
                    return pd.DataFrame()
            if include_genres:
                idx = np.intersect1d(idx, inc_genre_idx)
                if idx.size == 0:
                    return pd.DataFrame()

        # build user vector + request weights
        user_input = dict(user_input or {})
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

        # --- prefer (two-pass) ---
        pref_mask = np.zeros(idx.size, dtype=bool)
        if include_artists:
            pref_mask |= np.isin(idx, inc_artist_idx)
        if include_genres:
            pref_mask |= np.isin(idx, inc_genre_idx)

        if inc_mode == "prefer" and (include_artists or include_genres):
            idx_pref = idx[pref_mask]
            idx_rest = idx[~pref_mask]
        else:
            idx_pref = np.array([], dtype=np.int64)
            idx_rest = idx

        def pick_from(local_idx: np.ndarray, need: int, already_picked: Optional[np.ndarray] = None) -> List[int]:
            if need <= 0 or local_idx.size == 0:
                return []
            if already_picked is not None and already_picked.size:
                local_idx = local_idx[~np.isin(local_idx, already_picked)]
                if local_idx.size == 0:
                    return []

            dloc = self._weighted_distance(self.X[local_idx], u_scaled, w_req)

            if shuffle_within_top:
                rng = np.random.default_rng(random_state)
                dloc = dloc + rng.normal(loc=0.0, scale=1e-4, size=dloc.shape).astype(np.float32)

            pool_size = min(len(local_idx), max(int(need) * self.config.pool_multiplier, self.config.min_pool))
            top_local = self._topk_indices(dloc, pool_size)
            top_global = local_idx[top_local]

            if shuffle_within_top:
                rng = np.random.default_rng(random_state)
                rng.shuffle(top_global)

            return self._apply_constraints(
                top_global,
                k=int(need),
                max_per_artist=int(max_per_artist),
                exclude_track_ids=set(exclude_track_ids),
                artist_caps=None,
            )

        selected: List[int] = []
        picked1 = pick_from(idx_pref, int(k))
        selected.extend(picked1)

        remaining = int(k) - len(selected)
        if remaining > 0:
            picked2 = pick_from(idx_rest, remaining, already_picked=np.asarray(selected, dtype=np.int64))
            selected.extend(picked2)

        if not selected:
            return pd.DataFrame()

        cols = [c for c in ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity", "genres_list", "genres_str"]
                if c in self.df.columns]
        out = self.df.iloc[np.asarray(selected, dtype=np.int64)][cols].copy()

        if "artists" in out.columns:
            out["artists"] = out["artists"].apply(normalize_artists_field)
        out["track_genre"] = out.apply(two_genres_from_row, axis=1)

        if "tempo" in self.df.columns:
            sel = np.asarray(selected, dtype=np.int64)
            out["bpm"] = (
                pd.to_numeric(self.df.iloc[sel]["tempo"], errors="coerce")
                .round()
                .astype("Int64")
                .to_numpy()
            )

        # ---- MATCH (distance-based, 0..100) ----
        sel = np.asarray(selected, dtype=np.int64)

        d_all = self._weighted_distance(self.X[idx], u_scaled, w_req)

        # mappa global_index -> posizione in idx
        idx_pos = {int(g): i for i, g in enumerate(idx.tolist())}
        sel_pos = np.array([idx_pos[int(g)] for g in sel if int(g) in idx_pos], dtype=np.int64)

        d_sel = d_all[sel_pos]

        p10, p90 = np.percentile(d_all, [10, 90])
        den = (p90 - p10) if (p90 - p10) > 1e-9 else 1.0
        match = 100.0 * (1.0 - np.clip((d_sel - p10) / den, 0.0, 1.0))
        out["match"] = np.round(match, 2).astype(float)

        return out.reset_index(drop=True)


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
            # Accept also non-audio hard filters like "year" (present in df but not in feature vector)
            if (f not in self._feat_idx) and (f not in self.df.columns):
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
            idx, expanded_ranges_finite = self._filter_with_controlled_feature_expansion(
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
        out["_row_idx"] = top_global 

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
