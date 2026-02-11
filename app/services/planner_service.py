# app/services/planner_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np


from .data_store import DataStore
from .presets import PRESETS
from .recommender import PlaylistRecommender, RecommenderConfig


# ----------------------------
# Helpers
# ----------------------------
def _coerce_preset_name(preset_payload: Any) -> str:
    """
    The frontend may send:
      - "Warm-up Daytime"
      - {"name": "Warm-up Daytime"}
      - {"preset": "Warm-up Daytime"}
    Normalize to a preset name string.
    """
    if preset_payload is None:
        return ""
    if isinstance(preset_payload, str):
        return preset_payload.strip()

    if isinstance(preset_payload, dict):
        for key in ("name", "preset", "value", "label"):
            if key in preset_payload and isinstance(preset_payload[key], str):
                return preset_payload[key].strip()

    return str(preset_payload).strip()


def _preset_to_user_input(preset_name: str) -> Dict[str, float]:
    """
    Convert a preset definition into a user_input vector for the recommender.

    We use midpoint for ranged presets:
      tempo_min/max -> tempo = avg
      energy_min/max -> energy = avg
      ...

    If a value is missing, we don't set it (DataStore fills with mean in recommender).
    """
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")

    p = PRESETS[preset_name]

    def mid(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return (float(a) + float(b)) / 2.0

    user_input: Dict[str, float] = {}

    mapping_mid = {
        "tempo": ("tempo_min", "tempo_max"),
        "energy": ("energy_min", "energy_max"),
        "danceability": ("danceability_min", "danceability_max"),
        "valence": ("valence_min", "valence_max"),
        "acousticness": ("acousticness_min", "acousticness_max"),
        "instrumentalness": ("instrumentalness_min", "instrumentalness_max"),
        "liveness": ("liveness_min", "liveness_max"),
        "speechiness": ("speechiness_min", "speechiness_max"),
    }

    for feat, (kmin, kmax) in mapping_mid.items():
        v = mid(p.get(kmin), p.get(kmax))
        if v is not None:
            user_input[feat] = float(v)

    # loudness midpoint (negative range)
    if "loudness_min" in p and "loudness_max" in p:
        v = mid(p.get("loudness_min"), p.get("loudness_max"))
        if v is not None:
            user_input["loudness"] = float(v)

    # fixed categorical-ish features if you use them in presets
    for fixed in ("mode", "key", "time_signature"):
        if fixed in p and p[fixed] is not None:
            user_input[fixed] = float(p[fixed])

    return user_input


def _preset_to_ranges(preset_name: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Convert preset *_min/*_max into ranges dict consumed by recommender.

    Example:
      tempo_min/max -> ranges["tempo"] = (min, max)
    """
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")
    p = PRESETS[preset_name]

    mapping = {
        "tempo": ("tempo_min", "tempo_max"),
        "energy": ("energy_min", "energy_max"),
        "danceability": ("danceability_min", "danceability_max"),
        "valence": ("valence_min", "valence_max"),
        "acousticness": ("acousticness_min", "acousticness_max"),
        "instrumentalness": ("instrumentalness_min", "instrumentalness_max"),
        "liveness": ("liveness_min", "liveness_max"),
        "speechiness": ("speechiness_min", "speechiness_max"),
        "loudness": ("loudness_min", "loudness_max"),
    }

    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for feat, (kmin, kmax) in mapping.items():
        a = p.get(kmin, None)
        b = p.get(kmax, None)
        if a is None and b is None:
            continue
        ranges[feat] = (a, b)

    return ranges



def _split_ranges_tempo_vs_soft(
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]]
) -> Tuple[
    Dict[str, Tuple[Optional[float], Optional[float]]],
    Dict[str, Tuple[Optional[float], Optional[float]]],
]:
    """
    Split preset ranges into:
      - tempo_ranges: {"tempo": (min,max)}  (HARD, never relaxed)
      - soft_ranges: everything else (can be relaxed to fill k)
    """
    tempo_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    soft_ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    for f, (mn, mx) in (ranges or {}).items():
        if f == "tempo":
            tempo_ranges["tempo"] = (mn, mx)
        else:
            soft_ranges[f] = (mn, mx)

    return tempo_ranges, soft_ranges




def _list_from_preset(p: Dict[str, Any], key: str) -> List[str]:
    """
    Extract list-like fields from preset (safe normalization).
    """
    raw = p.get(key, [])
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = list(raw)  # last resort

    out: List[str] = []
    for x in raw:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _df_to_tracks(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert recommender output DF into a JSON-friendly list for frontend.
    """
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "track_id": str(row.get("track_id", "")).strip(),
                "track_name": str(row.get("track_name", "")).strip(),
                "artists": str(row.get("artists", "")).strip(),
                "album_name": str(row.get("album_name", "")).strip(),
                "track_genre": str(row.get("track_genre", "")).strip(),
                "popularity": float(row.get("popularity", 0) or 0),
                "match": float(row.get("match", 0) or 0),
            }
        )
    return out


# ----------------------------
# Service
# ----------------------------
@dataclass
class PlannerService:
    store: DataStore
    recommender: PlaylistRecommender

    @classmethod
    def from_store(cls, store: DataStore) -> "PlannerService":
        # Planner defaults: 50 tracks, cap per artist
        cfg = RecommenderConfig(k=50, max_per_artist=2)
        rec = PlaylistRecommender(store, cfg)
        return cls(store=store, recommender=rec)

    def generate_playlist_for_preset(
        self,
        preset_payload: Any,
        *,
        k: int = 50,
        max_per_artist: int = 2,
        exclude_track_ids: Optional[Set[str]] = None,
        shuffle_within_top: bool = True,
        random_state: int = 42,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main entry point for Planner.
        Returns: (preset_name, tracks_list)

        Planner philosophy:
          - strict_semantics=True  (no genre drift)
          - lock_tempo=True        (tempo/BPM is HARD)
          - preset can define include/exclude artists/genres + weights
          - fallback can be enabled to fill remaining slots without leaving allowed genre universe
        """
        preset_name = _coerce_preset_name(preset_payload)
        if not preset_name:
            raise ValueError("Missing preset name")
        if preset_name not in PRESETS:
            raise KeyError(f"Unknown preset: {preset_name}")

        p = PRESETS[preset_name]

        user_input = _preset_to_user_input(preset_name)
        ranges = _preset_to_ranges(preset_name)

        include_artists = _list_from_preset(p, "include_artists")
        include_genres = _list_from_preset(p, "include_genres")
        exclude_artists = _list_from_preset(p, "exclude_artists")
        exclude_genres = _list_from_preset(p, "exclude_genres")

        # Optional weights for quotas (must match list lengths if provided)
        artist_weights = p.get("artist_weights", None)
        genre_weights = p.get("genre_weights", None)

        # Optional behavior toggles
        allow_explicit = bool(p.get("allow_explicit", False))
        fallback = bool(p.get("fallback", True))

        # ----------------------------
        # STRADA B (Planner):
        # 1) tempo is HARD (pool builder)
        # 2) other ranges are SOFT and can be relaxed (pool builder only)
        # 3) include_artists / include_genres are applied ON THE POOL:
        #    - include_mode: preset can decide "must" or "prefer"
        # 4) exclusions always HARD
        # ----------------------------

        tempo_ranges, soft_ranges = _split_ranges_tempo_vs_soft(ranges)

        # preset toggles
        include_mode = str(p.get("include_mode", "prefer")).strip().lower()  # "must" | "prefer"
        if include_mode not in {"must", "prefer"}:
            include_mode = "prefer"

        # prefer_strength lets you tune how much "preferred genres/artists" are boosted
        prefer_strength = p.get("prefer_strength", 0.18)
        try:
            prefer_strength = float(prefer_strength)
        except Exception:
            prefer_strength = 0.18

        # ---- Step A: build POOL (tempo hard, soft ranges applied but relaxable) ----
        # We do this in tiers: start with soft ranges, then relax soft ranges if pool too small.
        # NEVER relax tempo.
        # Pool size should be > k (so include filters don't collapse results).
        pool_size = int(max(2000, 30 * int(k)))

        base_universe = self.recommender.build_universe_indices(
            include_artists=None,   # IMPORTANT: do NOT restrict universe here (we'll filter on pool)
            include_genres=None,
            exclude_artists=exclude_artists,
            exclude_genres=exclude_genres,
        )

        # Try strict-soft first, then relax by removing soft ranges gradually.
        # Order of relaxation: remove weakest first (you can tune this list).
        relax_order = ["speechiness", "liveness", "acousticness", "instrumentalness", "valence", "danceability", "energy", "loudness"]

        def make_soft_ranges(level: int) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
            """
            level=0 => keep all soft ranges
            level increases => drop more soft ranges
            """
            if not soft_ranges:
                return {}
            keep = dict(soft_ranges)
            for i in range(min(level, len(relax_order))):
                keep.pop(relax_order[i], None)
            return keep

        pool_df = None

        for level in range(0, len(relax_order) + 1):
            # tempo always present if defined in preset
            rr = dict(tempo_ranges)
            rr.update(make_soft_ranges(level))

            pool_df = self.recommender.build_pool(
                user_input=user_input,
                universe_idx=base_universe,
                pool_size=pool_size,
                allow_explicit=allow_explicit,
                exclude_track_ids=set(exclude_track_ids or set()),
                shuffle_within_top=True,
                random_state=int(random_state),
                ranges=rr,
                lock_tempo=True,   # HARD tempo at pool stage
            )

            # We want enough candidates before include filters.
            if pool_df is not None and len(pool_df) >= int(max(pool_size * 0.60, 1000)):
                break

        if pool_df is None or len(pool_df) == 0:
            return preset_name, []

        pool_idx = pool_df.index.to_numpy(dtype="int64")
        # IMPORTANT: pool_df is a slice of df rows, but we need original indices.
        # build_pool() returns df.iloc[top_global] without preserving original index by default.
        # So: we must carry original indices out of build_pool().
        #
        # Quick fix: ensure build_pool() returns the ORIGINAL indices in a column.
        # For now, fallback to mapping via track_id (slower but safe).
        #
        # We'll patch build_pool() in recommender later; for now do track_id mapping.
        base_df = self.recommender.df
        if "track_id" in pool_df.columns and "track_id" in base_df.columns:
            tids = pool_df["track_id"].astype(str).str.strip().tolist()
            tid2idx = pd.Series(base_df.index, index=base_df["track_id"].astype(str).str.strip()).to_dict()
            pool_idx = np.array([tid2idx.get(t) for t in tids if t in tid2idx], dtype=np.int64)
            pool_idx = np.unique(pool_idx)

        # ---- Step B: apply include/exclude on POOL and recommend ----
        df_out = self.recommender.recommend_from_pool(
            user_input=user_input,
            pool_idx=pool_idx,
            k=int(k),
            max_per_artist=int(max_per_artist),
            exclude_track_ids=set(exclude_track_ids or set()),
            allow_explicit=allow_explicit,
            shuffle_within_top=bool(shuffle_within_top),
            random_state=int(random_state),
            include_artists=include_artists,
            include_genres=include_genres,
            include_mode=include_mode,
            prefer_strength=prefer_strength,
            exclude_artists=exclude_artists,
            exclude_genres=exclude_genres,
        )

        # If include_mode="must" collapses results, fallback to "prefer" (still respecting BPM + excludes)
        if include_mode == "must" and (df_out is None or len(df_out) < int(k)) and fallback:
            df_out = self.recommender.recommend_from_pool(
                user_input=user_input,
                pool_idx=pool_idx,
                k=int(k),
                max_per_artist=int(max_per_artist),
                exclude_track_ids=set(exclude_track_ids or set()),
                allow_explicit=allow_explicit,
                shuffle_within_top=bool(shuffle_within_top),
                random_state=int(random_state),
                include_artists=include_artists,
                include_genres=include_genres,
                include_mode="prefer",
                prefer_strength=prefer_strength,
                exclude_artists=exclude_artists,
                exclude_genres=exclude_genres,
            )

        tracks = _df_to_tracks(df_out if df_out is not None else pd.DataFrame())
        return preset_name, tracks

