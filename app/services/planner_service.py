# app/services/planner_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .data_store import DataStore
from .presets import PRESETS
from .recommender import PlaylistRecommender, RecommenderConfig


def _coerce_preset_name(preset_payload: Any) -> str:
    """
    The frontend may send:
      - "Warm-up Daytime"
      - {"name": "Warm-up Daytime"}
      - {"preset": "Warm-up Daytime"} (if someone nested it)
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
    If a value is missing, we simply don't set it (DataStore fills with mean in recommender).
    """
    if preset_name not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")

    p = PRESETS[preset_name]

    def mid(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return (float(a) + float(b)) / 2.0

    user_input: Dict[str, float] = {}

    # Map the preset keys you use -> audio features expected by recommender/DataStore.
    # These names must match DataStore feature names (e.g., tempo, energy, danceability, valence...)
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

    # Optional: loudness range (often negative values). If you have it in presets, keep it.
    if "loudness_min" in p and "loudness_max" in p:
        v = mid(p.get("loudness_min"), p.get("loudness_max"))
        if v is not None:
            user_input["loudness"] = float(v)

    # Optional: mode/key/time_signature if preset uses them as fixed values
    # (If you don’t use them in presets, this stays empty.)
    for fixed in ("mode", "key", "time_signature"):
        if fixed in p and p[fixed] is not None:
            user_input[fixed] = float(p[fixed])

    return user_input


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
                "popularity": float(row.get("popularity", 0)) if row.get("popularity") is not None else 0,
                "match": float(row.get("match", 0)) if row.get("match") is not None else 0,
            }
        )
    return out


@dataclass
class PlannerService:
    store: DataStore
    recommender: PlaylistRecommender

    @classmethod
    def from_store(cls, store: DataStore) -> "PlannerService":
        # You can tune these defaults. The critical part: k defaults to 50.
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
        genre: str = "",
        shuffle_within_top: bool = True,
        random_state: int = 42,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Main entry point for the planner route.
        Returns: (preset_name, tracks_list)
        """
        preset_name = _coerce_preset_name(preset_payload)
        if not preset_name:
            raise ValueError("Missing preset name")

        user_input = _preset_to_user_input(preset_name)

        df = self.recommender.recommend(
            user_input=user_input,
            genre=genre,
            k=int(k),
            max_per_artist=int(max_per_artist),
            exclude_track_ids=set(exclude_track_ids or set()),
            shuffle_within_top=bool(shuffle_within_top),
            random_state=int(random_state),
        )

        tracks = _df_to_tracks(df)
        return preset_name, tracks