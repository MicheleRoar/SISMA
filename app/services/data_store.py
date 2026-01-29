# app/services/data_store.py
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .genre_builder import (
    build_artist_genre_lookup,
    enrich_tracks_with_genres,
    compute_ui_genres,
)

# Feature set (12)
AUDIO_FEATURES_12 = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


class DataStore:
    """
    Loads dataset once, cleans it, precomputes:
    - df_clean
    - X_raw, X_scaled
    - standardizer
    - ui_genres (top-k genres for dropdown)

    PATCHES (chirurgici, senza rimuovere funzioni):
      - genre matching token-safe (no substring): usa delimitatori |...|
      - cache genre -> indices per lookup veloce
      - normalizzazione coerente (lower + strip + rimozione accenti)
      - se genre non matcha nulla: ritorna array vuoto (no fallback al dataset intero)
      - aggiunta funzione NON-breaking: get_row_indices_by_genres(genres: List[str])
    """

    def __init__(
        self,
        tracks_csv_path: str = "data/archive/tracks.csv",
        artists_csv_path: str = "data/archive/artists.csv",
        feature_cols: Optional[List[str]] = None,
        cache_scaled: bool = True,
        # UI genre policy (as discussed)
        ui_genre_min_count: int = 50,
        ui_genre_top_k: int = 200,
    ):
        self.tracks_csv_path = tracks_csv_path
        self.artists_csv_path = artists_csv_path
        self.feature_cols = feature_cols or AUDIO_FEATURES_12
        self.cache_scaled = cache_scaled

        self.ui_genre_min_count = int(ui_genre_min_count)
        self.ui_genre_top_k = int(ui_genre_top_k)

        self.df_clean: Optional[pd.DataFrame] = None
        self.X_raw: Optional[np.ndarray] = None
        self.X_scaled: Optional[np.ndarray] = None
        self.standardizer: Optional[Standardizer] = None

        self._ui_genres: List[str] = []

        # --- NEW (internal caches, non breaking) ---
        self._genre_to_indices: Dict[str, np.ndarray] = {}
        self._has_genre_cache: bool = False

        self._load_and_prepare()

    # -----------------------
    # Internal helpers (NEW)
    # -----------------------
    @staticmethod
    def _norm_text(s: str) -> str:
        """
        Lower + strip + remove diacritics (accents).
        Usata per generi.
        """
        s = "" if s is None else str(s)
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        return s.lower().strip()

    @staticmethod
    def _split_or_syntax(s: str) -> List[str]:
        """
        Supporta una sintassi OR opzionale:
          "a||b||c"
        Senza cambiare la firma di get_row_indices_by_genre().
        """
        if not s:
            return []
        parts = [p.strip() for p in str(s).split("||")]
        return [p for p in parts if p]

    # -----------------------
    # Public API
    # -----------------------
    def get_df(self) -> pd.DataFrame:
        if self.df_clean is None:
            raise RuntimeError("DataStore not initialized.")
        return self.df_clean

    def get_X(self, scaled: bool = True) -> np.ndarray:
        if scaled:
            if self.X_scaled is None:
                raise RuntimeError("Scaled matrix not available.")
            return self.X_scaled
        if self.X_raw is None:
            raise RuntimeError("Raw matrix not available.")
        return self.X_raw

    def get_feature_names(self) -> List[str]:
        return list(self.feature_cols)

    def get_genres(self) -> List[str]:
        # per il dropdown
        return list(getattr(self, "ui_genres", []))

    def get_all_genres(self) -> List[str]:
        # per l’autocomplete
        return list(getattr(self, "all_genres", []))

    def get_row_indices_by_genre(self, genre: str) -> np.ndarray:
        """
        New dataset: genre match is token-safe, using either:
          1) cache genre->indices (fast path) on normalized tokens
          2) bounded string column (|g1|g2|...) for safety

        IMPORTANT PATCH:
          - niente substring matching: cerca "|genre|" (non "genre" dentro "k-pop")
          - se genre non matcha nulla -> ritorna array vuoto (nessun fallback globale)
          - supporta OR con "||" (es. "italian pop||classic italian pop")
        """
        df = self.get_df()
        if not genre:
            return np.arange(len(df), dtype=np.int64)

        # OR syntax support (non breaking)
        parts = self._split_or_syntax(genre)
        if len(parts) > 1:
            return self.get_row_indices_by_genres(parts)

        if "genres_list" not in df.columns and "genres_str" not in df.columns:
            # se non esistono colonne, comportamento legacy: tutto
            return np.arange(len(df), dtype=np.int64)

        g = self._norm_text(genre)
        if not g:
            return np.arange(len(df), dtype=np.int64)

        # 1) Fast path: cache genre -> indices
        if self._has_genre_cache and g in self._genre_to_indices:
            return self._genre_to_indices[g]

        # 2) Fallback: bounded string token match (|g|)
        # preferiamo genres_str_bounded se esiste, altrimenti proviamo a bounded-on-the-fly
        col = None
        if "genres_str_bounded" in df.columns:
            col = "genres_str_bounded"
        elif "genres_str" in df.columns:
            col = "genres_str"

        if col is None:
            return np.array([], dtype=np.int64)

        # token-safe: match |g|
        # Se col non è già bounded, proviamo a renderla bounded al volo
        if col == "genres_str_bounded":
            s = df[col].astype(str).map(self._norm_text)
            token = f"|{g}|"
            mask = s.str.contains(re.escape(token), regex=True)
        else:
            # genres_str è "g1|g2|g3": per match token-safe usiamo regex con boundaries di pipe
            # pattern: (^|\|)g(\||$)
            s = df[col].astype(str).map(self._norm_text)
            pattern = rf"(^|\|){re.escape(g)}(\||$)"
            mask = s.str.contains(pattern, regex=True)

        idx = np.flatnonzero(mask.to_numpy())
        # NO fallback globale se non troviamo niente
        return idx.astype(np.int64)

    def get_row_indices_by_genres(self, genres: List[str]) -> np.ndarray:
        """
        NEW (additiva, non rompe nulla):
        Ritorna l'unione degli indici per una lista di generi (OR).
        """
        df = self.get_df()
        if not genres:
            return np.arange(len(df), dtype=np.int64)

        parts = [self._norm_text(g) for g in genres if self._norm_text(g)]
        if not parts:
            return np.arange(len(df), dtype=np.int64)

        all_idx = []
        for g in parts:
            # usa cache se possibile
            if self._has_genre_cache and g in self._genre_to_indices:
                all_idx.append(self._genre_to_indices[g])
            else:
                # usa il metodo singolo (che è token-safe e ritorna vuoto se non matcha)
                all_idx.append(self.get_row_indices_by_genre(g))

        if not all_idx:
            return np.array([], dtype=np.int64)

        merged = np.unique(np.concatenate(all_idx).astype(np.int64))
        return merged.astype(np.int64)

    # -----------------------
    # Load / Clean / Prepare
    # -----------------------
    def _load_and_prepare(self) -> None:
        if not os.path.exists(self.tracks_csv_path):
            raise FileNotFoundError(f"tracks CSV not found: '{self.tracks_csv_path}'")
        if not os.path.exists(self.artists_csv_path):
            raise FileNotFoundError(f"artists CSV not found: '{self.artists_csv_path}'")

        tracks = pd.read_csv(self.tracks_csv_path)
        artists = pd.read_csv(self.artists_csv_path)

        # --- Artists UI list (for autocomplete) ---
        if "name" in artists.columns:
            artists["name"] = artists["name"].astype(str).str.strip()
        else:
            artists["name"] = ""

        if "popularity" in artists.columns:
            artists["popularity"] = pd.to_numeric(artists["popularity"], errors="coerce").fillna(0)
        else:
            artists["popularity"] = 0

        # soglia
        MIN_ARTIST_POP = 30

        ui_artists = (
            artists.loc[artists["popularity"] >= MIN_ARTIST_POP, "name"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        # dedup + sort
        self.ui_artists = sorted(ui_artists[ui_artists != ""].unique(), key=lambda s: s.lower())

        # --- Normalize column names to match the rest of the app ---
        # tracks.csv example has: id,name,artists,id_artists,...
        rename_map = {
            "id": "track_id",
            "name": "track_name",
        }
        for k, v in rename_map.items():
            if k in tracks.columns and v not in tracks.columns:
                tracks = tracks.rename(columns={k: v})

        # Ensure mandatory columns exist
        if "track_id" not in tracks.columns:
            raise ValueError("tracks.csv missing 'track_id' (or 'id')")
        if "track_name" not in tracks.columns:
            raise ValueError("tracks.csv missing 'track_name' (or 'name')")

        # --- Enrich with genres from artists.csv ---
        artist_to_genres = build_artist_genre_lookup(
            artists_df=artists,
            artist_id_col="id",
            genres_col="genres",
        )
        tracks = enrich_tracks_with_genres(
            tracks_df=tracks,
            artist_to_genres=artist_to_genres,
            track_artist_ids_col="id_artists",
            add_genres_list_col="genres_list",
            add_genres_str_col="genres_str",
            sep="|",
        )

        # --- NEW: create a bounded version for token-safe match ---
        # Keep original 'genres_str' for compatibility/debug, add 'genres_str_bounded' (non breaking)
        if "genres_str" in tracks.columns:
            # normalize to string, avoid NaN, avoid leading/trailing pipes duplication
            gs = tracks["genres_str"].fillna("").astype(str)
            # if empty -> "||"? no, keep empty
            tracks["genres_str_bounded"] = np.where(gs.str.len() > 0, "|" + gs + "|", "")
        else:
            tracks["genres_str_bounded"] = ""

        # --- NEW: build cache genre -> indices using genres_list (fast path) ---
        self._genre_to_indices = {}
        self._has_genre_cache = False
        if "genres_list" in tracks.columns:
            # build using normalized tokens
            # iterate rows; for speed and memory, store lists then convert to arrays
            tmp: Dict[str, List[int]] = {}
            # ensure genres_list is list-like; if it's not, treat as empty
            for i, gs in enumerate(tracks["genres_list"]):
                if not isinstance(gs, list) or not gs:
                    continue
                for g in gs:
                    gn = self._norm_text(g)
                    if not gn:
                        continue
                    tmp.setdefault(gn, []).append(int(i))
            # finalize to np arrays
            self._genre_to_indices = {g: np.asarray(idxs, dtype=np.int64) for g, idxs in tmp.items()}
            self._has_genre_cache = True

        # UI genres list (top200, min50)
        ui_genres, counter = compute_ui_genres(
            tracks,
            genres_list_col="genres_list",
            min_count=self.ui_genre_min_count,
            top_k=self.ui_genre_top_k,
        )

        self.ui_genres = sorted(ui_genres, key=lambda s: str(s).lower())
        self.all_genres = sorted(list(counter.keys()), key=lambda s: str(s).lower())

        df = tracks

        # explicit can be 0/1 already; still normalize
        if "explicit" in df.columns:
            df["explicit"] = pd.to_numeric(df["explicit"], errors="coerce").fillna(0).astype(int)

        for col in ["popularity", "duration_ms"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Coerce feature columns to numeric (create if missing)
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan

        # Basic cleaning
        df = df.dropna(subset=["track_name"])
        if "artists" in df.columns:
            df = df.dropna(subset=["artists"])

        # Fill missing features
        for col in self.feature_cols:
            if col == "key":
                df[col] = df[col].fillna(-1)
            elif col == "mode":
                df[col] = df[col].fillna(0)
            elif col == "time_signature":
                df[col] = df[col].fillna(4)
            else:
                med = df[col].median(skipna=True)
                if pd.isna(med):
                    med = 0.0
                df[col] = df[col].fillna(float(med))

        # Clip ranges
        bounded_01 = [
            "danceability", "energy", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence",
        ]
        for col in bounded_01:
            if col in df.columns:
                df[col] = df[col].clip(0.0, 1.0)

        if "tempo" in df.columns:
            df["tempo"] = df["tempo"].clip(20.0, 250.0)

        if "loudness" in df.columns:
            df["loudness"] = df["loudness"].clip(-60.0, 5.0)

        # Dedup by track_id if possible
        if "track_id" in df.columns:
            if "popularity" in df.columns:
                df = df.sort_values("popularity", ascending=False)
            df = df.drop_duplicates(subset=["track_id"], keep="first")

        # IMPORTANT: after dedup/reset, the cache indices built above refer to the pre-dedup row order.
        df = df.reset_index(drop=True)

        # Rebuild genre cache on the finalized df (chirurgico, ma necessario)
        self._genre_to_indices = {}
        self._has_genre_cache = False
        if "genres_list" in df.columns:
            tmp: Dict[str, List[int]] = {}
            for i, gs in enumerate(df["genres_list"]):
                if not isinstance(gs, list) or not gs:
                    continue
                for g in gs:
                    gn = self._norm_text(g)
                    if not gn:
                        continue
                    tmp.setdefault(gn, []).append(int(i))
            self._genre_to_indices = {g: np.asarray(idxs, dtype=np.int64) for g, idxs in tmp.items()}
            self._has_genre_cache = True

        # Build matrices
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        self.X_raw = X

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)

        self.standardizer = Standardizer(mean=mean, std=std)

        if self.cache_scaled:
            self.X_scaled = self.standardizer.transform(X).astype(np.float32)

        self.df_clean = df