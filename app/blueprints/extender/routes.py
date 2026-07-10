# app/blueprints/extender/routes.py
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from flask import Blueprint, jsonify, render_template, request

from app.services.genre_builder import build_artist_genre_lookup, enrich_tracks_with_genres
from scripts.deduplicate_tracks_and_artists import deduplicate_artists_by_id

bp = Blueprint("extender", __name__, url_prefix="/extender")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive"
TRACKS_ARCHIVE_PATH = ARCHIVE_DIR / "tracks.csv"
ARTISTS_ARCHIVE_PATH = ARCHIVE_DIR / "artists.csv"
BACKUPS_DIR = ARCHIVE_DIR / "_backups"

SPOTIFY_API = "https://api.spotify.com/v1"

FEATURE_COLS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]
TRACKS_OUT_COLUMNS = [
    "id", "name", "popularity", "duration_ms", "explicit", "artists", "id_artists",
    "release_date",
] + FEATURE_COLS
ARTISTS_OUT_COLUMNS = ["id", "followers", "genres", "name", "popularity"]


# ----------------------------
# Spotify HTTP helpers
# ----------------------------
def _spotify_get_json(token: str, url: str, params: Optional[dict] = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, params=params, timeout=20)

    if r.status_code == 429:
        wait = int(r.headers.get("Retry-After", 5))
        time.sleep(wait)
        return _spotify_get_json(token, url, params)

    r.raise_for_status()
    return r.json()


def _fetch_me(token: str) -> dict:
    return _spotify_get_json(token, f"{SPOTIFY_API}/me")


def _fetch_playlists(token: str, user_id: str, include_all: bool = False) -> List[dict]:
    """
    By default only returns playlists owned by `user_id`. If `include_all` is
    True, also includes playlists the user merely follows/collaborates on.
    """
    playlists: List[dict] = []
    url = f"{SPOTIFY_API}/me/playlists"
    params = {"limit": 50}

    while url:
        data = _spotify_get_json(token, url, params)
        for p in data.get("items", []) or []:
            owner = p.get("owner") or {}
            if include_all or owner.get("id") == user_id:
                playlists.append(p)
        url = data.get("next")
        params = None

    return playlists



def _fetch_playlist_tracks(token: str, playlist_id: str) -> List[dict]:
    tracks: List[dict] = []
    url = f"{SPOTIFY_API}/playlists/{playlist_id}/tracks"
    params = {"limit": 100}

    while url:
        data = _spotify_get_json(token, url, params)
        for item in data.get("items", []) or []:
            t = item.get("track")
            if not t or t.get("type") != "track" or not t.get("id"):
                continue
            tracks.append(t)
        url = data.get("next")
        params = None

    return tracks


def _fetch_audio_features(token: str, track_ids: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    rows: List[dict] = []
    warning: Optional[str] = None

    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i + 100]
        try:
            data = _spotify_get_json(token, f"{SPOTIFY_API}/audio-features", {"ids": ",".join(batch)})
        except requests.HTTPError as e:
            warning = (
                f"audio-features request failed ({e}) - this endpoint is restricted for most "
                "apps; new tracks will be missing audio features (danceability, energy, etc.)."
            )
            break
        rows.extend([x for x in (data.get("audio_features") or []) if x])

    cols = ["id"] + FEATURE_COLS
    df = pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)
    return df, warning


def _fetch_artists(token: str, artist_ids: List[str]) -> List[dict]:
    out: List[dict] = []

    for i in range(0, len(artist_ids), 50):
        batch = artist_ids[i:i + 50]
        data = _spotify_get_json(token, f"{SPOTIFY_API}/artists", {"ids": ",".join(batch)})
        for a in data.get("artists", []) or []:
            if a:
                out.append(a)

    return out


# ----------------------------
# Transform helpers (pure, no network - testable in isolation)
# ----------------------------
def _build_tracks_out(raw_tracks: List[dict], features_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in raw_tracks:
        rows.append({
            "id": t["id"],
            "name": t.get("name", ""),
            "popularity": t.get("popularity", 0),
            "duration_ms": t.get("duration_ms", 0),
            "explicit": int(bool(t.get("explicit"))),
            "artists": [a.get("name", "") for a in (t.get("artists") or [])],
            "id_artists": [a.get("id") for a in (t.get("artists") or []) if a.get("id")],
            "release_date": (t.get("album") or {}).get("release_date", ""),
        })

    if not rows:
        return pd.DataFrame(columns=TRACKS_OUT_COLUMNS)

    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)
    out = df.merge(features_df, on="id", how="left")

    for c in TRACKS_OUT_COLUMNS:
        if c not in out.columns:
            out[c] = None

    return out[TRACKS_OUT_COLUMNS]


def _build_artists_out(raw_artists: List[dict]) -> pd.DataFrame:
    rows = []
    for a in raw_artists:
        rows.append({
            "id": a["id"],
            "followers": float((a.get("followers") or {}).get("total") or 0),
            "genres": a.get("genres") or [],
            "name": a.get("name", ""),
            "popularity": a.get("popularity", 0),
        })

    if not rows:
        return pd.DataFrame(columns=ARTISTS_OUT_COLUMNS)

    return pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)[ARTISTS_OUT_COLUMNS]


def _load_existing_track_ids(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["id"], dtype=str)
    return set(df["id"].dropna())


def _load_existing_artists_min(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id", "genres"])
    return pd.read_csv(path, usecols=["id", "genres"], dtype=str)


def _flag_invisible_tracks(
    new_tracks: pd.DataFrame,
    fresh_artists: pd.DataFrame,
    existing_artists_min: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flags tracks whose artists resolve to zero genres - DataStore silently drops
    these on load (see app/services/data_store.py), so surface it instead of
    letting it happen invisibly after apply.
    """
    if new_tracks.empty:
        out = new_tracks.copy()
        out["invisible"] = pd.Series(dtype=bool)
        return out

    combined = pd.concat(
        [existing_artists_min[["id", "genres"]], fresh_artists[["id", "genres"]]],
        ignore_index=True,
    )
    artist_to_genres = build_artist_genre_lookup(combined, artist_id_col="id", genres_col="genres")
    enriched = enrich_tracks_with_genres(new_tracks, artist_to_genres, track_artist_ids_col="id_artists")

    out = new_tracks.copy()
    out["invisible"] = enriched["genres_list"].apply(lambda gs: len(gs) == 0)
    return out


def _df_to_json_records(df: pd.DataFrame) -> List[dict]:
    """pandas' own to_json handles numpy dtypes + NaN-to-null correctly, unlike a
    plain .to_dict(orient='records') passed straight to flask.jsonify."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


# ----------------------------
# Apply (archive merge) helpers
# ----------------------------
def _backup_archive() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUPS_DIR / ts
    dest.mkdir(parents=True, exist_ok=True)

    if TRACKS_ARCHIVE_PATH.exists():
        shutil.copy2(TRACKS_ARCHIVE_PATH, dest / "tracks.csv")
    if ARTISTS_ARCHIVE_PATH.exists():
        shutil.copy2(ARTISTS_ARCHIVE_PATH, dest / "artists.csv")

    return dest


def _apply_new_tracks(selected_tracks: List[dict]) -> Tuple[int, int]:
    existing = (
        pd.read_csv(TRACKS_ARCHIVE_PATH, dtype=str)
        if TRACKS_ARCHIVE_PATH.exists()
        else pd.DataFrame(columns=TRACKS_OUT_COLUMNS)
    )
    before = len(existing)

    if not selected_tracks:
        return before, before

    new_df = pd.DataFrame(selected_tracks).drop(columns=["invisible"], errors="ignore")

    for col in ("artists", "id_artists"):
        if col in new_df.columns:
            new_df[col] = new_df[col].apply(lambda v: str(v) if isinstance(v, list) else v)

    for c in TRACKS_OUT_COLUMNS:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[TRACKS_OUT_COLUMNS]

    merged = pd.concat([existing, new_df], ignore_index=True)
    sort_key = pd.to_numeric(merged["popularity"], errors="coerce").fillna(0)
    merged = merged.assign(_pop_sort=sort_key).sort_values("_pop_sort", ascending=False)
    merged = merged.drop(columns=["_pop_sort"]).drop_duplicates(subset=["id"], keep="first")
    merged = merged.reset_index(drop=True)

    merged.to_csv(TRACKS_ARCHIVE_PATH, index=False)
    return before, len(merged)


def _apply_new_artists(selected_artists: List[dict]) -> Tuple[int, int]:
    existing = (
        pd.read_csv(ARTISTS_ARCHIVE_PATH, dtype=str)
        if ARTISTS_ARCHIVE_PATH.exists()
        else pd.DataFrame(columns=ARTISTS_OUT_COLUMNS)
    )
    before = len(existing)

    if not selected_artists:
        return before, before

    new_df = pd.DataFrame(selected_artists)
    if "genres" in new_df.columns:
        new_df["genres"] = new_df["genres"].apply(lambda v: str(v) if isinstance(v, list) else v)

    for c in ARTISTS_OUT_COLUMNS:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[ARTISTS_OUT_COLUMNS]

    merged_raw = pd.concat([existing, new_df], ignore_index=True)
    merged = deduplicate_artists_by_id(merged_raw)

    merged.to_csv(ARTISTS_ARCHIVE_PATH, index=False)
    return before, len(merged)


# ----------------------------
# Routes
# ----------------------------
@bp.get("/")
def index():
    return render_template("extender/index.html")


@bp.post("/api/fetch")
def api_fetch():
    payload = request.get_json(force=True) or {}
    token = (payload.get("token") or "").strip()
    include_all_playlists = bool(payload.get("include_all_playlists", False))

    if not token:
        return jsonify({"ok": False, "error": "Token mancante"}), 400

    try:
        me = _fetch_me(token)
        user_id = me.get("id")

        playlists = _fetch_playlists(token, user_id, include_all=include_all_playlists)

        raw_tracks: List[dict] = []
        seen_ids = set()

        def _add_track(t: dict) -> None:
            if t["id"] not in seen_ids:
                seen_ids.add(t["id"])
                raw_tracks.append(t)

        for p in playlists:
            for t in _fetch_playlist_tracks(token, p["id"]):
                _add_track(t)

        track_ids = [t["id"] for t in raw_tracks]
        features_df, features_warning = _fetch_audio_features(token, track_ids)
        tracks_out = _build_tracks_out(raw_tracks, features_df)

        artist_ids = sorted({
            a.get("id")
            for t in raw_tracks
            for a in (t.get("artists") or [])
            if a.get("id")
        })
        raw_artists = _fetch_artists(token, artist_ids)
        artists_out = _build_artists_out(raw_artists)

    except requests.HTTPError as e:
        return jsonify({"ok": False, "error": f"Spotify API error: {e}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    existing_track_ids = _load_existing_track_ids(TRACKS_ARCHIVE_PATH)
    existing_artists_min = _load_existing_artists_min(ARTISTS_ARCHIVE_PATH)
    existing_artist_ids = set(existing_artists_min["id"].dropna())

    tracks_fetched = len(tracks_out)
    new_tracks = tracks_out[~tracks_out["id"].isin(existing_track_ids)].reset_index(drop=True)
    new_tracks = _flag_invisible_tracks(new_tracks, artists_out, existing_artists_min)

    artists_fetched = len(artists_out)
    new_artists = artists_out[~artists_out["id"].isin(existing_artist_ids)].reset_index(drop=True)
    new_artists = new_artists.sort_values(
        by="name", key=lambda s: s.str.lower(), kind="stable"
    ).reset_index(drop=True)

    summary = {
        "playlists": len(playlists),
        "tracks_fetched": tracks_fetched,
        "tracks_new": len(new_tracks),
        "tracks_skipped_existing": tracks_fetched - len(new_tracks),
        "artists_fetched": artists_fetched,
        "artists_new": len(new_artists),
        "artists_skipped_existing": artists_fetched - len(new_artists),
        "invisible_count": int(new_tracks["invisible"].sum()) if len(new_tracks) else 0,
        "audio_features_warning": features_warning,
    }

    return jsonify({
        "ok": True,
        "new_tracks": _df_to_json_records(new_tracks),
        "new_artists": _df_to_json_records(new_artists),
        "summary": summary,
    })


@bp.post("/api/apply")
def api_apply():
    payload = request.get_json(force=True) or {}
    sel_tracks = payload.get("tracks") or []
    sel_artists = payload.get("artists") or []

    if not isinstance(sel_tracks, list) or not isinstance(sel_artists, list):
        return jsonify({"ok": False, "error": "tracks/artists devono essere liste"}), 400

    if not sel_tracks and not sel_artists:
        return jsonify({"ok": False, "error": "Nessuna riga selezionata"}), 400

    try:
        backup_dir = _backup_archive()
        tracks_before, tracks_after = _apply_new_tracks(sel_tracks)
        artists_before, artists_after = _apply_new_artists(sel_artists)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok": True,
        "tracks_before": tracks_before,
        "tracks_after": tracks_after,
        "artists_before": artists_before,
        "artists_after": artists_after,
        "backup_dir": str(backup_dir),
    })
