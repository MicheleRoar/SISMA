# app/blueprints/discovery/routes.py
from __future__ import annotations

import ast
import json
from typing import Dict, List, Tuple

import pandas as pd
from flask import Blueprint, current_app, jsonify, render_template, request

from app.services.presets import PRESETS
from app.services.genre_builder import search_genres
from app.services.region_genres import get_region_payload

bp = Blueprint("discovery", __name__)



@bp.get("/api/region-genres")
def api_region_genres():
    iso = (request.args.get("iso") or "").strip() or None
    key = (request.args.get("key") or "").strip() or None
    top_n = request.args.get("top_n", type=int) or 120

    payload = get_region_payload(iso=iso, key=key, top_n=top_n)
    status = 200 if payload.get("ok") else 400
    return jsonify(payload), status


# ----------------------------
# Helpers
# ----------------------------
def _get_float(name: str, default: float) -> float:
    try:
        return float(request.args.get(name, default))
    except Exception:
        return float(default)


def _parse_listish(x) -> List[str]:
    """
    Parse values like "['A','B']" into ["A","B"].
    Works for artists.csv / tracks.csv formats.
    Returns [] on failure.
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


def _normalize_artists_field(x) -> str:
    """
    Make artists human-readable:
    - "['Uli']" -> "Uli"
    - ['A','B'] -> "A; B"
    - "A;B" -> "A;B" (kept)
    """
    lst = _parse_listish(x)
    if lst:
        return "; ".join(lst)
    s = "" if x is None else str(x)
    return s.strip()


def _parse_csv_param(raw: str) -> List[str]:
    """
    Parse comma-separated list like:
      "Battiato, De André" -> ["Battiato", "De André"]
    """
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


def _parse_artist_weights_param(raw: str) -> Dict[str, float]:
    """
    Optional JSON dict.
    Example: {"Franco Battiato":0.33,"Fabrizio De André":0.33,"Italian pop":0.34}
    Returns {} on failure.
    """
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in obj.items():
            if k is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            out[str(k).strip()] = fv
        return out
    except Exception:
        return {}


def _get_selected_buckets_from_request():
    artists_raw = (request.args.get("artists") or "").strip()
    genres_raw = (request.args.get("genres") or "").strip()
    genre_single = (request.args.get("genre") or "").strip()

    exclude_artists_raw = (request.args.get("exclude_artists") or "").strip()
    exclude_genres_raw = (request.args.get("exclude_genres") or "").strip()

    artists = _parse_csv_param(artists_raw)
    genres_list = _parse_csv_param(genres_raw)
    exclude_artists = _parse_csv_param(exclude_artists_raw)
    exclude_genres = _parse_csv_param(exclude_genres_raw)

    if (not genres_list) and genre_single:
        genres_list = [genre_single]

    return (
        artists,
        genres_list,
        artists_raw,
        genres_raw,
        genre_single,
        exclude_artists,
        exclude_genres,
        exclude_artists_raw,
        exclude_genres_raw,
    )


def _parse_isos(raw: str) -> List[str]:
    return [x.strip().upper() for x in (raw or "").split(",") if x.strip()]

def _region_genres_union(isos: List[str], top_n: int = 120) -> List[str]:
    out: List[str] = []
    seen = set()
    for iso in isos:
        payload = get_region_payload(iso=iso, key=None, top_n=top_n)
        if not payload.get("ok"):
            continue
        for g in (payload.get("genres") or []):
            gs = str(g).strip()
            if gs and gs not in seen:
                seen.add(gs)
                out.append(gs)
    return out



def _sort_and_dedup(playlist_df: pd.DataFrame) -> pd.DataFrame:
    if playlist_df is None or len(playlist_df) == 0:
        return playlist_df

    sort_cols: List[str] = []
    ascending: List[bool] = []

    if "match" in playlist_df.columns:
        playlist_df["match"] = pd.to_numeric(playlist_df["match"], errors="coerce")
        sort_cols.append("match")
        ascending.append(False)

    if "popularity" in playlist_df.columns:
        playlist_df["popularity"] = pd.to_numeric(playlist_df["popularity"], errors="coerce").fillna(0)
        sort_cols.append("popularity")
        ascending.append(False)

    if sort_cols:
        playlist_df = playlist_df.sort_values(sort_cols, ascending=ascending, na_position="last")

    if "track_id" in playlist_df.columns:
        playlist_df = playlist_df.drop_duplicates(subset=["track_id"], keep="first")

    if {"track_name", "artists"}.issubset(playlist_df.columns):
        tmp_name = playlist_df["track_name"].astype(str).str.lower().str.strip()
        tmp_art = playlist_df["artists"].astype(str).str.lower().str.strip()
        playlist_df["_dedup_key"] = tmp_name + "||" + tmp_art
        playlist_df = playlist_df.drop_duplicates(subset=["_dedup_key"], keep="first").drop(columns=["_dedup_key"])

    return playlist_df


def _build_user_input_from_track_row(r: pd.Series) -> Dict[str, float]:
    return {
        "danceability": float(r.get("danceability", 0.5)),
        "energy": float(r.get("energy", 0.5)),
        "instrumentalness": float(r.get("instrumentalness", 0.5)),
        "valence": float(r.get("valence", 0.5)),
        "acousticness": float(r.get("acousticness", 0.2)),
        "speechiness": float(r.get("speechiness", 0.1)),
        "liveness": float(r.get("liveness", 0.1)),
        "tempo": float(r.get("tempo", 120.0)),
        "loudness": float(r.get("loudness", -10.0)),
        "key": float(r.get("key", -1)),
        "mode": float(r.get("mode", 1)),
        "time_signature": float(r.get("time_signature", 4)),
    }


def _apply_similarity_blend(user_input: Dict[str, float]) -> Dict[str, float]:
    """
    In similarity mode, we start from base track features, then blend towards user sliders.
    """
    dc = {
        "danceability": request.args.get("dc_danceability", "0") == "1",
        "energy": request.args.get("dc_energy", "0") == "1",
        "instrumentalness": request.args.get("dc_instrumentalness", "0") == "1",
        "valence": request.args.get("dc_valence", "0") == "1",
    }

    alpha_main = 0.35
    alpha_adv = 0.55

    def blend_feature(name: str, alpha: float, use_dc: bool = False) -> None:
        if use_dc and dc.get(name, False):
            return
        if name not in request.args:
            return
        try:
            target = float(request.args.get(name))
            base = float(user_input.get(name, target))
            user_input[name] = (1 - alpha) * base + alpha * target
        except Exception:
            return

    for f in ["danceability", "energy", "instrumentalness", "valence"]:
        blend_feature(f, alpha_main, use_dc=True)

    for f in ["acousticness", "speechiness", "liveness", "tempo", "loudness"]:
        blend_feature(f, alpha_adv, use_dc=False)

    # discrete overrides
    try:
        if "key" in request.args:
            k = int(float(request.args.get("key")))
            if k != -1:
                user_input["key"] = float(k)
    except Exception:
        pass

    try:
        if "mode" in request.args:
            m = int(float(request.args.get("mode")))
            if m in (0, 1):
                user_input["mode"] = float(m)
    except Exception:
        pass

    try:
        if "time_signature" in request.args:
            ts = int(float(request.args.get("time_signature")))
            if ts in (3, 4, 5, 6, 7):
                user_input["time_signature"] = float(ts)
    except Exception:
        pass

    return user_input


def _build_bucket_weights(
    artists: List[str],
    genres_list: List[str],
    weights_dict: Dict[str, float],
) -> Tuple[List[float], List[float]]:
    """
    Patch: quota-mix + priority bucket (artist > genre) + robust fallback.

    Rules:
      - If weights_dict provides explicit values for some buckets, use them.
      - Any missing bucket gets a default weight.
      - Artist buckets get a priority boost by default (unless explicitly set).
    """
    ARTIST_PRIORITY_BOOST = 1.6

    a_w: List[float] = []
    g_w: List[float] = []

    for a in artists:
        if a in weights_dict:
            w = float(weights_dict[a])
        else:
            w = ARTIST_PRIORITY_BOOST
        a_w.append(max(0.0, w))

    for g in genres_list:
        if g in weights_dict:
            w = float(weights_dict[g])
        else:
            w = 1.0
        g_w.append(max(0.0, w))

    if (sum(a_w) + sum(g_w)) <= 0:
        a_w = [ARTIST_PRIORITY_BOOST] * len(artists)
        g_w = [1.0] * len(genres_list)

    return a_w, g_w


# ----------------------------
# Routes
# ----------------------------
@bp.route("/", methods=["GET"])
def index():
    store = current_app.config["DATASTORE"]
    genres_top = store.get_genres()

    defaults = {
        "danceability": 0.5,
        "energy": 0.5,
        "instrumentalness": 0.5,
        "valence": 0.5,
        "acousticness": 0.2,
        "speechiness": 0.1,
        "liveness": 0.1,
        "tempo": 120.0,
        "loudness": -10.0,
        "key": 0.0,
        "mode": 1.0,
        "time_signature": 4.0,
        "genre": "",
        "artists": "",
        "genres": "",
        "exclude_artists": "",
        "exclude_genres": "",
        "region_isos": "",
    }

    return render_template(
        "discovery/index.html",
        genres=genres_top,
        presets=sorted(PRESETS.keys()),
        defaults=defaults,
        results=None,
        message="",
    )


@bp.route("/generate", methods=["GET"])
def generate():
    recommender = current_app.config["RECOMMENDER"]
    store = current_app.config["DATASTORE"]

    track_id = (request.args.get("track_id") or "").strip()
    region_isos_raw = (request.args.get("region_isos") or "").strip()

    (
        artists,
        genres_list,
        artists_raw,
        genres_raw,
        genre_single,
        exclude_artists,
        exclude_genres,
        exclude_artists_raw,
        exclude_genres_raw,
    ) = _get_selected_buckets_from_request()

    # ---- region merge (NO UI overwrite) ----
    top_n = request.args.get("top_n", type=int) or 120  # safe default
    region_isos = _parse_isos(region_isos_raw)
    region_genres = _region_genres_union(region_isos, top_n=top_n) if region_isos else []

    # include_genres final = manual + region (dedup), manual first
    include_genres_final: List[str] = []
    seen = set()

    for g in genres_list:
        if g and g not in seen:
            include_genres_final.append(g)
            seen.add(g)

    for g in region_genres:
        if g and g not in seen:
            include_genres_final.append(g)
            seen.add(g)

    artist_weights_raw = (request.args.get("artist_weights") or "").strip()
    weights_dict = _parse_artist_weights_param(artist_weights_raw)

    bucket_count = len(artists) + len(genres_list)


    # ----------------------------
    # RANGE PARSING (NEW)
    # ----------------------------
    def _get_range(name: str):
        raw_min = request.args.get(f"{name}_min", None)
        raw_max = request.args.get(f"{name}_max", None)

        mn = None
        mx = None

        try:
            if raw_min is not None and str(raw_min).strip() != "":
                mn = float(raw_min)
        except Exception:
            mn = None

        try:
            if raw_max is not None and str(raw_max).strip() != "":
                mx = float(raw_max)
        except Exception:
            mx = None

        if (mn is not None) and (mx is not None) and (mn > mx):
            mn, mx = mx, mn

        return mn, mx

    RANGE_FEATURES = [
        "danceability",
        "energy",
        "instrumentalness",
        "valence",
        "acousticness",
        "speechiness",
        "liveness",
        "tempo",
        "loudness",
    ]

    ranges = {}
    for f in RANGE_FEATURES:
        mn, mx = _get_range(f)
        if (mn is not None) or (mx is not None):
            ranges[f] = (mn, mx)

    # ----------------------------
    # DONTCARE
    # ----------------------------
    dontcare = {
        "danceability": request.args.get("dc_danceability", "0") == "1",
        "energy": request.args.get("dc_energy", "0") == "1",
        "instrumentalness": request.args.get("dc_instrumentalness", "0") == "1",
        "valence": request.args.get("dc_valence", "0") == "1",
    }

    # ----------------------------
    # Bucket weights (PATCH)
    # ----------------------------
    artist_weights, genre_weights = _build_bucket_weights(artists, genres_list, weights_dict)

    # ----------------------------
    # DISCOVERY PATCH: strict semantics + HARD tempo when requested
    # ----------------------------
    lock_tempo = "tempo" in ranges  # tempo becomes HARD only if user provided tempo_min/max

    # ---------------------------
    # SIMILARITY MODE (track_id)
    # ---------------------------
    if track_id:
        df = store.get_df()
        row = df.loc[df["track_id"].astype(str) == track_id]
        if row.empty:
            return jsonify({"error": "track_not_found"}), 404
        r = row.iloc[0]

        user_input = _build_user_input_from_track_row(r)
        user_input = _apply_similarity_blend(user_input)


        # 1) Build a BIG pool around the similarity user_input, constrained by ranges.
        #    lock_tempo=True iff user set tempo_min/max (hard BPM when requested)
        pool_df = recommender.build_pool(
            user_input=user_input,
            ranges=ranges,
            lock_tempo=lock_tempo,
            allow_explicit=False,
            exclude_track_ids={track_id},   # avoid recommending the seed track back
            shuffle_within_top=True,
            random_state=42,
            dontcare=dontcare,
        )

        if "_row_idx" not in pool_df.columns:
            raise RuntimeError("build_pool must return a '_row_idx' column (update recommender.py)")

        pool_idx = pool_df["_row_idx"].to_numpy()


        # 2) Second stage: filter/rank INSIDE the pool
        playlist_df = recommender.recommend_from_pool(
            user_input=user_input,
            pool_idx=pool_idx,
            k=50,
            max_per_artist=2,
            include_artists=artists,
            include_genres=include_genres_final,
            include_mode="prefer",          # DISCOVERY: adding "pop" must not shrink the pool
            exclude_artists=exclude_artists,
            exclude_genres=exclude_genres,
            exclude_track_ids=set(),
            allow_explicit=False,
            dontcare=dontcare,
            weight_overrides=None,
            shuffle_within_top=True,
            random_state=42,
        )


        playlist_df = _sort_and_dedup(playlist_df).head(50).reset_index(drop=True)

        if request.args.get("format", "").lower() == "json":
            return jsonify(playlist_df.to_dict(orient="records"))

        genres_top = store.get_genres()

        defaults = {
            **user_input,
            "genre": genre_single,
            "artists": artists_raw,
            "genres": genres_raw,
            "exclude_artists": exclude_artists_raw,
            "exclude_genres": exclude_genres_raw,
            "region_isos": region_isos_raw,

        }

        for f in RANGE_FEATURES:
            mn, mx = ranges.get(f, (None, None))
            if mn is not None:
                defaults[f"{f}_min"] = mn
            if mx is not None:
                defaults[f"{f}_max"] = mx

        selected_label = f'{r.get("track_name","")} — {_normalize_artists_field(r.get("artists",""))}'

        # informative message when tempo is locked and results are short
        extra = ""
        if lock_tempo and "bpm" in playlist_df.columns and len(playlist_df) < 50:
            tmin, tmax = ranges.get("tempo", (None, None))
            if tmin is not None or tmax is not None:
                extra = f" Tempo locked ({tmin}–{tmax} BPM), widen it to get more tracks."

        msg = f"Similar to: <strong>{selected_label}</strong> ({len(playlist_df)} tracks).{extra}"

        return render_template(
            "discovery/index.html",
            genres=genres_top,
            presets=sorted(PRESETS.keys()),
            defaults=defaults,
            results=playlist_df.to_dict(orient="records"),
            message=msg,
        )

    # ---------------------------
    # CUSTOM MODE
    # ---------------------------
    user_input = {
        "danceability": _get_float("danceability", 0.5),
        "energy": _get_float("energy", 0.5),
        "instrumentalness": _get_float("instrumentalness", 0.0),
        "valence": _get_float("valence", 0.5),
        "acousticness": _get_float("acousticness", 0.2),
        "speechiness": _get_float("speechiness", 0.1),
        "liveness": _get_float("liveness", 0.1),
        "tempo": _get_float("tempo", 120.0),
        "loudness": _get_float("loudness", -10.0),
        "key": _get_float("key", 0.0),
        "mode": _get_float("mode", 1.0),
        "time_signature": _get_float("time_signature", 4.0),
    }

    # 1) Build a BIG pool from preset-like ranges (soft, but tempo can be HARD)
    pool_df = recommender.build_pool(
        user_input=user_input,
        ranges=ranges,
        lock_tempo=lock_tempo,          # True iff tempo range provided
        allow_explicit=False,
        exclude_track_ids=set(),        # custom mode: nothing to exclude by id
        shuffle_within_top=True,
        random_state=42,
        dontcare=dontcare,
    )

    # pool indices for second-stage filtering/ranking
    if "_row_idx" not in pool_df.columns:
        raise RuntimeError("build_pool must return a '_row_idx' column (update recommender.py)")

    pool_idx = pool_df["_row_idx"].to_numpy()


    # 2) Second stage: apply preferences INSIDE the pool
    playlist_df = recommender.recommend_from_pool(
        user_input=user_input,
        pool_idx=pool_idx,
        k=50,
        max_per_artist=2,
        include_artists=artists,
        include_genres=include_genres_final,
        include_mode="prefer",          # discovery behavior: don't shrink pool
        exclude_artists=exclude_artists,
        exclude_genres=exclude_genres,
        exclude_track_ids={track_id},   # keep it out even after ranking
        allow_explicit=False,
        dontcare=dontcare,
        weight_overrides=None,
        shuffle_within_top=True,
        random_state=42,
    )


    playlist_df = _sort_and_dedup(playlist_df).head(50).reset_index(drop=True)

    if request.args.get("format", "").lower() == "json":
        return jsonify(playlist_df.to_dict(orient="records"))

    genres_top = store.get_genres()

    defaults = {
        **user_input,
        "genre": genre_single,
        "artists": artists_raw,
        "genres": genres_raw,
        "exclude_artists": exclude_artists_raw,
        "exclude_genres": exclude_genres_raw,
        "region_isos": region_isos_raw,
    }
    for f in RANGE_FEATURES:
        mn, mx = ranges.get(f, (None, None))
        if mn is not None:
            defaults[f"{f}_min"] = mn
        if mx is not None:
            defaults[f"{f}_max"] = mx

    # informative message when tempo is locked and results are short
    extra = ""
    if lock_tempo and len(playlist_df) < 50:
        tmin, tmax = ranges.get("tempo", (None, None))
        if tmin is not None or tmax is not None:
            extra = f" Tempo locked ({tmin}–{tmax} BPM), widen it to get more tracks."

    return render_template(
        "discovery/index.html",
        genres=genres_top,
        presets=sorted(PRESETS.keys()),
        defaults=defaults,
        results=playlist_df.to_dict(orient="records"),
        message=f"Generated playlist with {len(playlist_df)} tracks.{extra}",
    )


@bp.route("/preset", methods=["GET"])
def preset():
    name = (request.args.get("name") or "").strip()
    p = PRESETS.get(name)
    if not p:
        return jsonify({"error": "preset_not_found"}), 404
    return jsonify({"name": name, **p})


@bp.route("/track_search", methods=["GET"])
def track_search():
    store = current_app.config["DATASTORE"]
    df = store.get_df()

    q = (request.args.get("q") or "").strip().lower()
    if len(q) < 2:
        return jsonify([])

    name = df["track_name"].astype(str).str.lower()
    artists = df["artists"].astype(str).str.lower()

    mask = name.str.contains(q, regex=False) | artists.str.contains(q, regex=False)
    hits = df.loc[mask, ["track_id", "track_name", "artists", "popularity"]].copy()

    if hits.empty:
        return jsonify([])

    if "popularity" in hits.columns:
        hits["popularity"] = pd.to_numeric(hits["popularity"], errors="coerce").fillna(0)
        hits = hits.sort_values("popularity", ascending=False)

    hits = hits.head(10)

    results = []
    for _, rr in hits.iterrows():
        tid = str(rr.get("track_id", ""))
        label = f'{rr.get("track_name","")} — {_normalize_artists_field(rr.get("artists",""))}'
        results.append({"track_id": tid, "label": label})

    return jsonify(results)


@bp.route("/track_features", methods=["GET"])
def track_features():
    store = current_app.config["DATASTORE"]
    df = store.get_df()

    track_id = (request.args.get("track_id") or "").strip()
    if not track_id:
        return jsonify({"error": "missing_track_id"}), 400

    row = df.loc[df["track_id"].astype(str) == track_id]
    if row.empty:
        return jsonify({"error": "track_not_found"}), 404

    r = row.iloc[0]

    return jsonify(
        {
            "track_id": track_id,
            "label": f'{r.get("track_name","")} — {_normalize_artists_field(r.get("artists",""))}',
            "features": {
                "danceability": float(r.get("danceability", 0.5)),
                "energy": float(r.get("energy", 0.5)),
                "instrumentalness": float(r.get("instrumentalness", 0.5)),
                "valence": float(r.get("valence", 0.5)),
                "acousticness": float(r.get("acousticness", 0.2)),
                "speechiness": float(r.get("speechiness", 0.1)),
                "liveness": float(r.get("liveness", 0.1)),
                "tempo": float(r.get("tempo", 120.0)),
                "loudness": float(r.get("loudness", -10.0)),
                "key": int(float(r.get("key", -1))),
                "mode": int(float(r.get("mode", 1))),
                "time_signature": int(float(r.get("time_signature", 4))),
            },
        }
    )


@bp.route("/genre_search", methods=["GET"])
def genre_search():
    """
    Task #2: robust genre search.
    Supports substring + multi-token (e.g. "brit pop", "italia", "rock").
    """
    store = current_app.config["DATASTORE"]

    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify([])

    genres = store.get_all_genres()  # list[str]
    hits = search_genres(genres, q, limit=200)
    return jsonify(hits)


@bp.route("/artist_search", methods=["GET"])
def artist_search():
    store = current_app.config["DATASTORE"]
    q = (request.args.get("q") or "").strip().lower()

    if len(q) < 3:
        return jsonify([])

    artists = getattr(store, "ui_artists", [])
    if not artists:
        return jsonify([])

    out: List[str] = []

    for a in artists:
        al = a.lower()
        if al.startswith(q):
            out.append(a)
            if len(out) >= 50:
                return jsonify(out)

    for a in artists:
        al = a.lower()
        if q in al and not al.startswith(q):
            out.append(a)
            if len(out) >= 50:
                break

    return jsonify(out)