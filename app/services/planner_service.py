# app/services/planner_service.py
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.services.presets import PRESETS
from app.services.recommender import PlaylistRecommender


# ----------------------------
# small utils
# ----------------------------

VALID_WEEKDAYS = {0, 1, 2, 3, 4, 5, 6}
DEFAULT_WEEKDAYS = [1, 2, 3, 4, 5]


def _dayiso_to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _stable_int_seed(*parts: str, mod: int = 2_147_483_647) -> int:
    """
    Deterministic seed from strings, stable across processes.
    """
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if not s:
        return bool(default)
    return s in {"1", "true", "yes", "on"}


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, tuple):
        return [str(v).strip() for v in x if str(v).strip()]

    s = str(x).strip()
    if not s:
        return []

    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]

    return [s]


def _clean_range_pair(v: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Accepts:
      - [min,max] or (min,max)
      - {"min":..,"max":..}
      - single numeric (treated as midpoint -> no hard bounds)
    Returns (mn, mx) as floats or None.
    """
    if v is None:
        return (None, None)

    if isinstance(v, dict):
        mn = v.get("min", None)
        mx = v.get("max", None)
        try:
            mn = None if mn is None or mn == "" else float(mn)
        except Exception:
            mn = None
        try:
            mx = None if mx is None or mx == "" else float(mx)
        except Exception:
            mx = None
        return (mn, mx)

    if isinstance(v, (list, tuple)) and len(v) >= 2:
        a, b = v[0], v[1]
        try:
            a = None if a is None or a == "" else float(a)
        except Exception:
            a = None
        try:
            b = None if b is None or b == "" else float(b)
        except Exception:
            b = None
        return (a, b)

    return (None, None)


def _midpoint(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return float(b)
    if b is None:
        return float(a)
    return (float(a) + float(b)) / 2.0


def _normalize_weekdays(values: Any) -> List[int]:
    out: List[int] = []
    if not isinstance(values, list):
        values = DEFAULT_WEEKDAYS

    for x in values:
        try:
            v = int(x)
            if v in VALID_WEEKDAYS:
                out.append(v)
        except Exception:
            continue

    out = sorted(set(out))
    return out or DEFAULT_WEEKDAYS.copy()


def _compute_day_isos(window_start: date, weeks: int, weekdays: List[int], cols: int = 14) -> List[str]:
    """
    Compute actual slot dates inside the visible planner window.
    JS weekdays convention:
      Mon=1 .. Sat=6, Sun=0
    """
    day_isos: List[str] = []
    weekday_set = set(_normalize_weekdays(weekdays))

    for c in range(cols):
        d = window_start + timedelta(days=c)
        week_index = (c // 7) + 1
        if week_index > int(weeks):
            continue

        js_day = (d.weekday() + 1) % 7
        if js_day not in weekday_set:
            continue

        day_isos.append(d.isoformat())

    return day_isos


def _df_to_tracks_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Frontend-friendly payload for planner slots.
    Includes summary-friendly audio features.
    """
    if df is None or df.empty:
        return []

    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        bpm_val = r.get("bpm", r.get("tempo", None))
        match_val = r.get("match", None)
        popularity_val = r.get("popularity", None)

        energy_val = r.get("energy", None)
        valence_val = r.get("valence", None)
        danceability_val = r.get("danceability", None)

        out.append({
            "track_id": str(r.get("track_id", "") or "").strip(),
            "title": str(r.get("track_name", "") or "").strip(),
            "artist": str(r.get("artists", "") or "").strip(),
            "genre": str(r.get("track_genre", "") or "").strip(),

            "bpm": None if pd.isna(bpm_val) else float(bpm_val),
            "match": None if pd.isna(match_val) else float(match_val),
            "popularity": None if pd.isna(popularity_val) else float(popularity_val),

            "energy": None if pd.isna(energy_val) else float(energy_val),
            "valence": None if pd.isna(valence_val) else float(valence_val),
            "danceability": None if pd.isna(danceability_val) else float(danceability_val),
        })
    return out


# ----------------------------
# Planner request model
# ----------------------------

@dataclass
class PlannerRequest:
    """
    Canonical form used by PlannerService. Built from a Discovery payload.
    """
    user_input: Dict[str, float]
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]]

    include_artists: List[str]
    include_genres: List[str]
    include_keywords: List[str]
    include_mode: str  # "must" | "must_any" | "prefer"

    allow_explicit: bool

    exclude_artists: List[str]
    exclude_genres: List[str]
    exclude_keywords: List[str]

    dontcare: Dict[str, bool]
    weight_overrides: Dict[str, float]

    strict_semantics: bool
    lock_tempo: bool
    pool_size: int

    popularity_tier: str


# ----------------------------
# Planner Service
# ----------------------------

class PlannerService:
    """
    Planner behavior:
      - strict semantics always ON
      - tempo lock honored only if tempo range exists
      - a single slot is generated from one global master pool
      - total tracks = number of actual slot occurrences * k
      - distribution across days uses popularity buckets
    """

    def __init__(self, recommender: PlaylistRecommender):
        self.rec = recommender

    @classmethod
    def from_app(cls, app) -> "PlannerService":
        rec = app.config.get("RECOMMENDER", None)
        if rec is None:
            raise RuntimeError("Missing app.config['RECOMMENDER']")
        if not isinstance(rec, PlaylistRecommender):
            if not hasattr(rec, "recommend_from_pool"):
                raise RuntimeError("RECOMMENDER is not a PlaylistRecommender-like object")
        return cls(rec)

    # ---------- payload parsing ----------

    def _parse_discovery_payload(self, payload: Dict[str, Any]) -> PlannerRequest:
        p = dict(payload or {})

        if "discovery" in p and isinstance(p["discovery"], dict):
            base = {k: v for k, v in p.items() if k != "discovery"}
            base.update(p["discovery"])
            p = base

        preset = (p.get("preset") or p.get("preset_name") or "").strip()
        if preset and preset in PRESETS:
            preset_payload = copy.deepcopy(PRESETS[preset])
            if isinstance(preset_payload, dict):
                preset_payload.update({k: v for k, v in p.items() if k not in {"preset", "preset_name"}})
                p = preset_payload

        # UI alias: mood -> valence
        if "mood_min" in p and "valence_min" not in p:
            p["valence_min"] = p.get("mood_min")
        if "mood_max" in p and "valence_max" not in p:
            p["valence_max"] = p.get("mood_max")
        if "dc_mood" in p and "dc_valence" not in p:
            p["dc_valence"] = p.get("dc_mood")

        # explicit user_input
        user_input: Dict[str, float] = {}
        if isinstance(p.get("user_input"), dict):
            for k, v in p["user_input"].items():
                try:
                    if v is None or v == "":
                        continue
                    user_input[str(k)] = float(v)
                except Exception:
                    continue

        # canonical ranges
        ranges_raw = p.get("ranges") or p.get("feature_ranges") or p.get("filters") or {}
        ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        if isinstance(ranges_raw, dict):
            for f, v in ranges_raw.items():
                mn, mx = _clean_range_pair(v)
                if mn is None and mx is None:
                    continue
                ranges[str(f)] = (mn, mx)

        # fallback from flat form fields
        if not ranges:
            features = [
                "danceability",
                "energy",
                "valence",
                "tempo",
                "loudness",
                "instrumentalness",
                "acousticness",
                "speechiness",
                "liveness",
                "year",
            ]
            for f in features:
                mn_key = f"{f}_min"
                mx_key = f"{f}_max"
                if mn_key in p or mx_key in p:
                    try:
                        mn = p.get(mn_key, None)
                        mx = p.get(mx_key, None)
                        mn = None if mn is None or mn == "" else float(mn)
                        mx = None if mx is None or mx == "" else float(mx)
                        if mn is not None or mx is not None:
                            ranges[f] = (mn, mx)
                    except Exception:
                        continue

        # derive midpoint user_input from ranges when missing
        if not user_input:
            for f, (mn, mx) in ranges.items():
                if f == "year":
                    continue
                m = _midpoint(mn, mx)
                if m is not None:
                    user_input[f] = float(m)

        include_artists = _as_list_str(
            p.get("include_artists") or p.get("artists_include") or p.get("artists") or []
        )
        include_genres = _as_list_str(
            p.get("include_genres") or p.get("genres_include") or p.get("genres") or []
        )
        include_keywords = _as_list_str(
            p.get("include_keywords") or p.get("keywords_include") or p.get("keywords") or []
        )

        exclude_artists = _as_list_str(p.get("exclude_artists") or p.get("artists_exclude") or [])
        exclude_genres = _as_list_str(p.get("exclude_genres") or p.get("genres_exclude") or [])
        exclude_keywords = _as_list_str(p.get("exclude_keywords") or p.get("keywords_exclude") or [])

        include_mode = (p.get("include_mode") or "prefer").strip().lower()
        if include_mode not in {"must", "must_any", "prefer"}:
            include_mode = "prefer"

        allow_explicit = _as_bool(p.get("allow_explicit", 0), default=False)

        dontcare: Dict[str, bool] = {}
        if isinstance(p.get("dontcare"), dict):
            for f, flag in p["dontcare"].items():
                dontcare[str(f)] = _as_bool(flag, default=False)

        for f in [
            "danceability",
            "energy",
            "loudness",
            "valence",
            "tempo",
            "instrumentalness",
            "acousticness",
            "speechiness",
            "liveness",
            "year",
        ]:
            kdc = f"dc_{f}"
            if kdc in p and f not in dontcare:
                dontcare[f] = _as_bool(p.get(kdc), default=False)

        weight_overrides: Dict[str, float] = {}
        if isinstance(p.get("weight_overrides"), dict):
            for f, w in p["weight_overrides"].items():
                try:
                    weight_overrides[str(f)] = float(w)
                except Exception:
                    continue

        # Planner policy: always strict. No semantic widening circus.
        strict_semantics = True

        # lock_tempo only matters if tempo range exists
        lock_tempo = _as_bool(p.get("lock_tempo", True), default=True)
        if "tempo" not in ranges:
            lock_tempo = False

        pool_size = int(p.get("pool_size", 10000) or 10000)
        pool_size = max(2000, min(pool_size, 50000))

        popularity_tier = str(p.get("popularity_tier", "") or "").strip().lower()

        return PlannerRequest(
            user_input=user_input,
            ranges=ranges,
            include_artists=include_artists,
            include_genres=include_genres,
            include_keywords=include_keywords,
            include_mode=include_mode,
            allow_explicit=allow_explicit,
            exclude_artists=exclude_artists,
            exclude_genres=exclude_genres,
            exclude_keywords=exclude_keywords,
            dontcare=dontcare,
            weight_overrides=weight_overrides,
            strict_semantics=strict_semantics,
            lock_tempo=lock_tempo,
            pool_size=pool_size,
            popularity_tier=popularity_tier,
        )

    # ---------- pool building ----------

    def _build_master_pool(
        self,
        req: PlannerRequest,
        *,
        total_needed: int,
        random_state: int,
        exclude_track_ids: Set[str],
    ) -> pd.DataFrame:
        """
        Build one global candidate pool for the whole slot.
        """
        universe_idx = self.rec.build_universe_indices(
            include_artists=req.include_artists,
            include_genres=req.include_genres,
            exclude_artists=req.exclude_artists,
            exclude_genres=req.exclude_genres,
        )

        target_pool_size = max(int(total_needed) * 4, req.pool_size, 2000)
        target_pool_size = min(target_pool_size, 50000)

        pool = self.rec.build_pool(
            user_input=req.user_input,
            universe_idx=universe_idx,
            pool_size=target_pool_size,
            allow_explicit=req.allow_explicit,
            exclude_track_ids=exclude_track_ids,
            shuffle_within_top=True,
            random_state=int(random_state),
            dontcare=req.dontcare,
            weight_overrides=req.weight_overrides,
            ranges=req.ranges,
            lock_tempo=req.lock_tempo,
            popularity_tier=req.popularity_tier,
            popularity_genres=req.include_genres,
        )

        if pool is None or pool.empty:
            return pd.DataFrame()

        # second-pass filtering to fully honor include/exclude/keywords inside the built pool
        if "_row_idx" not in pool.columns:
            return pd.DataFrame()

        pool_idx = pool["_row_idx"].to_numpy()

        pool2 = self.rec.recommend_from_pool(
            user_input=req.user_input,
            pool_idx=pool_idx,
            k=min(len(pool_idx), target_pool_size),
            max_per_artist=999999,  # don't constrain here; constrain during day assignment
            exclude_track_ids=set(exclude_track_ids),
            allow_explicit=req.allow_explicit,
            shuffle_within_top=False,
            random_state=int(random_state),
            weight_overrides=req.weight_overrides,
            dontcare=req.dontcare,
            include_artists=req.include_artists,
            include_genres=req.include_genres,
            include_mode=req.include_mode,
            exclude_artists=req.exclude_artists,
            exclude_genres=req.exclude_genres,
            include_keywords=req.include_keywords,
            exclude_keywords=req.exclude_keywords,
        )

        if pool2 is None or pool2.empty:
            return pd.DataFrame()

        if "track_id" in pool2.columns:
            pool2 = pool2.drop_duplicates(subset=["track_id"], keep="first").copy()

        if "popularity" in pool2.columns:
            pool2["popularity"] = pd.to_numeric(pool2["popularity"], errors="coerce").fillna(0.0)
        else:
            pool2["popularity"] = 0.0

        if "match" in pool2.columns:
            pool2["match"] = pd.to_numeric(pool2["match"], errors="coerce").fillna(0.0)
        else:
            pool2["match"] = 0.0

        pool2 = pool2.sort_values(["match", "popularity"], ascending=[False, False]).reset_index(drop=True)
        return pool2

    def _split_popularity_buckets(self, pool_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the master pool into ranked popularity buckets.
        """
        if pool_df is None or pool_df.empty:
            return {
                "top_seed": pd.DataFrame(),
                "mid_high": pd.DataFrame(),
                "mid": pd.DataFrame(),
                "deep": pd.DataFrame(),
            }

        df = pool_df.copy()
        df["popularity"] = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0.0)

        # Keep match first, popularity second.
        df = df.sort_values(["match", "popularity"], ascending=[False, False]).reset_index(drop=True)

        n = len(df)
        if n <= 4:
            return {
                "top_seed": df.iloc[:1].copy(),
                "mid_high": df.iloc[1:2].copy(),
                "mid": df.iloc[2:3].copy(),
                "deep": df.iloc[3:].copy(),
            }

        b1 = max(1, int(n * 0.15))
        b2 = max(b1 + 1, int(n * 0.40))
        b3 = max(b2 + 1, int(n * 0.70))

        return {
            "top_seed": df.iloc[:b1].copy(),
            "mid_high": df.iloc[b1:b2].copy(),
            "mid": df.iloc[b2:b3].copy(),
            "deep": df.iloc[b3:].copy(),
        }

    def _bucket_quota_for_k(self, k: int) -> List[Tuple[str, int]]:
        """
        Quotas sum to k. Scaled from 10/15/15/10 for k=50.
        """
        k = int(max(1, k))
        ratios = [
            ("top_seed", 0.20),
            ("mid_high", 0.30),
            ("mid", 0.30),
            ("deep", 0.20),
        ]

        quotas: List[Tuple[str, int]] = []
        running = 0
        for i, (name, r) in enumerate(ratios):
            if i < len(ratios) - 1:
                q = int(round(k * r))
                quotas.append((name, q))
                running += q
            else:
                quotas.append((name, max(0, k - running)))

        # tiny correction in case rounding overshoots
        total = sum(q for _, q in quotas)
        if total > k:
            extra = total - k
            fixed: List[Tuple[str, int]] = []
            for name, q in reversed(quotas):
                if extra > 0 and q > 0:
                    dec = min(extra, q)
                    q -= dec
                    extra -= dec
                fixed.append((name, q))
            quotas = list(reversed(fixed))

        return quotas

    def _distribute_pool_across_days(
        self,
        pool_df: pd.DataFrame,
        *,
        day_isos: List[str],
        k: int,
        max_per_artist: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Distribute one master pool across all slot days.
        No track reuse inside the slot plan.
        """
        day_isos = [str(d).strip() for d in (day_isos or []) if str(d).strip()]
        day_isos = sorted(dict.fromkeys(day_isos))

        if pool_df is None or pool_df.empty:
            return {d: [] for d in day_isos}

        buckets = self._split_popularity_buckets(pool_df)
        bucket_order = self._bucket_quota_for_k(k)

        used_track_ids: Set[str] = set()
        playlists_by_day: Dict[str, List[Dict[str, Any]]] = {}

        for day_iso in day_isos:
            day_rows: List[pd.Series] = []
            artist_counts: Dict[str, int] = {}

            def can_take(row: pd.Series) -> bool:
                tid = str(row.get("track_id", "") or "").strip()
                artist = str(row.get("artists", "") or "").strip().lower()

                if not tid or tid in used_track_ids:
                    return False
                if artist_counts.get(artist, 0) >= int(max_per_artist):
                    return False
                return True

            def take_row(row: pd.Series) -> None:
                tid = str(row.get("track_id", "") or "").strip()
                artist = str(row.get("artists", "") or "").strip().lower()

                day_rows.append(row)
                used_track_ids.add(tid)
                artist_counts[artist] = artist_counts.get(artist, 0) + 1

            # pass 1: follow bucket quotas
            for bucket_name, quota in bucket_order:
                if quota <= 0:
                    continue

                taken = 0
                bucket_df = buckets.get(bucket_name, pd.DataFrame())
                if bucket_df is None or bucket_df.empty:
                    continue

                for _, row in bucket_df.iterrows():
                    if not can_take(row):
                        continue
                    take_row(row)
                    taken += 1
                    if taken >= quota:
                        break

            # pass 2: fill remaining from whole pool
            if len(day_rows) < int(k):
                for _, row in pool_df.iterrows():
                    if not can_take(row):
                        continue
                    take_row(row)
                    if len(day_rows) >= int(k):
                        break

            day_df = pd.DataFrame(day_rows).head(int(k)).copy()
            playlists_by_day[day_iso] = _df_to_tracks_payload(day_df)

        return playlists_by_day

    # ---------- public generation ----------

    def generate_for_discovery_payload(
        self,
        *,
        discovery_payload: Dict[str, Any],
        day_isos: List[str],
        k: int = 50,
        max_per_artist: int = 2,
        cooldown_days: int = 2,
        exclude_track_ids_global: Optional[Set[str]] = None,
        seed: int = 42,
        slot_id: str = "slot",
    ) -> Dict[str, Any]:
        """
        Generate playlists for multiple days for a single slot.

        New behavior:
          - build ONE global master pool
          - total_needed = len(day_isos) * k
          - distribute across days using popularity buckets
          - no reuse of tracks inside this slot generation
        """
        exclude_track_ids_global = set(exclude_track_ids_global or set())
        day_isos = [str(d).strip() for d in (day_isos or []) if str(d).strip()]
        day_isos = sorted(dict.fromkeys(day_isos))

        req = self._parse_discovery_payload(discovery_payload)

        total_needed = len(day_isos) * int(k)
        rs = _stable_int_seed(str(seed), slot_id, "master_pool")

        master_pool = self._build_master_pool(
            req,
            total_needed=total_needed,
            random_state=rs,
            exclude_track_ids=exclude_track_ids_global,
        )

        playlists_by_day = self._distribute_pool_across_days(
            master_pool,
            day_isos=day_isos,
            k=int(k),
            max_per_artist=int(max_per_artist),
        )

        report: Dict[str, Any] = {
            "k": int(k),
            "max_per_artist": int(max_per_artist),
            "cooldown_days": int(cooldown_days),  # kept for compatibility; not active in slot-global mode
            "strict_semantics": bool(req.strict_semantics),
            "lock_tempo": bool(req.lock_tempo),
            "include_mode": req.include_mode,
            "popularity_tier": req.popularity_tier,
            "days": day_isos,
            "num_days": int(len(day_isos)),
            "tracks_per_day": int(k),
            "total_needed": int(total_needed),
            "master_pool_size": int(len(master_pool)) if master_pool is not None else 0,
            "include_artists": req.include_artists,
            "include_genres": req.include_genres,
            "include_keywords": req.include_keywords,
            "exclude_artists": req.exclude_artists,
            "exclude_genres": req.exclude_genres,
            "exclude_keywords": req.exclude_keywords,
            "ranges": req.ranges,
        }

        return {
            "playlistsByDay": playlists_by_day,
            "report": report,
        }

    def generate_for_preset_occurrences(
        self,
        preset_name: str,
        *,
        day_isos: List[str],
        k: int = 50,
        max_per_artist: int = 2,
        cooldown_days: int = 2,
        exclude_track_ids_global: Optional[Set[str]] = None,
        seed: int = 42,
        slot_id: str = "slot",
    ) -> Dict[str, Any]:
        preset_name = (preset_name or "").strip()
        if preset_name not in PRESETS:
            return {
                "playlistsByDay": {},
                "report": {"error": "invalid_preset", "preset": preset_name},
            }

        payload = copy.deepcopy(PRESETS[preset_name])
        if not isinstance(payload, dict):
            payload = {"preset": preset_name}

        payload.setdefault("preset", preset_name)

        return self.generate_for_discovery_payload(
            discovery_payload=payload,
            day_isos=day_isos,
            k=k,
            max_per_artist=max_per_artist,
            cooldown_days=cooldown_days,
            exclude_track_ids_global=exclude_track_ids_global,
            seed=seed,
            slot_id=slot_id,
        )

    # ---------- one-shot plan preparation ----------

    def prepare_plan(
        self,
        *,
        discovery_payload: Dict[str, Any],
        rule: Dict[str, Any],
        k: int = 50,
        max_per_artist: int = 2,
        cooldown_days: int = 2,
        window_start_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete planner slot package for the frontend.

        Returns:
          {
            startDateISO,
            slots: {
              rule_id: {
                ...slot metadata...,
                playlistsByDay,
                report
              }
            },
            report
          }
        """

        START_DAY = "10:00"
        END_DAY = "24:00"
        STEP_MIN = 30
        COLS = 14

        def _time_to_min(t: str) -> int:
            hh, mm = t.split(":")
            return int(hh) * 60 + int(mm)

        def _clamp_step(m: int) -> int:
            return int(round(m / STEP_MIN) * STEP_MIN)

        def _monday_of(d: date) -> date:
            return d - timedelta(days=d.weekday())

        if window_start_iso:
            try:
                window_start = _monday_of(_dayiso_to_date(window_start_iso))
            except Exception:
                window_start = _monday_of(date.today())
        else:
            window_start = _monday_of(date.today())

        startDateISO = window_start.isoformat()

        start_min = _time_to_min(START_DAY)
        end_min = _time_to_min(END_DAY)
        rows = max(0, int((end_min - start_min) / STEP_MIN))

        name = str(rule.get("name") or "Slot").strip() or "Slot"
        color = str(rule.get("color") or "#FFD403").strip() or "#FFD403"
        slot_start = str(rule.get("start") or "10:00").strip() or "10:00"
        slot_end = str(rule.get("end") or "11:00").strip() or "11:00"

        weeks = int(rule.get("weeks", 2) or 2)
        weeks = max(1, min(weeks, 8))

        weekdays = _normalize_weekdays(rule.get("weekdays") or DEFAULT_WEEKDAYS)

        day_isos = _compute_day_isos(window_start, weeks, weekdays, cols=COLS)

        # stable fingerprint
        disc_fingerprint = hashlib.sha256(
            json.dumps(discovery_payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()[:10]

        rule_id = (
            f"r_{slot_start.replace(':', '')}_{slot_end.replace(':', '')}"
            f"_w{weeks}_d{''.join(map(str, weekdays))}_{disc_fingerprint}"
        )

        gen = self.generate_for_discovery_payload(
            discovery_payload=discovery_payload,
            day_isos=day_isos,
            k=int(k),
            max_per_artist=int(max_per_artist),
            cooldown_days=int(cooldown_days),
            exclude_track_ids_global=set(),
            seed=_stable_int_seed("plan", rule_id, startDateISO),
            slot_id=rule_id,
        )

        playlistsByDay = gen.get("playlistsByDay", {}) or {}
        report = gen.get("report", {}) or {}

        # optional grid-like placement logic kept local for parity with current planner math
        a = _clamp_step(_time_to_min(slot_start))
        b = _clamp_step(_time_to_min(slot_end))
        a = max(start_min, min(a, end_min))
        b = max(start_min, min(b, end_min))

        # We do not return grid/rows/cols because the current frontend rebuilds them locally.
        # But the math is kept here for future expansion and sanity.
        _ = rows
        if b <= a:
            report["warning"] = "slot_end_not_after_slot_start"

        slot_payload = {
            "id": rule_id,
            "name": name,
            "color": color,
            "start": slot_start,
            "end": slot_end,
            "weeks": weeks,
            "weekdays": weekdays,
            "k": int(k),
            "max_per_artist": int(max_per_artist),
            "cooldown_days": int(cooldown_days),
            "discovery": discovery_payload,
            "playlistsByDay": playlistsByDay,
            "report": report,
        }

        return {
            "startDateISO": startDateISO,
            "slots": {rule_id: slot_payload},
            "report": report,
        }