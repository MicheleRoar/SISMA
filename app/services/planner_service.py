# app/services/planner_service.py
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.services.presets import PRESETS
from app.services.recommender import PlaylistRecommender


# ----------------------------
# small utils
# ----------------------------

def _dayiso_to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _stable_int_seed(*parts: str, mod: int = 2_147_483_647) -> int:
    """
    Deterministic seed from strings, stable across processes.
    """
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


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

    # NEW: allow CSV strings "a,b,c"
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

    # scalar -> ignore as hard bound
    return (None, None)


def _df_to_tracks_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Frontend-friendly payload.
    """
    if df is None or df.empty:
        return []

    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        out.append({
            "track_id": str(r.get("track_id", "") or "").strip(),
            "title": str(r.get("track_name", "") or "").strip(),
            "artist": str(r.get("artists", "") or "").strip(),
            "genre": str(r.get("track_genre", "") or "").strip(),
            "bpm": None if pd.isna(r.get("bpm", None)) else int(r.get("bpm")),
            "match": None if pd.isna(r.get("match", None)) else float(r.get("match")),
        })
    return out


# ----------------------------
# Planner request model
# ----------------------------

@dataclass
class PlannerRequest:
    """
    Canonical form used by PlannerService. Built from a "discovery payload"
    coming from the frontend (Discovery -> Planner).
    """
    user_input: Dict[str, float]
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]]

    include_artists: List[str]
    include_genres: List[str]
    include_mode: str  # "must" | "prefer"
    allow_explicit: bool

    exclude_artists: List[str]
    exclude_genres: List[str]

    dontcare: Dict[str, bool]
    weight_overrides: Dict[str, float]

    # Planner safety knobs
    strict_semantics: bool
    lock_tempo: bool

    # Optional explicit pool/universe tuning
    pool_size: int


# ----------------------------
# Planner Service
# ----------------------------

class PlannerService:
    """
    Generates day-keyed playlists from a Discovery payload.
    Planner rules:
      - strict semantics by default (no genre widening / no semantic drift)
      - tempo can be locked as HARD if requested (recommended for Planner)
      - supports cooldown across days (Planner-only behavior)
    """

    def __init__(self, recommender: PlaylistRecommender):
        self.rec = recommender

    @classmethod
    def from_app(cls, app) -> "PlannerService":
        rec = app.config.get("RECOMMENDER", None)
        if rec is None:
            raise RuntimeError("Missing app.config['RECOMMENDER']")
        if not isinstance(rec, PlaylistRecommender):
            # allow subclasses, but sanity check the API surface
            if not hasattr(rec, "recommend_from_pool"):
                raise RuntimeError("RECOMMENDER is not a PlaylistRecommender-like object")
        return cls(rec)

    # ---------- payload parsing ----------

    def _parse_discovery_payload(self, payload: Dict[str, Any]) -> PlannerRequest:
        """
        Accepts flexible shapes coming from Discovery (or preset expansion).
        We only need a stable subset.
        """
        p = dict(payload or {})

        # --- support: payload could be nested under "discovery" ---
        if "discovery" in p and isinstance(p["discovery"], dict):
            # merge shallow (discovery keys win)
            base = {k: v for k, v in p.items() if k != "discovery"}
            base.update(p["discovery"])
            p = base

        # --- allow preset shortcut (legacy or convenience) ---
        preset = (p.get("preset") or p.get("preset_name") or "").strip()
        if preset and preset in PRESETS:
            # preset should behave like "discovery template"
            # discovery keys still override preset keys
            preset_payload = copy.deepcopy(PRESETS[preset])
            if isinstance(preset_payload, dict):
                preset_payload.update({k: v for k, v in p.items() if k not in {"preset", "preset_name"}})
                p = preset_payload

        # user_input: either explicit dict or derived from midpoints later
        user_input = {}
        if isinstance(p.get("user_input"), dict):
            for k, v in p["user_input"].items():
                try:
                    if v is None or v == "":
                        continue
                    user_input[str(k)] = float(v)
                except Exception:
                    continue

        # ranges: try multiple common keys
        ranges_raw = p.get("ranges") or p.get("feature_ranges") or p.get("filters") or {}
        ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        if isinstance(ranges_raw, dict):
            for f, v in ranges_raw.items():
                mn, mx = _clean_range_pair(v)
                if mn is None and mx is None:
                    continue
                ranges[str(f)] = (mn, mx)


        # --- ALIAS: UI uses "mood" but the system uses "valence" ---
        # If discovery form sends mood_min/mood_max (0..1), map them to valence_*.
        if "mood_min" in p and "valence_min" not in p:
            p["valence_min"] = p.get("mood_min")
        if "mood_max" in p and "valence_max" not in p:
            p["valence_max"] = p.get("mood_max")

        # dontcare alias too (dc_mood -> dc_valence)
        if "dc_mood" in p and "dc_valence" not in p:
            p["dc_valence"] = p.get("dc_mood")



        # --- NEW: fallback from flat form fields: *_min/_max ---
        # Discovery form currently sends keys like "danceability_min", "tempo_max", ...
        # Build canonical ranges from them if ranges{} not provided.
        if not ranges:
            FEATURES = [
                "danceability", "energy", "valence",
                "tempo", "loudness",
                "instrumentalness", "acousticness", "speechiness", "liveness",
            ]

            for f in FEATURES:
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
                        pass


        # includes / excludes
        include_artists = _as_list_str(p.get("include_artists") or p.get("artists_include") or p.get("artists") or [])
        include_genres = _as_list_str(p.get("include_genres") or p.get("genres_include") or p.get("genres") or [])
        exclude_artists = _as_list_str(p.get("exclude_artists") or p.get("artists_exclude") or [])
        exclude_genres = _as_list_str(p.get("exclude_genres") or p.get("genres_exclude") or [])

        include_mode = (p.get("include_mode") or "prefer").strip().lower()
        if include_mode not in {"must", "prefer"}:
            include_mode = "prefer"

        allow_explicit = str(p.get("allow_explicit", "")).strip().lower() in {"1", "true", "yes", "on"}


        dontcare = {}
        if isinstance(p.get("dontcare"), dict):
            for f, flag in p["dontcare"].items():
                dontcare[str(f)] = bool(flag)

        # --- NEW: dontcare from flat fields dc_* (e.g. dc_danceability="1") ---
        for f in ["danceability", "energy", "loudness", "valence", "tempo"]:
            kdc = f"dc_{f}"
            if kdc in p and f not in dontcare:
                dontcare[f] = str(p.get(kdc)).strip() in {"1", "true", "True", "yes", "on"}


        weight_overrides = {}
        if isinstance(p.get("weight_overrides"), dict):
            for f, w in p["weight_overrides"].items():
                try:
                    weight_overrides[str(f)] = float(w)
                except Exception:
                    continue

        # planner-specific defaults:
        # strict_semantics: in Planner sempre True (no widening)
        # lock_tempo: se il payload ha tempo range, la trattiamo come HARD
        strict_semantics = bool(p.get("strict_semantics", True))
        lock_tempo = bool(p.get("lock_tempo", True))
        if "tempo" not in ranges:
            # se non hai chiesto un range tempo, lock_tempo è irrilevante
            lock_tempo = False

        # pool size: serve a rendere stabile la selezione tra giorni, prima dei constraints
        pool_size = int(p.get("pool_size", 10000) or 10000)
        pool_size = max(2000, min(pool_size, 50000))

        return PlannerRequest(
            user_input=user_input,
            ranges=ranges,
            include_artists=include_artists,
            include_genres=include_genres,
            include_mode=include_mode,
            allow_explicit=allow_explicit,
            exclude_artists=exclude_artists,
            exclude_genres=exclude_genres,
            dontcare=dontcare,
            weight_overrides=weight_overrides,
            strict_semantics=strict_semantics,
            lock_tempo=lock_tempo,
            pool_size=pool_size,
        )

    # ---------- generation core ----------

    def _build_day_pool(
        self,
        req: PlannerRequest,
        *,
        random_state: int,
        exclude_track_ids: Set[str],
    ) -> pd.DataFrame:
        """
        Planner approach:
          1) build a stable universe from includes/excludes
          2) build a pool with strict semantics + (optional) hard tempo lock
          3) recommend_from_pool inside that pool (so includes "prefer/must" still works)
        """
        universe_idx = self.rec.build_universe_indices(
            include_artists=req.include_artists,
            include_genres=req.include_genres,
            exclude_artists=req.exclude_artists,
            exclude_genres=req.exclude_genres,
        )

        # Build a wide-ish pool in the relevant universe, deterministic by random_state
        # lock_tempo makes tempo range HARD (no expansion)
        pool = self.rec.build_pool(
            user_input=req.user_input,
            universe_idx=universe_idx,
            pool_size=req.pool_size,
            allow_explicit=req.allow_explicit,
            exclude_track_ids=exclude_track_ids,
            shuffle_within_top=True,
            random_state=int(random_state),
            dontcare=req.dontcare,
            weight_overrides=req.weight_overrides,
            ranges=req.ranges,
            lock_tempo=req.lock_tempo,
        )

        return pool

    def _recommend_from_day_pool(
        self,
        req: PlannerRequest,
        pool_df: pd.DataFrame,
        *,
        k: int,
        max_per_artist: int,
        random_state: int,
        exclude_track_ids: Set[str],
    ) -> pd.DataFrame:
        if pool_df is None or pool_df.empty:
            return pd.DataFrame()

        if "_row_idx" not in pool_df.columns:
            return pd.DataFrame()

        pool_idx = pool_df["_row_idx"].to_numpy()

        # Planner should NOT expand semantics: we enforce strict_semantics + lock_tempo upstream in pool.
        # recommend_from_pool will still honor include_mode and exclusions again.
        out = self.rec.recommend_from_pool(
            user_input=req.user_input,
            pool_idx=pool_idx,
            k=int(k),
            max_per_artist=int(max_per_artist),
            exclude_track_ids=set(exclude_track_ids),
            allow_explicit=req.allow_explicit,
            shuffle_within_top=True,
            random_state=int(random_state),
            weight_overrides=req.weight_overrides,
            dontcare=req.dontcare,
            include_artists=req.include_artists,
            include_genres=req.include_genres,
            include_mode=req.include_mode,
            exclude_artists=req.exclude_artists,
            exclude_genres=req.exclude_genres,
        )
        return out

    # ---------- public API ----------

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

        cooldown_days:
          - excludes tracks used in the previous N days (Planner behavior only)
        Determinism:
          - each day uses a stable random_state derived from (slot_id, dayISO, seed)
        """
        exclude_track_ids_global = set(exclude_track_ids_global or set())
        day_isos = [str(d).strip() for d in (day_isos or []) if str(d).strip()]
        day_isos = sorted(set(day_isos))

        req = self._parse_discovery_payload(discovery_payload)

        playlists_by_day: Dict[str, List[Dict[str, Any]]] = {}
        report: Dict[str, Any] = {
            "k": int(k),
            "max_per_artist": int(max_per_artist),
            "cooldown_days": int(cooldown_days),
            "strict_semantics": bool(req.strict_semantics),
            "lock_tempo": bool(req.lock_tempo),
            "days": day_isos,
        }

        # store track usage per day for cooldown
        used_by_day: Dict[str, Set[str]] = {}

        for day_iso in day_isos:
            # cooldown: union of previous N days already generated in THIS call
            ex = set(exclude_track_ids_global)

            if cooldown_days and cooldown_days > 0:
                d0 = _dayiso_to_date(day_iso)
                for j in range(1, int(cooldown_days) + 1):
                    prev = (d0 - timedelta(days=j)).isoformat()
                    ex |= used_by_day.get(prev, set())

            # deterministic per-day seed
            rs = _stable_int_seed(str(seed), slot_id, day_iso)

            pool = self._build_day_pool(req, random_state=rs, exclude_track_ids=ex)
            out_df = self._recommend_from_day_pool(
                req,
                pool,
                k=int(k),
                max_per_artist=int(max_per_artist),
                random_state=rs,
                exclude_track_ids=ex,
            )

            tracks = _df_to_tracks_payload(out_df)
            playlists_by_day[day_iso] = tracks

            # update used set for cooldown
            used = set()
            for t in tracks:
                tid = str(t.get("track_id", "") or "").strip()
                if tid:
                    used.add(tid)
            used_by_day[day_iso] = used

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
        """
        Legacy helper: treat preset as a discovery payload template.
        """
        preset_name = (preset_name or "").strip()
        if preset_name not in PRESETS:
            return {
                "playlistsByDay": {},
                "report": {"error": "invalid_preset", "preset": preset_name},
            }

        payload = copy.deepcopy(PRESETS[preset_name])
        if not isinstance(payload, dict):
            payload = {"preset": preset_name}

        # mark preset name so FE can still display it if needed
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

    # ------------------------------------------------------------------
    # NEW: one-shot plan preparation (Discovery -> Planner)
    # ------------------------------------------------------------------
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
        Builds a complete "plan" for the Planner UI in one shot:
          - defines a 2-week window (14 columns) starting on Monday
          - computes all day occurrences from (weeks + weekdays)
          - generates playlistsByDay for ALL occurrences immediately
          - builds a cell-grid assignment (rows x 14) for the time range

        Returns a JSON-serializable dict:
          {
            startDateISO, rows, cols,
            grid: [[None|{ruleId}], ...],
            rules: { ruleId: { ... , playlistsByDay } }
          }
        """

        # ---------- calendar constants (must match planner.js) ----------
        START_DAY = "10:00"
        END_DAY = "24:00"
        STEP_MIN = 30
        COLS = 14  # 2 weeks

        def _time_to_min(t: str) -> int:
            hh, mm = t.split(":")
            return int(hh) * 60 + int(mm)

        def _clamp_step(m: int) -> int:
            return int(round(m / STEP_MIN) * STEP_MIN)

        def _monday_of(d: date) -> date:
            # Python: Monday=0 ... Sunday=6
            return d - timedelta(days=d.weekday())

        # window start (Monday)
        if window_start_iso:
            try:
                window_start = _monday_of(_dayiso_to_date(window_start_iso))
            except Exception:
                window_start = _monday_of(date.today())
        else:
            window_start = _monday_of(date.today())

        startDateISO = window_start.isoformat()

        # rows in day
        start_min = _time_to_min(START_DAY)
        end_min = _time_to_min(END_DAY)
        rows = max(0, int((end_min - start_min) / STEP_MIN))

        # ---------- rule fields ----------
        name = str(rule.get("name") or "Slot").strip() or "Slot"
        color = str(rule.get("color") or "#FFD403").strip() or "#FFD403"
        slot_start = str(rule.get("start") or "10:00").strip() or "10:00"
        slot_end = str(rule.get("end") or "11:00").strip() or "11:00"

        weeks = int(rule.get("weeks", 2) or 2)
        weeks = max(1, min(weeks, 8))

        weekdays = rule.get("weekdays") or [1, 2, 3, 4, 5]  # JS getDay(): Mon=1..Sun=0
        if not isinstance(weekdays, list):
            weekdays = [1, 2, 3, 4, 5]
        # normalize to ints 0..6
        wds: List[int] = []
        for x in weekdays:
            try:
                v = int(x)
                if v in (0, 1, 2, 3, 4, 5, 6):
                    wds.append(v)
            except Exception:
                pass
        if not wds:
            wds = [1, 2, 3, 4, 5]
        wds = sorted(set(wds))

        # ---------- compute occurrences (day_isos) ----------
        # columns 0..13 map to dates window_start + c
        day_isos: List[str] = []
        for c in range(COLS):
            d = window_start + timedelta(days=c)
            week_index = (c // 7) + 1  # 1..2
            if week_index > weeks:
                continue
            # convert Python weekday (Mon=0..Sun=6) to JS getDay() (Sun=0..Sat=6)
            js_day = (d.weekday() + 1) % 7
            if js_day not in set(wds):
                continue
            day_isos.append(d.isoformat())

        # ---------- stable rule id ----------
        # stable across refreshes: depends on discovery + rule time/weekdays/weeks
        disc_fingerprint = hashlib.sha256(
            (repr(sorted(discovery_payload.items()))).encode("utf-8", errors="ignore")
        ).hexdigest()[:10]

        rule_id = f"r_{slot_start.replace(':','')}_{slot_end.replace(':','')}_w{weeks}_d{''.join(map(str,wds))}_{disc_fingerprint}"

        # ---------- generate ALL playlists now (this is your requirement) ----------
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

        # ---------- build grid assignment ----------
        grid: List[List[Optional[Dict[str, str]]]] = [
            [None for _ in range(COLS)] for _ in range(rows)
        ]

        a = _clamp_step(_time_to_min(slot_start))
        b = _clamp_step(_time_to_min(slot_end))
        a = max(start_min, min(a, end_min))
        b = max(start_min, min(b, end_min))
        if b > a:
            r0 = int((a - start_min) / STEP_MIN)
            r1 = int((b - start_min) / STEP_MIN) - 1

            day_set = set(day_isos)
            for c in range(COLS):
                d = (window_start + timedelta(days=c)).isoformat()
                if d not in day_set:
                    continue
                for r in range(r0, r1 + 1):
                    if 0 <= r < rows:
                        grid[r][c] = {"ruleId": rule_id}

        rules = {
            rule_id: {
                "id": rule_id,
                "name": name,
                "color": color,
                "start": slot_start,
                "end": slot_end,
                "weeks": weeks,
                "weekdays": wds,
                "k": int(k),
                "max_per_artist": int(max_per_artist),
                "cooldown_days": int(cooldown_days),
                "discovery": discovery_payload,
                "playlistsByDay": playlistsByDay,
                "report": report,
            }
        }

        return {
            "startDateISO": startDateISO,
            "slots": {
                rule_id: {
                    "id": rule_id,
                    "name": name,
                    "color": color,
                    "start": slot_start,
                    "end": slot_end,
                    "weeks": weeks,
                    "weekdays": wds,
                    "k": int(k),
                    "max_per_artist": int(max_per_artist),
                    "cooldown_days": int(cooldown_days),
                    "discovery": discovery_payload,
                    "playlistsByDay": playlistsByDay,
                }
            },
            "report": report,
}


