# app/blueprints/planner/routes.py
from __future__ import annotations

from typing import Any, Dict, List, Set

from flask import Blueprint, current_app, jsonify, render_template, request

from app.services.presets import PRESETS
from app.services.planner_service import PlannerService

bp = Blueprint("planner", __name__, url_prefix="/planner")


def _svc() -> PlannerService:
    return PlannerService.from_app(current_app)


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, tuple):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []


def _as_set_str(x: Any) -> Set[str]:
    return {s for s in _as_list_str(x) if s}


@bp.get("/")
def index():
    # Presets solo come fallback/compat (es. se arrivi al planner senza draft)
    presets = list(PRESETS.keys())
    return render_template("planner/planner.html", presets=presets)


# -------------------------------------------------------------------
# Legacy: single playlist (keeps older planner.js working if needed)
# -------------------------------------------------------------------
@bp.post("/api/generate")
def api_generate():
    """
    Back-compat endpoint:
      POST { preset, k, seed, exclude_track_ids, day_iso? }
    Returns:
      { ok: true, tracks: [...] }
    """
    try:
        payload = request.get_json(force=True) or {}

        preset_name = (payload.get("preset") or "").strip()
        if preset_name not in PRESETS:
            return jsonify({"ok": False, "error": "Preset non valido"}), 400

        k = int(payload.get("k", 50) or 50)

        seed = payload.get("seed", 42)
        try:
            seed = int(seed)
        except Exception:
            seed = 42

        exclude_track_ids = payload.get("exclude_track_ids") or []
        if not isinstance(exclude_track_ids, list):
            exclude_track_ids = []

        day_iso = str(payload.get("day_iso") or "1970-01-01").strip() or "1970-01-01"

        res = _svc().generate_for_preset_occurrences(
            preset_name,
            day_isos=[day_iso],
            k=k,
            max_per_artist=int(payload.get("max_per_artist", 2) or 2),
            cooldown_days=int(payload.get("cooldown_days", 0) or 0),
            exclude_track_ids_global={str(x).strip() for x in exclude_track_ids if str(x).strip()},
            seed=seed,
            slot_id=str(payload.get("slot_id") or "legacy").strip() or "legacy",
        )

        tracks = (res.get("playlistsByDay") or {}).get(day_iso, []) or []
        return jsonify({"ok": True, "tracks": tracks, "report": res.get("report", {})})

    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


# -------------------------------------------------------------------
# Core: batch generation for multiple day occurrences (Planner v2)
# -------------------------------------------------------------------
@bp.post("/api/generate_batch")
def api_generate_batch():
    """
    Main endpoint used by new planner.js:
      POST {
        slot_id?: string,
        discovery?: {...},   # preferred
        preset?: string,     # fallback
        day_isos: ["YYYY-MM-DD", ...],
        k?: int,
        max_per_artist?: int,
        cooldown_days?: int,
        seed?: int,
        exclude_track_ids?: [track_id...]
      }

    Returns:
      { ok:true, playlistsByDay:{day:[tracks...]}, report:{...} }
    """
    try:
        payload = request.get_json(force=True) or {}

        # days
        day_isos = payload.get("day_isos") or payload.get("days") or []
        if not isinstance(day_isos, list):
            return jsonify({"ok": False, "error": "day_isos deve essere una lista"}), 400
        day_isos = [str(d).strip() for d in day_isos if str(d).strip()]
        if not day_isos:
            return jsonify({"ok": True, "playlistsByDay": {}, "report": {"reason": "no_days"}})

        # knobs
        k = int(payload.get("k", 50) or 50)
        max_per_artist = int(payload.get("max_per_artist", 2) or 2)
        cooldown_days = int(payload.get("cooldown_days", 2) or 2)

        seed = payload.get("seed", 42)
        try:
            seed = int(seed)
        except Exception:
            seed = 42

        slot_id = str(payload.get("slot_id") or "slot").strip() or "slot"

        # exclusions
        exclude_track_ids = payload.get("exclude_track_ids") or []
        if not isinstance(exclude_track_ids, list):
            exclude_track_ids = []
        exclude_track_ids_global = {str(x).strip() for x in exclude_track_ids if str(x).strip()}

        # discovery payload preferred
        discovery = payload.get("discovery") or None
        if discovery is not None and not isinstance(discovery, dict):
            return jsonify({"ok": False, "error": "discovery deve essere un oggetto"}), 400

        svc = _svc()

        if discovery:
            res = svc.generate_for_discovery_payload(
                discovery_payload=discovery,
                day_isos=day_isos,
                k=k,
                max_per_artist=max_per_artist,
                cooldown_days=cooldown_days,
                exclude_track_ids_global=exclude_track_ids_global,
                seed=seed,
                slot_id=slot_id,
            )
        else:
            # fallback preset (legacy)
            preset_name = (payload.get("preset") or "").strip()
            if preset_name not in PRESETS:
                return jsonify({"ok": False, "error": "Manca discovery oppure preset valido"}), 400

            res = svc.generate_for_preset_occurrences(
                preset_name,
                day_isos=day_isos,
                k=k,
                max_per_artist=max_per_artist,
                cooldown_days=cooldown_days,
                exclude_track_ids_global=exclude_track_ids_global,
                seed=seed,
                slot_id=slot_id,
            )

        return jsonify({
            "ok": True,
            "playlistsByDay": res.get("playlistsByDay", {}),
            "report": res.get("report", {}),
        })

    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500



# -------------------------------------------------------------------
# NEW: prepare full plan in one shot (Discovery -> Planner)
# -------------------------------------------------------------------
@bp.post("/api/prepare_plan")
def api_prepare_plan():
    """
    One-shot: Discovery -> Planner
    POST {
      discovery: {...},          # raw discovery payload from form
      rule: {name,color,start,end,weeks,weekdays},
      k?: int,
      max_per_artist?: int,
      cooldown_days?: int,
      window_start_iso?: "YYYY-MM-DD"   # optional
    }

    Returns:
      { ok:true, plan:{...} }
    """
    try:
        payload = request.get_json(force=True) or {}

        discovery = payload.get("discovery") or {}
        if not isinstance(discovery, dict):
            return jsonify({"ok": False, "error": "discovery deve essere un oggetto"}), 400

        rule = payload.get("rule") or {}
        if not isinstance(rule, dict):
            return jsonify({"ok": False, "error": "rule deve essere un oggetto"}), 400

        k = int(payload.get("k", 50) or 50)
        max_per_artist = int(payload.get("max_per_artist", 2) or 2)
        cooldown_days = int(payload.get("cooldown_days", 2) or 2)
        window_start_iso = payload.get("window_start_iso")

        svc = _svc()

        # IMPORTANT: requires PlannerService.prepare_plan(...)
        plan = svc.prepare_plan(
            discovery_payload=discovery,
            rule=rule,
            k=k,
            max_per_artist=max_per_artist,
            cooldown_days=cooldown_days,
            window_start_iso=str(window_start_iso).strip() if window_start_iso else None,
        )

        return jsonify({"ok": True, "plan": plan})

    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500




# -------------------------------------------------------------------
# Future: commit plan to Spotify + export JSON
# -------------------------------------------------------------------
@bp.post("/api/commit_plan")
def api_commit_plan():
    """
    Stub endpoint (to implement next):
      POST {
        plan: [
          {
            slot_id, day_iso, start, end, name, color,
            tracks: [ {track_id, ...} ]
          }, ...
        ]
      }

    Should:
      - create one Spotify playlist per entry
      - add tracks
      - return export JSON with playlist_url, day, time, name, slot_id
    """
    try:
        payload = request.get_json(force=True) or {}
        plan = payload.get("plan") or []
        if not isinstance(plan, list):
            return jsonify({"ok": False, "error": "plan deve essere una lista"}), 400

        # For now: validate shape lightly and return a placeholder.
        # Implementation will go through your Spotify blueprint/token flow.
        export = []
        for item in plan:
            if not isinstance(item, dict):
                continue
            export.append({
                "slot_id": str(item.get("slot_id") or "").strip(),
                "day_iso": str(item.get("day_iso") or "").strip(),
                "start": str(item.get("start") or "").strip(),
                "end": str(item.get("end") or "").strip(),
                "name": str(item.get("name") or "").strip(),
                "color": str(item.get("color") or "").strip(),
                "playlist_url": None,  # to be filled when we actually create Spotify playlists
            })

        return jsonify({
            "ok": True,
            "message": "commit_plan non ancora implementato (stub).",
            "export": export,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
