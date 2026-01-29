from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify

from app.services.data_store import DataStore
from app.services.presets import PRESETS
from app.services.planner_service import PlannerService

bp = Blueprint("planner", __name__, url_prefix="/planner")

_store: DataStore | None = None
_service: PlannerService | None = None


def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
    return _store


def get_service() -> PlannerService:
    global _service
    if _service is None:
        _service = PlannerService.from_store(get_store())
    return _service


@bp.route("/")
def index():
    presets = list(PRESETS.keys())
    return render_template("planner/planner.html", presets=presets)


@bp.post("/api/generate")
def api_generate():
    try:
        payload = request.get_json(force=True) or {}

        preset_name = (payload.get("preset") or "").strip()
        if preset_name not in PRESETS:
            return jsonify({"ok": False, "error": "Preset non valido"}), 400

        k = int(payload.get("k", 50))

        # seed per randomizzazione (fondamentale per playlist diverse)
        seed = payload.get("seed", 42)
        try:
            seed = int(seed)
        except Exception:
            seed = 42

        exclude_track_ids = payload.get("exclude_track_ids") or []
        if not isinstance(exclude_track_ids, list):
            exclude_track_ids = []

        _, tracks = get_service().generate_playlist_for_preset(
            preset_name,
            k=k,
            exclude_track_ids={str(x) for x in exclude_track_ids if x},
            random_state=seed,
            shuffle_within_top=True,
        )

        out = []
        for t in tracks:
            out.append({
                "title": t.get("track_name", ""),
                "artist": t.get("artists", ""),
                "genre": t.get("track_genre", ""),
                "bpm": t.get("tempo", None),
                "track_id": t.get("track_id", ""),
            })

        return jsonify({"ok": True, "tracks": out})

    except Exception as e:
        return jsonify(
            {"ok": False, "error": f"{type(e).__name__}: {e}"},
            500,
        )