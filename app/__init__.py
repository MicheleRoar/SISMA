from flask import Flask
from pathlib import Path

from .services.data_store import DataStore
from .services.recommender import PlaylistRecommender, RecommenderConfig
from .services.planner_service import PlannerService


def create_app():
    base_dir = Path(__file__).resolve().parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    app = Flask(
        __name__,
        template_folder=str(templates_dir),
        static_folder=str(static_dir),
    )

    # Blueprints
    from app.blueprints.discovery import bp as discovery_bp
    from app.blueprints.spotify import bp as spotify_bp
    from app.blueprints.planner import bp as planner_bp

    app.register_blueprint(discovery_bp)
    app.register_blueprint(spotify_bp)
    app.register_blueprint(planner_bp)

    # -------------------------
    # Core services (singletons)
    # -------------------------
    store = DataStore(
        tracks_csv_path="data/archive/tracks.csv",
        artists_csv_path="data/archive/artists.csv",
        ui_genre_min_count=50,
        ui_genre_top_k=200,
    )

    config = RecommenderConfig(k=50, max_per_artist=2)
    recommender = PlaylistRecommender(store, config=config)

    planner_service = PlannerService(recommender)

    app.config["DATASTORE"] = store
    app.config["RECOMMENDER"] = recommender
    app.config["PLANNER_SERVICE"] = planner_service

    return app
