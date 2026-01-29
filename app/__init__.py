from flask import Flask
from pathlib import Path

from .services.data_store import DataStore
from .services.recommender import PlaylistRecommender, RecommenderConfig


def create_app():
    base_dir = Path(__file__).resolve().parent  # .../SISMA/app
    templates_dir = base_dir / "templates"      # .../SISMA/app/templates
    static_dir = base_dir / "static"            # .../SISMA/app/static

    app = Flask(
        __name__,
        template_folder=str(templates_dir),
        static_folder=str(static_dir),
    )

    # Blueprints
    from app.blueprints.discovery import bp as discovery_bp
    from app.blueprints.spotify import bp as spotify_bp
    from app.blueprints.planner import bp as planner_bp

    app.register_blueprint(discovery_bp)   # /
    app.register_blueprint(spotify_bp)     # /spotify/callback
    app.register_blueprint(planner_bp)     # /planner (per ora)

    store = DataStore(
        tracks_csv_path="data/archive/tracks.csv",
        artists_csv_path="data/archive/artists.csv",
        ui_genre_min_count=50,
        ui_genre_top_k=200,
    )

    config = RecommenderConfig(k=50, max_per_artist=2)
    recommender = PlaylistRecommender(store, config=config)

    app.config["DATASTORE"] = store
    app.config["RECOMMENDER"] = recommender

    return app
