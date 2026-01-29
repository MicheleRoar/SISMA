from flask import Blueprint, render_template

bp = Blueprint("spotify", __name__)

@bp.route("/spotify/callback", methods=["GET"])
def spotify_callback():
    return render_template("spotify_callback.html")
