# app/services/presets.py
# Preset derivati da Rules.xlsx — versione RANGE (min/max)
# + campi Planner (bucket include/exclude + allow_explicit + fallback)
#
# NOTE (Feb 2026):
# - Alcune feature (instrumentalness, acousticness, speechiness, liveness) possono essere "sparse" e
#   ridurre drasticamente il pool se usate come filtri HARD.
# - Per riempire sempre ~50 brani, le disattiviamo temporaneamente mettendo None.
# - Restano HARD: tempo, energy, danceability, valence, loudness (+ eventuali buckets).

PRESETS = {
    "Warm-up Daytime": dict(
        # --- ranges (HARD) ---
        tempo_min=70, tempo_max=90,
        energy_min=0.45, energy_max=0.65,
        danceability_min=0.35, danceability_max=0.55,
        valence_min=0.65, valence_max=0.85,
        loudness_min=-16.0, loudness_max=-10.0,

        # --- temporarily disabled (avoid sparse hard filters) ---
        acousticness_min=None, acousticness_max=None,
        instrumentalness_min=None, instrumentalness_max=None,
        speechiness_min=None, speechiness_max=None,
        liveness_min=None, liveness_max=None,

        # --- planner controls ---
        allow_explicit=False,
        fallback=True,

        # --- buckets (empty = global universe) ---
        include_artists=[],
        include_genres=[],
        exclude_artists=[],
        exclude_genres=[],

        artist_weights=None,
        genre_weights=None,
    ),

    "Lunch Orders (Variety)": dict(
        tempo_min=90, tempo_max=110,
        energy_min=0.70, energy_max=0.90,
        danceability_min=0.60, danceability_max=0.80,
        valence_min=0.70, valence_max=0.90,
        loudness_min=-12.0, loudness_max=-6.0,

        acousticness_min=None, acousticness_max=None,
        instrumentalness_min=None, instrumentalness_max=None,
        speechiness_min=None, speechiness_max=None,
        liveness_min=None, liveness_max=None,

        allow_explicit=False,
        fallback=True,

        include_artists=[],
        include_genres=[],
        exclude_artists=[],
        exclude_genres=[],

        artist_weights=None,
        genre_weights=None,
    ),

    "Lunch Consumption (Slow Down)": dict(
        tempo_min=60, tempo_max=80,
        energy_min=0.35, energy_max=0.55,
        danceability_min=0.25, danceability_max=0.45,
        valence_min=0.55, valence_max=0.75,
        loudness_min=-16.0, loudness_max=-10.0,

        acousticness_min=None, acousticness_max=None,
        instrumentalness_min=None, instrumentalness_max=None,
        speechiness_min=None, speechiness_max=None,
        liveness_min=None, liveness_max=None,

        allow_explicit=False,
        fallback=True,

        include_artists=[],
        include_genres=[],
        exclude_artists=[],
        exclude_genres=[],

        artist_weights=None,
        genre_weights=None,
    ),

    "Afternoon Walk-in (Upsell)": dict(
        tempo_min=70, tempo_max=90,
        energy_min=0.45, energy_max=0.65,
        danceability_min=0.35, danceability_max=0.55,
        valence_min=0.65, valence_max=0.85,
        loudness_min=-16.0, loudness_max=-10.0,

        acousticness_min=None, acousticness_max=None,
        instrumentalness_min=None, instrumentalness_max=None,
        speechiness_min=None, speechiness_max=None,
        liveness_min=None, liveness_max=None,

        allow_explicit=False,
        fallback=True,

        include_artists=[],
        include_genres=[],
        exclude_artists=[],
        exclude_genres=[],

        artist_weights=None,
        genre_weights=None,
    ),

    "Aperitivo (High Spend Push)": dict(
        tempo_min=90, tempo_max=110,
        energy_min=0.70, energy_max=0.85,
        danceability_min=0.55, danceability_max=0.70,
        valence_min=0.75, valence_max=0.90,
        loudness_min=-14.0, loudness_max=-8.0,

        acousticness_min=None, acousticness_max=None,
        instrumentalness_min=None, instrumentalness_max=None,
        speechiness_min=None, speechiness_max=None,
        liveness_min=None, liveness_max=None,

        allow_explicit=False,
        fallback=True,

        include_artists=[],
        include_genres=[],
        exclude_artists=[],
        exclude_genres=[],

        artist_weights=None,
        genre_weights=None,
    ),

    "Dinner Orders (Efficiency)": dict(
        tempo_min=105, tempo_max=125,
        energy_min=0.70, energy_max=0.85,
        danceability_min=0.55, danceability_max=0.70,
        valence_min=0.65, valence_max=0.80,
        loudness_min=-12.0, loudness_max=-6.0,

        acousticness_min=None, acousticness_max=None,
        instrumentalness_min=None, instrumentalness_max=None,
        speechiness_min=None, speechiness_max=None,
        liveness_min=None, liveness_max=None,

        allow_explicit=False,
        fallback=True,

        include_artists=[],
        include_genres=[],
        exclude_artists=[],
        exclude_genres=[],

        artist_weights=None,
        genre_weights=None,
    ),

    "Dinner Consumption (Stability)": dict(
        tempo_min=80, tempo_max=100,
        energy_min=0.45, energy_max=0.60,
        danceability_min=0.35, danceability_max=0.55,
        valence_min=0.55, valence_max=0.75,
        loudness_min=-14.0, loudness_max=-8.0,

        acousticness_min=None, acousticness_max=None,
        instrumentalness_min=None, instrumentalness_max=None,
        speechiness_min=None, speechiness_max=None,
        liveness_min=None, liveness_max=None,

        allow_explicit=False,
        fallback=True,

        include_artists=[],
        include_genres=[],
        exclude_artists=[],
        exclude_genres=[],

        artist_weights=None,
        genre_weights=None,
    ),

    "Closing / Soft Exit": dict(
        tempo_min=55, tempo_max=75,
        energy_min=0.20, energy_max=0.35,
        danceability_min=0.15, danceability_max=0.30,
        valence_min=0.45, valence_max=0.65,
        loudness_min=-18.0, loudness_max=-12.0,

        acousticness_min=None, acousticness_max=None,
        instrumentalness_min=None, instrumentalness_max=None,
        speechiness_min=None, speechiness_max=None,
        liveness_min=None, liveness_max=None,

        allow_explicit=False,
        fallback=True,

        include_artists=[],
        include_genres=[],
        exclude_artists=[],
        exclude_genres=[],

        artist_weights=None,
        genre_weights=None,
    ),
}
