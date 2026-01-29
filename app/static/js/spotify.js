// static/js/spotify.js
(() => {
  // === CONFIG ===
  const CLIENT_ID = "5c7386061e1b4d46ac74c69d70cdee21";
  const REDIRECT_URI = "http://127.0.0.1:5001/spotify/callback";
  //const REDIRECT_URI = `${window.location.origin}/spotify/callback`;
  const SCOPES = ["playlist-modify-private"].join(" "); // oppure playlist-modify-public

  const AUTH_URL = "https://accounts.spotify.com/authorize";
  const TOKEN_URL = "https://accounts.spotify.com/api/token";
  const API_BASE = "https://api.spotify.com/v1";

  function msg(html) {
    const el = document.getElementById("spotify_msg");
    if (!el) return;
    el.style.display = "block";
    el.innerHTML = html;
  }

  function getSelectedTrackIdsFromTable() {
  return Array.from(document.querySelectorAll(".include-track:checked"))
    .map(el => String(el.dataset.trackId || "").trim())
    .filter(Boolean);
}

  function saveSelectionState() {
    // salva dove tornare + quali checkbox erano selezionati
    sessionStorage.setItem("sisma_return_url", window.location.href);
    const selected = getSelectedTrackIdsFromTable();
    sessionStorage.setItem("sisma_selected_tracks", JSON.stringify(selected));
  }

  function restoreSelectionState() {
    const raw = sessionStorage.getItem("sisma_selected_tracks");
    if (!raw) return;

    let selected = [];
    try { selected = JSON.parse(raw) || []; } catch (e) { selected = []; }

    if (!Array.isArray(selected) || selected.length === 0) return;

    const selectedSet = new Set(selected);

    document.querySelectorAll(".include-track").forEach((el) => {
      const tid = String(el.dataset.trackId || "").trim();
      if (!tid) return;
      el.checked = selectedSet.has(tid);
    });
  }


  function base64urlEncode(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer);
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  async function sha256(plain) {
    return crypto.subtle.digest("SHA-256", new TextEncoder().encode(plain));
  }

  function randomString(len = 64) {
    const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    const bytes = crypto.getRandomValues(new Uint8Array(len));
    return Array.from(bytes, b => possible[b % possible.length]).join("");
  }

  function setToken(accessToken, expiresInSec) {
    localStorage.setItem("spotify_access_token", accessToken);
    localStorage.setItem("spotify_expires_at", String(Date.now() + expiresInSec * 1000));
  }

  function getToken() {
    const t = localStorage.getItem("spotify_access_token");
    const exp = Number(localStorage.getItem("spotify_expires_at") || "0");
    if (!t || Date.now() > exp) return null;
    return t;
  }

  async function spotifyFetch(path, { method = "GET", body } = {}) {
    const token = getToken();
    if (!token) throw new Error("Not connected to Spotify. Click 'Connect Spotify' first.");
    const res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`Spotify API ${res.status}: ${txt}`);
    }
    return res.json();
  }

  function toTrackUri(trackId) {
    if (trackId.startsWith("spotify:track:")) return trackId;
    return `spotify:track:${trackId}`;
  }

  // === AUTH: start ===
  async function spotifyLogin() {
    const codeVerifier = randomString(64);
    const codeChallenge = base64urlEncode(await sha256(codeVerifier));
    sessionStorage.setItem("spotify_code_verifier", codeVerifier);

    const params = new URLSearchParams({
      response_type: "code",
      client_id: CLIENT_ID,
      scope: SCOPES,
      redirect_uri: REDIRECT_URI,
      code_challenge_method: "S256",
      code_challenge: codeChallenge,
    });

    console.log("REDIRECT_URI being sent:", REDIRECT_URI);
    window.location.href = `${AUTH_URL}?${params.toString()}`;
  }

  // === AUTH: callback ===
  async function spotifyHandleCallback() {
    const url = new URL(window.location.href);
    const code = url.searchParams.get("code");
    const error = url.searchParams.get("error");
    if (error) throw new Error(`Spotify auth error: ${error}`);
    if (!code) return; // niente da fare

    const codeVerifier = sessionStorage.getItem("spotify_code_verifier");
    if (!codeVerifier) throw new Error("Missing PKCE code_verifier (session). Retry Connect Spotify.");

    const body = new URLSearchParams({
      client_id: CLIENT_ID,
      grant_type: "authorization_code",
      code,
      redirect_uri: REDIRECT_URI,
      code_verifier: codeVerifier,
    });

    const res = await fetch(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`Token exchange failed: ${res.status} ${txt}`);
    }

    const data = await res.json(); // {access_token, expires_in, ...}
    setToken(data.access_token, data.expires_in);

    return data.access_token;
  }

  // === CORE: create playlist from current results ===
  async function getGeneratedTracksFromCurrentForm() {
    // prende gli stessi parametri del form (quelli che hai usato per generare la tabella)
    const form = document.getElementById("playlist_form");
    if (!form) throw new Error("Form not found.");
    const params = new URLSearchParams(new FormData(form));
    params.set("format", "json");

    const baseUrl = form.getAttribute("action").split("#")[0];
    const url = `${baseUrl}?${params.toString()}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Cannot fetch generated tracks JSON (${res.status}).`);
    const rows = await res.json();

    // ci aspettiamo track_id in ogni record
    const ids = rows.map(r => String(r.track_id || "").trim()).filter(Boolean);
    if (ids.length === 0) {
      throw new Error("No track_id found in JSON. Ensure playlist_df includes 'track_id'.");
    }
    return ids;
  }

  async function createSpotifyPlaylistWithTracks(trackIds) {
    const me = await spotifyFetch("/me");
    const date = new Date();
    const name = `SISMA • ${date.toISOString().slice(0, 10)} • 50 tracks`;
    const description = "Generated by SISMA Music Discovering Tool";

    const playlist = await spotifyFetch(`/users/${me.id}/playlists`, {
      method: "POST",
      body: { name, description, public: false },
    });

    // add tracks (max 100 per call)
    const uris = trackIds.map(toTrackUri);
    await spotifyFetch(`/playlists/${playlist.id}/tracks`, {
      method: "POST",
      body: { uris },
    });

    return playlist.external_urls?.spotify || `https://open.spotify.com/playlist/${playlist.id}`;
  }

  // === UI wiring ===
  function updateButtons() {
    const addBtn = document.getElementById("btn_spotify_add");
    if (!addBtn) return;
    addBtn.disabled = !getToken();
  }

  async function onAddToSpotify() {
    msg("Creating playlist on Spotify…");
    const trackIds = await getGeneratedTracksFromCurrentForm();
    const url = await createSpotifyPlaylistWithTracks(trackIds);
    msg(`Playlist created: <a href="${url}" target="_blank" rel="noopener">Open in Spotify</a>`);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const connectBtn = document.getElementById("btn_spotify_connect");
    const addBtn = document.getElementById("btn_spotify_add");

    if (connectBtn) {
      connectBtn.addEventListener("click", async () => {
        try {
          saveSelectionState();   
          await spotifyLogin();  
        } catch (e) {
          msg(`${e?.message || String(e)}`);
        }
      });
    }

    if (addBtn) {
      addBtn.addEventListener("click", async () => {
        try {
          await onAddToSpotify();
        } catch (e) {
          msg(`${e?.message || String(e)}`);
        }
      });
    }

    updateButtons();
  });

  // export for callback page
  window.spotifyHandleCallback = spotifyHandleCallback;
})();
