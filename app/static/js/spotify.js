// static/js/spotify.js
(() => {
  // === CONFIG ===
  const CLIENT_ID = "5c7386061e1b4d46ac74c69d70cdee21";
  const REDIRECT_URI = "http://127.0.0.1:5001/spotify/callback";
  const SCOPES = ["playlist-modify-private"].join(" ");

  const AUTH_URL = "https://accounts.spotify.com/authorize";
  const TOKEN_URL = "https://accounts.spotify.com/api/token";
  const API_BASE = "https://api.spotify.com/v1";

  // planner localStorage key
  const LS_PLAN_KEY = "sisma_planner_plan_v1";

  const WEEKDAY_EN = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

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
    sessionStorage.setItem("sisma_return_url", window.location.href);

    const selected = getSelectedTrackIdsFromTable();
    sessionStorage.setItem("sisma_selected_tracks", JSON.stringify(selected));

    const form = document.getElementById("playlist_form");
    if (form && document.querySelector(".song-table")) {
      const params = new URLSearchParams(new FormData(form));
      params.set("format", "json");

      const baseUrl = (form.getAttribute("action") || window.location.pathname).split("#")[0];
      const jsonUrl = `${baseUrl}?${params.toString()}`;

      sessionStorage.setItem("sisma_discovery_json_url", jsonUrl);
    }
  }

  function restoreSelectionState() {
    const raw = sessionStorage.getItem("sisma_selected_tracks");
    if (!raw) return;

    let selected = [];
    try { selected = JSON.parse(raw) || []; } catch { selected = []; }

    if (!Array.isArray(selected) || selected.length === 0) return;
    const selectedSet = new Set(selected);

    document.querySelectorAll(".include-track").forEach((el) => {
      const tid = String(el.dataset.trackId || "").trim();
      if (!tid) return;
      el.checked = selectedSet.has(tid);
    });
  }


  function restoreDiscoveryResultsIfNeeded() {
    const jsonUrl = sessionStorage.getItem("sisma_discovery_json_url");
    if (!jsonUrl) return;

    // Se la tabella esiste già, pulisci e basta
    if (document.querySelector(".song-table")) {
      sessionStorage.removeItem("sisma_discovery_json_url");
      return;
    }

    // Se non sei nel Discovery, non fare nulla
    const form = document.getElementById("playlist_form");
    if (!form) return;

    try {
      const htmlUrl = jsonUrl
        .replace(/([?&])format=json(&|$)/, "$1")
        .replace(/[?&]$/, "");

      // evita redirect identico alla pagina corrente
      const currentNoHash = window.location.href.split("#")[0];
      if (currentNoHash === htmlUrl) return;

      sessionStorage.removeItem("sisma_discovery_json_url");
      window.location.href = htmlUrl;
    } catch (e) {
      console.error("Failed to restore discovery results:", e);
    }
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


  function updateSpotifyStatusUI() {
    const token = getToken();

    const statusWrap = document.querySelector(".spotify-status");
    const dot = statusWrap ? statusWrap.querySelector(".dot") : null;
    const label = statusWrap ? statusWrap.querySelector(".muted") : null;

    const connectBtn = document.getElementById("btn_spotify_connect");
    const plannerBtn = document.getElementById("btnCommitSpotify");
    const addBtn = document.getElementById("btn_spotify_add");

    const connected = !!token;

    if (dot) {
      dot.classList.remove("dot-off", "dot-on");
      dot.classList.add(connected ? "dot-on" : "dot-off");
    }

    if (label) {
      label.textContent = connected
        ? "Spotify: connected"
        : "Spotify: not connected";
    }

    if (connectBtn) {
      connectBtn.disabled = connected;
      connectBtn.textContent = connected ? "Spotify Connected" : "Connect Spotify";
      connectBtn.setAttribute("aria-disabled", connected ? "true" : "false");
    }

    if (plannerBtn) plannerBtn.disabled = !connected;
    if (addBtn) addBtn.disabled = !connected;
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
      throw new Error(`Spotify API ${res.status} on ${method} ${path}: ${txt}`);
    }

    return res.json();
  }

  function toTrackUri(trackId) {
    if (!trackId) return null;
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

    window.location.href = `${AUTH_URL}?${params.toString()}`;
  }

  // === AUTH: callback ===
  async function spotifyHandleCallback() {
    const url = new URL(window.location.href);
    const code = url.searchParams.get("code");
    const error = url.searchParams.get("error");
    if (error) throw new Error(`Spotify auth error: ${error}`);
    if (!code) return;

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

    const data = await res.json();
    setToken(data.access_token, data.expires_in);

    const returnUrl = sessionStorage.getItem("sisma_return_url") || "/";
    sessionStorage.removeItem("spotify_code_verifier");
    sessionStorage.removeItem("sisma_return_url");

    window.location.href = returnUrl;
    return data.access_token;
  }

  // =========================
  // DISCOVERY
  // =========================
  async function getGeneratedTracksFromCurrentForm() {
    const form = document.getElementById("playlist_form");
    if (!form) throw new Error("Form not found.");

    const params = new URLSearchParams(new FormData(form));
    params.set("format", "json");

    const baseUrl = form.getAttribute("action").split("#")[0];
    const url = `${baseUrl}?${params.toString()}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Cannot fetch generated tracks JSON (${res.status}).`);

    const rows = await res.json();
    const ids = rows.map(r => String(r.track_id || "").trim()).filter(Boolean);

    if (ids.length === 0) {
      throw new Error("No track_id found in JSON. Ensure playlist_df includes 'track_id'.");
    }
    return ids;
  }

  async function createSpotifyPlaylistNamed(name, trackIds, description = "") {
    const me = await spotifyFetch("/me");

    const playlist = await spotifyFetch(`/users/${me.id}/playlists`, {
      method: "POST",
      body: {
        name,
        description: description || "Generated by SISMA",
        public: false
      },
    });

    const uris = trackIds.map(toTrackUri).filter(Boolean);

    // Spotify: max 100 uris per request
    for (let i = 0; i < uris.length; i += 100) {
      const chunk = uris.slice(i, i + 100);
      await spotifyFetch(`/playlists/${playlist.id}/tracks`, {
        method: "POST",
        body: { uris: chunk },
      });
    }

    return {
      id: playlist.id,
      url: playlist.external_urls?.spotify || `https://open.spotify.com/playlist/${playlist.id}`,
      name: playlist.name
    };
  }

  async function onAddToSpotify() {
    msg("Creating playlist on Spotify…");
    const trackIds = await getGeneratedTracksFromCurrentForm();

    const date = new Date();
    const name = `SISMA • ${date.toISOString().slice(0, 10)} • 50 tracks`;
    const description = "Generated by SISMA Music Discovering Tool";

    const out = await createSpotifyPlaylistNamed(name, trackIds, description);
    msg(`Playlist created: <a href="${out.url}" target="_blank" rel="noopener">Open in Spotify</a>`);
  }

  // =========================
  // PLANNER
  // =========================
  function pad2(n) {
    return String(n).padStart(2, "0");
  }

  function fmtLocalISODate(d) {
    return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
  }

  function parseISODate(s) {
    return new Date(`${s}T00:00:00`);
  }

  function normalizeWeekdays(wd) {
    const out = [];
    const seen = new Set();

    (wd || []).forEach(x => {
      const n = Number(x);
      if (Number.isFinite(n) && n >= 0 && n <= 6 && !seen.has(n)) {
        seen.add(n);
        out.push(n);
      }
    });

    return out.length ? out : [1, 2, 3, 4, 5];
  }

  function getEnabledPlaylist(slot, dayISO) {
    if (!slot || !slot.playlistsByDay) return [];
    const playlist = Array.isArray(slot.playlistsByDay[dayISO]) ? slot.playlistsByDay[dayISO] : [];
    return playlist.filter(t => t && t.enabled !== false);
  }

  function getTrackId(t) {
    return String(
      t?.track_id ??
      t?.id ??
      t?.spotify_id ??
      ""
    ).trim();
  }

  function buildPlannerOccurrences() {
    const raw = localStorage.getItem(LS_PLAN_KEY);
    if (!raw) throw new Error("Planner vuoto in localStorage.");

    const obj = JSON.parse(raw);
    const startDateISO = obj?.startDateISO;
    const slots = obj?.slots || {};

    if (!startDateISO) throw new Error("startDateISO mancante nel planner.");
    const startDate = parseISODate(startDateISO);

    const occurrences = [];

    for (const slotId of Object.keys(slots)) {
      const slot = slots[slotId];
      if (!slot) continue;

      const weekdays = new Set(normalizeWeekdays(slot.weekdays));
      const weeks = Math.max(1, Number(slot.weeks) || 1);

      for (let c = 0; c < 14; c++) {
        const weekIndex = Math.floor(c / 7) + 1;
        if (weekIndex > weeks) continue;

        const d = new Date(startDate);
        d.setDate(startDate.getDate() + c);

        if (!weekdays.has(d.getDay())) continue;

        const dayISO = fmtLocalISODate(d);
        const tracks = getEnabledPlaylist(slot, dayISO);
        const trackIds = tracks.map(getTrackId).filter(Boolean);

        if (!trackIds.length) continue;

        occurrences.push({
          slotId,
          slotName: String(slot.name || "Slot").trim(),
          start: String(slot.start || "").trim(),
          end: String(slot.end || "").trim(),
          color: slot.color || "#FFD403",
          weekIndex,
          dayISO,
          weekday: d.getDay(),
          weekdayLabel: WEEKDAY_EN[d.getDay()],
          trackIds
        });
      }
    }

    // ordinamento stabile: settimana -> ora inizio -> nome slot -> giorno
    occurrences.sort((a, b) => {
      if (a.weekIndex !== b.weekIndex) return a.weekIndex - b.weekIndex;
      if (a.dayISO !== b.dayISO) return a.dayISO.localeCompare(b.dayISO);
      if (a.start !== b.start) return a.start.localeCompare(b.start);
      return a.slotName.localeCompare(b.slotName, undefined, { sensitivity: "base" });
    });

    return occurrences;
  }

  function formatHourDot(s) {
    return String(s || "").trim().replace(":", ".");
  }

  function buildPlannerPlaylistName(entry) {
    return [
      `Week_${entry.weekIndex}`,
      entry.weekdayLabel,
      `${entry.slotName} ${formatHourDot(entry.start)} - ${formatHourDot(entry.end)}`
    ].join(" - ");
  }

  async function commitPlannerToSpotify() {
    msg("Publishing planner to Spotify…");

    const entries = buildPlannerOccurrences();
    if (!entries.length) throw new Error("Nessuna playlist del planner trovata o nessun brano abilitato.");

    const created = [];

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      const name = buildPlannerPlaylistName(entry);
      const description = `SISMA Planner • ${entry.dayISO} • Week ${entry.weekIndex} • ${entry.start}-${entry.end}`;

      msg(`Creating ${i + 1}/${entries.length}: ${name}`);

      const out = await createSpotifyPlaylistNamed(name, entry.trackIds, description);
      created.push(out);
    }

    const html = [
      `<div><strong>${created.length}</strong> playlist create.</div>`,
      ...created.slice(0, 20).map(x =>
        `<div>• <a href="${x.url}" target="_blank" rel="noopener">${x.name}</a></div>`
      ),
      created.length > 20 ? `<div>…e altre ${created.length - 20}</div>` : ""
    ].join("");

    msg(html);
  }

  // =========================
  // UI wiring
  // =========================
  function updateButtons() {
    updateSpotifyStatusUI();
  }

  document.addEventListener("DOMContentLoaded", () => {
    const connectBtn = document.getElementById("btn_spotify_connect");
    const addBtn = document.getElementById("btn_spotify_add");
    const plannerBtn = document.getElementById("btnCommitSpotify");

    if (connectBtn) {
      connectBtn.addEventListener("click", async () => {
        try {
          if (getToken()) {
            updateSpotifyStatusUI();
            return;
          }

          sessionStorage.setItem("sisma_return_url", window.location.href);

          if (document.querySelector(".include-track")) {
            saveSelectionState();
          }

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

    if (plannerBtn) {
      plannerBtn.addEventListener("click", async () => {
        try {
          await commitPlannerToSpotify();
        } catch (e) {
          msg(`${e?.message || String(e)}`);
        }
      });
    }

    restoreSelectionState();
    restoreDiscoveryResultsIfNeeded();
    updateButtons();
  });

  window.spotifyHandleCallback = spotifyHandleCallback;
})();