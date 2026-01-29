// app/static/js/main.js

(function () {
  // -----------------------------
  // Dual-range sliders (0..100)
  // Expects:
  //  - inputs: {name}_min_ui, {name}_max_ui
  //  - fill bar: {name}_fill
  //  - hidden: {name}_min, {name}_max
  //  - optional legacy midpoint hidden: {name}
  //  - label: {name}_out
  // Wrap element: .range-wrap[data-name="{name}"]
  // -----------------------------
  function clampRanges(name) {
    const minEl = document.getElementById(`${name}_min_ui`);
    const maxEl = document.getElementById(`${name}_max_ui`);
    const fill = document.getElementById(`${name}_fill`);

    const hMin = document.getElementById(`${name}_min`);
    const hMax = document.getElementById(`${name}_max`);
    const hMid = document.getElementById(name); // optional legacy midpoint

    const out = document.getElementById(`${name}_out`);

    if (!minEl || !maxEl || !hMin || !hMax) return;

    let a = parseInt(minEl.value, 10);
    let b = parseInt(maxEl.value, 10);
    if (Number.isNaN(a)) a = 0;
    if (Number.isNaN(b)) b = 100;

    // keep order
    if (a > b) {
      if (document.activeElement === minEl) b = a;
      else a = b;
      minEl.value = String(a);
      maxEl.value = String(b);
    }

    // fill bar (optional)
    if (fill) {
      fill.style.left = `${a}%`;
      fill.style.width = `${Math.max(0, b - a)}%`;
    }

    // hidden payload: 0..1
    hMin.value = (a / 100).toFixed(3);
    hMax.value = (b / 100).toFixed(3);

    // optional legacy midpoint
    if (hMid) hMid.value = (((a + b) / 2) / 100).toFixed(3);

    // label
    if (out) out.textContent = `${a}–${b}`;
  }

  function initDualRangeWrap(wrap) {
    const name = wrap.getAttribute("data-name");
    if (!name) return;
    const minEl = document.getElementById(`${name}_min_ui`);
    const maxEl = document.getElementById(`${name}_max_ui`);
    if (!minEl || !maxEl) return;

    const onChange = () => clampRanges(name);
    minEl.addEventListener("input", onChange);
    maxEl.addEventListener("input", onChange);

    clampRanges(name);
  }

  document.querySelectorAll(".range-wrap").forEach(initDualRangeWrap);

  // -----------------------------
  // Dont care checkbox (only for the 4 main ones)
  // Expects:
  //  - checkbox: dc_{name}_ui
  //  - hidden: dc_{name} (0/1)
  //  - we disable BOTH min/max sliders: {name}_min_ui / {name}_max_ui
  // NOTE: if UI checkbox doesn't exist, this is a no-op (safe).
  // -----------------------------
  function bindDontCareRange(name) {
    const dc = document.getElementById(`dc_${name}_ui`);
    const hidden = document.getElementById(`dc_${name}`);
    const minEl = document.getElementById(`${name}_min_ui`);
    const maxEl = document.getElementById(`${name}_max_ui`);
    const wrap = document.querySelector(`.range-wrap[data-name="${name}"]`);

    if (!dc || !hidden || !minEl || !maxEl) return;

    function sync() {
      const on = dc.checked;
      hidden.value = on ? "1" : "0";

      minEl.disabled = on;
      maxEl.disabled = on;

      const opacity = on ? "0.35" : "1";
      minEl.style.opacity = opacity;
      maxEl.style.opacity = opacity;
      if (wrap) wrap.style.opacity = "1"; // keep labels readable; sliders already dim
    }

    dc.addEventListener("change", sync);
    sync();
  }

  ["danceability", "energy", "instrumentalness", "valence"].forEach(bindDontCareRange);

  // -----------------------------
  // Preset loader
  // -----------------------------
  const presetSel = document.getElementById("preset");

  function setDualRange01(name, mn01, mx01) {
    const minEl = document.getElementById(`${name}_min_ui`);
    const maxEl = document.getElementById(`${name}_max_ui`);
    if (!minEl || !maxEl) return;

    const a = Math.round(Math.max(0, Math.min(1, Number(mn01))) * 100);
    const b = Math.round(Math.max(0, Math.min(1, Number(mx01))) * 100);

    minEl.value = String(a);
    maxEl.value = String(b);

    // trigger clamp+hidden sync
    minEl.dispatchEvent(new Event("input", { bubbles: true }));
    maxEl.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function setIfPresent(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    if (val === undefined || val === null || val === "") return;
    el.value = String(val);
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }

  function resetDontCareOff() {
    ["danceability", "energy", "instrumentalness", "valence"].forEach((f) => {
      const ui = document.getElementById(`dc_${f}_ui`);
      const h = document.getElementById(`dc_${f}`);
      if (ui) ui.checked = false;
      if (h) h.value = "0";
      // re-enable sliders
      const minEl = document.getElementById(`${f}_min_ui`);
      const maxEl = document.getElementById(`${f}_max_ui`);
      if (minEl) {
        minEl.disabled = false;
        minEl.style.opacity = "1";
      }
      if (maxEl) {
        maxEl.disabled = false;
        maxEl.style.opacity = "1";
      }
    });
  }

  async function loadPreset(name) {
    if (!name) return;

    const res = await fetch(`/preset?name=${encodeURIComponent(name)}`);
    if (!res.ok) return;

    const p = await res.json();

    // preset implies you care (for the 4 main)
    resetDontCareOff();

    // MAIN dual ranges (0..1 -> 0..100)
    if (p.danceability_min != null && p.danceability_max != null) setDualRange01("danceability", p.danceability_min, p.danceability_max);
    if (p.energy_min != null && p.energy_max != null) setDualRange01("energy", p.energy_min, p.energy_max);
    if (p.instrumentalness_min != null && p.instrumentalness_max != null) setDualRange01("instrumentalness", p.instrumentalness_min, p.instrumentalness_max);
    if (p.valence_min != null && p.valence_max != null) setDualRange01("valence", p.valence_min, p.valence_max);

    // ADVANCED numeric min/max (these are <input type="number">, not sliders)
    setIfPresent("tempo_min", p.tempo_min);
    setIfPresent("tempo_max", p.tempo_max);

    setIfPresent("loudness_min", p.loudness_min);
    setIfPresent("loudness_max", p.loudness_max);

    setIfPresent("acousticness_min", p.acousticness_min);
    setIfPresent("acousticness_max", p.acousticness_max);

    setIfPresent("speechiness_min", p.speechiness_min);
    setIfPresent("speechiness_max", p.speechiness_max);

    setIfPresent("liveness_min", p.liveness_min);
    setIfPresent("liveness_max", p.liveness_max);

    // Genre preset (chips) — include only
    if (typeof window.SISMA_setGenres === "function") {
      const g = (typeof p.genre === "string") ? p.genre.trim() : "";
      if (g) window.SISMA_setGenres([g]);
      else window.SISMA_setGenres([]);
    }
  }

  if (presetSel) {
    presetSel.addEventListener("change", (ev) => loadPreset(ev.target.value));
  }
})();


// --- Song search + autocomplete ---
(function () {
  const form = document.querySelector("form.form");
  const submitBtn = form ? form.querySelector('button[type="submit"]') : null;

  const input = document.getElementById("song_query");
  const hiddenId = document.getElementById("song_track_id");
  const box = document.getElementById("song_suggestions");
  const enable = document.getElementById("enable_song_mode");

  if (!form || !submitBtn || !input || !hiddenId || !box || !enable) return;

  function hideBox() {
    box.style.display = "none";
    box.innerHTML = "";
  }

  function updateSubmitState() {
    const needsTrack = enable.checked;
    const hasTrack = hiddenId.value.trim().length > 0;
    submitBtn.disabled = needsTrack && !hasTrack;
  }

  function setSongMode(on) {
    input.disabled = !on;

    if (!on) {
      input.value = "";
      hiddenId.value = "";
      hideBox();
    } else {
      input.focus();
    }
    updateSubmitState();
  }

  function cleanArtistLabel(label) {
    if (!label) return "";
    const parts = String(label).split(" — ");
    if (parts.length < 2) return String(label);

    const title = parts[0].trim();
    let artists = parts.slice(1).join(" — ").trim();

    if (artists.startsWith("[") && artists.endsWith("]")) {
      artists = artists.slice(1, -1).trim();
      artists = artists.replace(/['"]/g, "").replace(/\s*,\s*/g, ", ").trim();
    }
    artists = artists.replace(/^['"]|['"]$/g, "").trim();
    return `${title} — ${artists}`;
  }

  function clampNum(x, lo, hi) {
    const v = Number(x);
    if (Number.isNaN(v)) return null;
    return Math.max(lo, Math.min(hi, v));
  }

  function setNumRange(idMin, idMax, val, halfWidth, lo, hi, decimals) {
    const minEl = document.getElementById(idMin);
    const maxEl = document.getElementById(idMax);
    if (!minEl || !maxEl) return;

    const v = clampNum(val, lo, hi);
    if (v === null) return;

    const a = clampNum(v - halfWidth, lo, hi);
    const b = clampNum(v + halfWidth, lo, hi);
    const d = (typeof decimals === "number") ? decimals : 3;

    minEl.value = a.toFixed(d);
    maxEl.value = b.toFixed(d);

    minEl.dispatchEvent(new Event("input", { bubbles: true }));
    maxEl.dispatchEvent(new Event("input", { bubbles: true }));
  }

  async function applyTrackFeatures(trackId) {
    const res = await fetch(`/track_features?track_id=${encodeURIComponent(trackId)}`);
    if (!res.ok) return;

    const data = await res.json();
    const f = (data && data.features) ? data.features : null;
    if (!f) return;

    // preset -> Custom (avoid confusion)
    const preset = document.getElementById("preset");
    if (preset) preset.value = "";

    // turn dont-care off for 4 main ranges
    ["danceability", "energy", "instrumentalness", "valence"].forEach((name) => {
      const dcUi = document.getElementById(`dc_${name}_ui`);
      const dcH = document.getElementById(`dc_${name}`);
      if (dcUi) dcUi.checked = false;
      if (dcH) dcH.value = "0";
    });

    // Set dual ranges around the base song (tight range: +/- widthUi points)
    function setAround(name, val01, widthUi) {
      const minEl = document.getElementById(`${name}_min_ui`);
      const maxEl = document.getElementById(`${name}_max_ui`);
      if (!minEl || !maxEl) return;

      const mid = Math.round(Math.max(0, Math.min(1, Number(val01))) * 100);
      const a = Math.max(0, mid - widthUi);
      const b = Math.min(100, mid + widthUi);

      minEl.value = String(a);
      maxEl.value = String(b);

      minEl.dispatchEvent(new Event("input", { bubbles: true }));
      maxEl.dispatchEvent(new Event("input", { bubbles: true }));
    }

    setAround("danceability", f.danceability, 8);
    setAround("energy", f.energy, 8);
    setAround("instrumentalness", f.instrumentalness, 8);
    setAround("valence", f.valence, 8);

    // Advanced hidden single values (legacy) + ALSO set min/max ranges,
    // because backend typically prioritizes *_min/_max for ranges.
    const tempo = document.getElementById("tempo");
    const loudness = document.getElementById("loudness");
    const acousticness = document.getElementById("acousticness");
    const speechiness = document.getElementById("speechiness");
    const liveness = document.getElementById("liveness");

    const mode = document.getElementById("mode");
    const key = document.getElementById("key");
    const timeSig = document.getElementById("time_signature");

    if (tempo && f.tempo != null) tempo.value = String(f.tempo);
    if (loudness && f.loudness != null) loudness.value = String(f.loudness);
    if (acousticness && f.acousticness != null) acousticness.value = String(f.acousticness);
    if (speechiness && f.speechiness != null) speechiness.value = String(f.speechiness);
    if (liveness && f.liveness != null) liveness.value = String(f.liveness);

    if (mode && f.mode != null) mode.value = String(f.mode);
    if (key && f.key != null) key.value = String(f.key);
    if (timeSig && f.time_signature != null) timeSig.value = String(f.time_signature);

    // Now set numeric ranges around base song:
    // Tempo: ±6 BPM, Loudness: ±3 dB, others: ±0.06
    setNumRange("tempo_min", "tempo_max", f.tempo, 6, 20, 250, 1);
    setNumRange("loudness_min", "loudness_max", f.loudness, 3, -60, 5, 2);

    setNumRange("acousticness_min", "acousticness_max", f.acousticness, 0.06, 0, 1, 3);
    setNumRange("speechiness_min", "speechiness_max", f.speechiness, 0.06, 0, 1, 3);
    setNumRange("liveness_min", "liveness_max", f.liveness, 0.06, 0, 1, 3);
  }

  setSongMode(enable.checked);
  enable.addEventListener("change", () => setSongMode(enable.checked));

  let lastQuery = "";
  let timer = null;

  function showSuggestions(items) {
    if (!items || items.length === 0) {
      hideBox();
      return;
    }
    box.innerHTML = items
      .map((it) => `<div class="suggestion-item" data-id="${it.track_id}">${cleanArtistLabel(it.label)}</div>`)
      .join("");
    box.style.display = "block";
  }

  async function fetchSuggestions(q) {
    const res = await fetch(`/track_search?q=${encodeURIComponent(q)}`);
    if (!res.ok) return [];
    return await res.json();
  }

  input.addEventListener("input", () => {
    const q = input.value.trim();
    hiddenId.value = "";
    updateSubmitState();

    if (q.length < 2) {
      hideBox();
      return;
    }

    clearTimeout(timer);
    timer = setTimeout(async () => {
      if (q === lastQuery) return;
      lastQuery = q;
      const items = await fetchSuggestions(q);
      showSuggestions(items);
    }, 180);
  });

  box.addEventListener("click", async (e) => {
    const item = e.target.closest(".suggestion-item");
    if (!item) return;

    const tid = item.getAttribute("data-id") || "";
    hiddenId.value = tid;
    input.value = item.textContent;
    hideBox();

    if (tid) await applyTrackFeatures(tid);
    updateSubmitState();
  });

  document.addEventListener("click", (e) => {
    if (!box.contains(e.target) && e.target !== input) hideBox();
  });

  form.addEventListener("submit", (e) => {
    if (enable.checked && !hiddenId.value.trim()) {
      e.preventDefault();
      updateSubmitState();
    }
  });
})();


// --- Reset button ---
(function () {
  const resetBtn = document.getElementById("btn_reset");
  if (!resetBtn) return;

  resetBtn.addEventListener("click", () => {
    window.location.href = "/";
  });
})();


/// --- Genre multi-select + autocomplete (include + exclude) ---
(function () {
  function parseCsvList(raw) {
    return String(raw || "")
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
  }

  function makeGenrePicker(opts) {
    const input = document.getElementById(opts.inputId);
    const box = document.getElementById(opts.boxId);
    const chips = document.getElementById(opts.chipsId);
    const hidden = document.getElementById(opts.hiddenId);

    if (!input || !box || !chips || !hidden) return null;

    const TOP = (window.SISMA_TOP_GENRES || [])
      .slice()
      .sort((a, b) => String(a).localeCompare(String(b)));

    const MIN_CHARS = 2;
    let timer = null;
    let lastQuery = "";

    const selected = new Map(); // key=lowercase genre, value=original

    function hideBox() {
      box.style.display = "none";
      box.innerHTML = "";
    }

    function emitFormInput() {
      const form = document.getElementById("playlist_form");
      if (form) form.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function syncHidden() {
      hidden.value = Array.from(selected.values()).join(",");
      emitFormInput();
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function renderChips() {
      const values = Array.from(selected.values());
      if (values.length === 0) {
        chips.innerHTML = "";
        chips.style.display = "none";
        return;
      }
      chips.style.display = "flex";
      chips.innerHTML = values
        .map((g) => {
          const safe = escapeHtml(g);
          return `
            <span class="chip" data-name="${safe}">
              ${safe}
              <button type="button" aria-label="Remove ${safe}" title="Remove">×</button>
            </span>
          `;
        })
        .join("");
    }

    function addGenre(name) {
      const cleaned = String(name || "").trim();
      if (!cleaned) return;
      const key = cleaned.toLowerCase();
      if (selected.has(key)) return;

      selected.set(key, cleaned);
      renderChips();
      syncHidden();

      input.value = "";
      input.focus();
      hideBox();
    }

    function removeGenre(name) {
      const key = String(name || "").toLowerCase().trim();
      if (!key) return;
      selected.delete(key);
      renderChips();
      syncHidden();
    }

    chips.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      const chip = e.target.closest(".chip");
      if (!chip) return;
      const name = chip.getAttribute("data-name") || chip.textContent;
      removeGenre(name);
    });

    async function fetchGenres(q) {
      const res = await fetch(`/genre_search?q=${encodeURIComponent(q)}`);
      if (!res.ok) return [];
      return await res.json();
    }

    function showSuggestions(items) {
      if (!items || items.length === 0) return hideBox();

      const filtered = items.filter(
        (x) => !selected.has(String(x).toLowerCase().trim())
      );
      if (filtered.length === 0) return hideBox();

      box.innerHTML = filtered
        .slice(0, 30)
        .map((g) => {
          const safe = escapeHtml(g);
          return `<div class="suggestion-item" data-genre="${safe}">${safe}</div>`;
        })
        .join("");
      box.style.display = "block";
    }

    function showTop() {
      showSuggestions(TOP);
    }

    input.addEventListener("focus", () => {
      if (input.value.trim().length === 0) showTop();
    });

    input.addEventListener("input", () => {
      const q = input.value.trim();
      if (q.length === 0) return showTop();
      if (q.length < MIN_CHARS) return hideBox();

      clearTimeout(timer);
      timer = setTimeout(async () => {
        if (q === lastQuery) return;
        lastQuery = q;
        const items = await fetchGenres(q);
        showSuggestions(items);
      }, 150);
    });

    input.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      if (box.style.display !== "block") return;
      const first = box.querySelector(".suggestion-item");
      if (!first) return;
      e.preventDefault();
      addGenre(first.getAttribute("data-genre") || first.textContent);
    });

    box.addEventListener("click", (e) => {
      const item = e.target.closest(".suggestion-item");
      if (!item) return;
      addGenre(item.getAttribute("data-genre") || item.textContent);
    });

    document.addEventListener("click", (e) => {
      if (!box.contains(e.target) && e.target !== input) hideBox();
    });

    // init from hidden DEFAULTS (important: do BEFORE any syncHidden)
    parseCsvList(hidden.value).forEach((g) => {
      const cleaned = String(g).trim();
      if (!cleaned) return;
      selected.set(cleaned.toLowerCase(), cleaned);
    });
    renderChips(); // do NOT syncHidden here (avoid rewriting server defaults)

    return {
      set(arr) {
        selected.clear();
        (arr || []).forEach((g) => {
          const cleaned = String(g || "").trim();
          if (!cleaned) return;
          selected.set(cleaned.toLowerCase(), cleaned);
        });
        renderChips();
        syncHidden();
        hideBox();
        input.value = "";
      },
    };
  }

  // include genres
  const includePicker = makeGenrePicker({
    inputId: "genre_query",
    boxId: "genre_suggestions",
    chipsId: "genre_chips",
    hiddenId: "genres",
  });

  // exclude genres
  makeGenrePicker({
    inputId: "exclude_genre_query",
    boxId: "exclude_genre_suggestions",
    chipsId: "exclude_genre_chips",
    hiddenId: "exclude_genres",
  });

  // Preset helper stays for INCLUDE only
  window.SISMA_setGenres = function (arr) {
    if (includePicker) includePicker.set(arr || []);
  };
})();


// --- Artist multi-select + autocomplete (include + exclude) ---
// PATCH: keep artist_weights aligned with artists; default new weights=1.0; do NOT wipe defaults.
(function () {
  function parseCsvList(raw) {
    return String(raw || "")
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
  }

  function parseWeights(raw) {
    const arr = String(raw || "")
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean)
      .map((x) => Number(x));
    return arr.map((x) => (Number.isFinite(x) ? x : 1.0));
  }

  function makeArtistPicker(opts) {
    const input = document.getElementById(opts.inputId);
    const box = document.getElementById(opts.boxId);
    const chips = document.getElementById(opts.chipsId);
    const hidden = document.getElementById(opts.hiddenId);
    const hiddenWeights = opts.hiddenWeightsId
      ? document.getElementById(opts.hiddenWeightsId)
      : null;

    if (!input || !box || !chips || !hidden) return null;

    const MIN_CHARS = 3;
    const DEBOUNCE_MS = 220;
    const MAX_ITEMS = 12;

    const TOP = (window.SISMA_TOP_ARTISTS || [])
      .slice()
      .sort((a, b) => String(a).localeCompare(String(b)));

    let timer = null;
    let lastQuery = "";
    let reqSeq = 0;
    const cache = new Map();

    const selected = new Map(); // key lower -> original
    const weights = new Map();  // key lower -> number

    function hideBox() {
      box.style.display = "none";
      box.innerHTML = "";
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function emitFormInput() {
      const form = document.getElementById("playlist_form");
      if (form) form.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function syncHidden() {
      const values = Array.from(selected.values());
      hidden.value = values.join(",");

      if (hiddenWeights) {
        // Keep alignment: same ordering as artists
        const ws = values.map((name) => {
          const k = String(name).toLowerCase().trim();
          const w = weights.get(k);
          return (typeof w === "number" && Number.isFinite(w)) ? String(w) : "1.0";
        });
        hiddenWeights.value = ws.join(",");
      }

      emitFormInput();
    }

    function renderChips() {
      const values = Array.from(selected.values());
      if (values.length === 0) {
        chips.style.display = "none";
        chips.innerHTML = "";
        return;
      }
      chips.style.display = "flex";
      chips.innerHTML = values
        .map((name) => {
          const safe = escapeHtml(name);
          return `
            <span class="chip" data-name="${safe}">
              ${safe}
              <button type="button" aria-label="Remove ${safe}" title="Remove">×</button>
            </span>
          `;
        })
        .join("");
    }

    function addArtist(name, wOpt) {
      const cleaned = String(name || "").trim();
      if (!cleaned) return;

      const key = cleaned.toLowerCase();
      if (selected.has(key)) return;

      selected.set(key, cleaned);
      const w = Number(wOpt);
      weights.set(key, Number.isFinite(w) ? w : 1.0);

      renderChips();
      syncHidden();

      input.value = "";
      input.focus();
      hideBox();
    }

    function removeArtist(name) {
      const key = String(name || "").toLowerCase().trim();
      if (!key) return;

      selected.delete(key);
      weights.delete(key);

      renderChips();
      syncHidden();
    }

    chips.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      const chip = e.target.closest(".chip");
      if (!chip) return;
      const name = chip.getAttribute("data-name") || chip.textContent;
      removeArtist(name);
    });

    async function fetchArtists(qLower) {
      if (cache.has(qLower)) return cache.get(qLower);

      const res = await fetch(`/artist_search?q=${encodeURIComponent(qLower)}`, {
        headers: { Accept: "application/json" },
      });
      if (!res.ok) return [];

      const items = await res.json();
      cache.set(qLower, items || []);
      return items || [];
    }

    function showSuggestions(items) {
      if (!items || items.length === 0) return hideBox();

      const filtered = [];
      for (const x of items) {
        const k = String(x).toLowerCase().trim();
        if (!k) continue;
        if (selected.has(k)) continue;
        filtered.push(String(x));
        if (filtered.length >= MAX_ITEMS) break;
      }

      if (filtered.length === 0) return hideBox();

      box.innerHTML = filtered
        .map((a) => {
          const safe = escapeHtml(a);
          return `<div class="suggestion-item" data-artist="${safe}">${safe}</div>`;
        })
        .join("");

      box.style.display = "block";
    }

    function showTop() {
      if (!TOP || TOP.length === 0) return hideBox();
      showSuggestions(TOP);
    }

    input.addEventListener("focus", () => {
      if (input.value.trim().length === 0) showTop();
    });

    input.addEventListener("input", () => {
      const q = input.value.trim();
      const qLower = q.toLowerCase();

      if (qLower.length === 0) {
        showTop();
        return;
      }

      if (qLower.length < MIN_CHARS) {
        lastQuery = qLower;
        hideBox();
        return;
      }

      clearTimeout(timer);
      timer = setTimeout(async () => {
        if (qLower === lastQuery) return;
        lastQuery = qLower;

        const mySeq = ++reqSeq;
        const items = await fetchArtists(qLower);
        if (mySeq !== reqSeq) return;

        showSuggestions(items);
      }, DEBOUNCE_MS);
    });

    input.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      if (box.style.display !== "block") return;

      const first = box.querySelector(".suggestion-item");
      if (!first) return;

      e.preventDefault();
      addArtist(first.getAttribute("data-artist") || first.textContent, 1.0);
    });

    box.addEventListener("click", (e) => {
      const item = e.target.closest(".suggestion-item");
      if (!item) return;
      addArtist(item.getAttribute("data-artist") || item.textContent, 1.0);
    });

    document.addEventListener("click", (e) => {
      if (!box.contains(e.target) && e.target !== input) hideBox();
    });

    // init from hidden DEFAULTS (important)
    const initArtists = parseCsvList(hidden.value);
    const initWeights = hiddenWeights ? parseWeights(hiddenWeights.value) : [];

    initArtists.forEach((a, i) => {
      const cleaned = String(a).trim();
      if (!cleaned) return;
      const key = cleaned.toLowerCase();
      selected.set(key, cleaned);
      const w = (hiddenWeights && initWeights[i] != null) ? initWeights[i] : 1.0;
      weights.set(key, Number.isFinite(w) ? w : 1.0);
    });

    renderChips(); // do NOT syncHidden here (avoid rewriting server defaults)

    return {
      set(arr, wArr) {
        selected.clear();
        weights.clear();
        (arr || []).forEach((a, i) => {
          const cleaned = String(a || "").trim();
          if (!cleaned) return;
          const key = cleaned.toLowerCase();
          selected.set(key, cleaned);
          const w = (wArr && wArr[i] != null) ? Number(wArr[i]) : 1.0;
          weights.set(key, Number.isFinite(w) ? w : 1.0);
        });
        renderChips();
        syncHidden();
        hideBox();
        input.value = "";
      },
    };
  }

  // include artists (with weights)
  makeArtistPicker({
    inputId: "artist_query",
    boxId: "artist_suggestions",
    chipsId: "artist_chips",
    hiddenId: "artists",
    hiddenWeightsId: "artist_weights",
  });

  // exclude artists (no weights)
  makeArtistPicker({
    inputId: "exclude_artist_query",
    boxId: "exclude_artist_suggestions",
    chipsId: "exclude_artist_chips",
    hiddenId: "exclude_artists",
  });
})();