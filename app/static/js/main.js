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
  //
  // Optional dataset on wrap:
  //  - data-min / data-max for scaling real values
  //  - OR data-db="1" for loudness mapping (-60..+5)
  // -----------------------------

  function getScale(wrap) {
    if (!wrap) return { dmin: 0, dmax: 1 };

    // loudness special-case
    if (wrap.dataset.db === "1") return { dmin: -60, dmax: 5 };

    const dmin = wrap.dataset.min != null ? Number(wrap.dataset.min) : 0;
    const dmax = wrap.dataset.max != null ? Number(wrap.dataset.max) : 1;
    return { dmin, dmax };
  }

  function clampRanges(name) {
    const wrap = document.querySelector(`.range-wrap[data-name="${name}"]`);
    const minEl = document.getElementById(`${name}_min_ui`);
    const maxEl = document.getElementById(`${name}_max_ui`);
    const fill = document.getElementById(`${name}_fill`);

    const hMin = document.getElementById(`${name}_min`);
    const hMax = document.getElementById(`${name}_max`);
    const hMid = document.getElementById(name); // optional legacy midpoint
    const out = document.getElementById(`${name}_out`);

    if (!minEl || !maxEl) return;

    let a = Number(minEl.value);
    let b = Number(maxEl.value);

    // keep a <= b
    if (a > b) {
      const tmp = a;
      a = b;
      b = tmp;
      minEl.value = String(a);
      maxEl.value = String(b);
    }

    // update fill (0..100)
    if (fill) {
      fill.style.left = `${a}%`;
      fill.style.width = `${b - a}%`;
    }

    // scaled values -> hidden fields
    const { dmin, dmax } = getScale(wrap);
    const fromPct = (pct) => dmin + (pct / 100) * (dmax - dmin);

    if (hMin) hMin.value = String(fromPct(a));
    if (hMax) hMax.value = String(fromPct(b));

    // optional midpoint legacy hidden
    if (hMid) hMid.value = String(fromPct((a + b) / 2));

    // output label
    if (out) {
      if (wrap && wrap.dataset.db === "1") {
        // show dB nicely
        const va = fromPct(a).toFixed(1);
        const vb = fromPct(b).toFixed(1);
        out.textContent = `${va} – ${vb} dB`;
      } else {
        out.textContent = `${Math.round(a)} – ${Math.round(b)}`;
      }
    }
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
  // Dont care checkbox (optional UI)
  // If checkbox is missing in HTML, this does nothing.
  // -----------------------------
  function bindDontCareRange(name) {
    const dcUi = document.getElementById(`dc_${name}_ui`); // optional
    const hidden = document.getElementById(`dc_${name}`);
    const minEl = document.getElementById(`${name}_min_ui`);
    const maxEl = document.getElementById(`${name}_max_ui`);
    const wrap = document.querySelector(`.range-wrap[data-name="${name}"]`);

    if (!dcUi || !hidden || !minEl || !maxEl) return;

    function sync() {
      const on = dcUi.checked;
      hidden.value = on ? "1" : "0";

      minEl.disabled = on;
      maxEl.disabled = on;

      const opacity = on ? "0.35" : "1";
      minEl.style.opacity = opacity;
      maxEl.style.opacity = opacity;
      if (wrap) wrap.style.opacity = "1";
    }

    dcUi.addEventListener("change", sync);
    sync();
  }

  ["danceability", "energy", "loudness", "valence"].forEach(bindDontCareRange);

  // -----------------------------
  // Preset loader
  // -----------------------------
  const presetSel = document.getElementById("preset");

  function setDualRange01(name, mn01, mx01) {
    // sets UI sliders from 0..1 values
    const minEl = document.getElementById(`${name}_min_ui`);
    const maxEl = document.getElementById(`${name}_max_ui`);
    if (!minEl || !maxEl) return;

    const clamp01 = (x) => Math.max(0, Math.min(1, Number(x)));
    const a = Math.round(clamp01(mn01) * 100);
    const b = Math.round(clamp01(mx01) * 100);

    minEl.value = String(a);
    maxEl.value = String(b);

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
    ["danceability", "energy", "loudness", "valence"].forEach((f) => {
      const h = document.getElementById(`dc_${f}`);
      if (h) h.value = "0";

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
    resetDontCareOff();

    // ONLY the 4 main dual sliders:
    if (p.danceability_min != null && p.danceability_max != null) setDualRange01("danceability", p.danceability_min, p.danceability_max);
    if (p.energy_min != null && p.energy_max != null) setDualRange01("energy", p.energy_min, p.energy_max);
    if (p.valence_min != null && p.valence_max != null) setDualRange01("valence", p.valence_min, p.valence_max);

    // loudness comes in dB usually, so set advanced min/max instead (and let UI be whatever)
    setIfPresent("loudness_min", p.loudness_min);
    setIfPresent("loudness_max", p.loudness_max);

    // advanced numeric min/max
    setIfPresent("tempo_min", p.tempo_min);
    setIfPresent("tempo_max", p.tempo_max);

    setIfPresent("instrumentalness_min", p.instrumentalness_min);
    setIfPresent("instrumentalness_max", p.instrumentalness_max);

    setIfPresent("acousticness_min", p.acousticness_min);
    setIfPresent("acousticness_max", p.acousticness_max);

    setIfPresent("speechiness_min", p.speechiness_min);
    setIfPresent("speechiness_max", p.speechiness_max);

    setIfPresent("liveness_min", p.liveness_min);
    setIfPresent("liveness_max", p.liveness_max);

    setIfPresent("mode", p.mode);
    setIfPresent("key", p.key);
    setIfPresent("time_signature", p.time_signature);
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
    ["danceability", "energy", "loudness", "valence"].forEach((name) => {
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

      // blocca il submit del form (altrimenti ti genera la playlist)
      e.preventDefault();
      e.stopPropagation();

      // se dropdown aperto, prendi il primo suggerimento
      if (box.style.display === "block") {
        const first = box.querySelector(".suggestion-item");
        if (first) {
          addGenre(first.getAttribute("data-genre") || first.textContent);
          return;
        }
      }

      // altrimenti crea chip dalla keyword scritta
      addGenre(input.value);
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

const includeArtistPicker = makeArtistPicker({
  inputId: "artist_query",
  boxId: "artist_suggestions",
  chipsId: "artist_chips",
  hiddenId: "artists",
  hiddenWeightsId: "artist_weights",
});

const excludeArtistPicker = makeArtistPicker({
  inputId: "exclude_artist_query",
  boxId: "exclude_artist_suggestions",
  chipsId: "exclude_artist_chips",
  hiddenId: "exclude_artists",
});

})();

function getCurrentGenres() {
  const hidden = document.getElementById("genres");
  return hidden ? parseHiddenList(hidden.value) : [];
}


// =====================================================
// WORLD.SVG MAP -> MULTI-SELECT COUNTRIES -> GENRES UNION
// (NO SimpleMaps)
// =====================================================

// ---- Region selection state ----
const REGION_STATE = {
  enabled: true,
  mode: "union",
  selected: new Set(),
  cache: new Map(),
  suppressChips: true,
  prevGenres: null
};

// Optional labels (fallback to ISO if unknown)
const ISO2_LABEL = {
  IT: "Italy",
  FR: "France",
  DE: "Germany",
  ES: "Spain",
  PT: "Portugal",
  GB: "United Kingdom",
  IE: "Ireland",
  US: "United States",
  CA: "Canada",
  CH: "Switzerland",
  AT: "Austria",
  NL: "Netherlands",
  BE: "Belgium",
  SE: "Sweden",
  NO: "Norway",
  DK: "Denmark",
  FI: "Finland",
  IS: "Iceland",
  PL: "Poland",
  CZ: "Czech Republic",
  HU: "Hungary",
  RO: "Romania",
  BG: "Bulgaria",
  GR: "Greece",
  TR: "Turkey",
  AL: "Albania",
  AU: "Australia",
  IN: "India",
  JP: "Japan",
  CN: "China",
  MX: "Mexico",
  AR: "Argentina",
  BR: "Brazil",
  CL: "Chile",
  CO: "Colombia",
  PE: "Peru",
  VE: "Venezuela"
};


// Map colors (JS-enforced: overrides inline SVG fills)
const MAP_COLORS = {
  defaultFill: "#0b0b0c",
  defaultStroke: "rgba(255,255,255,.55)",
  hoverFill: "#1a1a1d",
  activeFill: "#FFD403",
  activeStroke: "#ffffff",
};

// -----------------------------
// Helpers: hidden list + chips
// -----------------------------
function parseHiddenList(value) {
  if (!value) return [];
  return value
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
}

function writeHiddenList(el, items) {
  el.value = items.join(","); // CSV
}

function renderChips(containerEl, items, hiddenInputEl) {
  containerEl.innerHTML = "";

  items.forEach((genre) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.dataset.value = genre;
    chip.textContent = genre;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip-remove";
    btn.textContent = "×";
    btn.addEventListener("click", () => {
      const current = parseHiddenList(hiddenInputEl.value);
      const next = current.filter(g => g !== genre);
      writeHiddenList(hiddenInputEl, next);
      renderChips(containerEl, next, hiddenInputEl);
    });

    chip.appendChild(btn);
    containerEl.appendChild(chip);
  });
}

function setIncludeGenres(genres, { replace = true, render = true } = {}) {
  const hidden = document.getElementById("genres");
  const chips = document.getElementById("genre_chips");
  if (!hidden) return;

  const current = parseHiddenList(hidden.value);
  const set = new Set(current);

  const incoming = (genres || [])
    .map(g => (g || "").trim())
    .filter(Boolean);

  let next;
  if (replace) {
    next = Array.from(new Set(incoming));
  } else {
    incoming.forEach(g => set.add(g));
    next = Array.from(set);
  }

  writeHiddenList(hidden, next);

  // In region-mode: suppress chips (but keep hidden updated)
  if (render && chips) {
    renderChips(chips, next, hidden);
  } else if (chips && REGION_STATE.suppressChips) {
    chips.innerHTML = "";
  }
}

// -----------------------------
// API: fetch region genres
// -----------------------------
async function fetchRegionGenres(iso) {
  iso = String(iso || "").toUpperCase();
  if (!iso) return [];

  if (REGION_STATE.cache.has(iso)) return REGION_STATE.cache.get(iso);

  const base = window.location.origin;
  const url = `${base}/api/region-genres?iso=${encodeURIComponent(iso)}&top_n=120`;

  const res = await fetch(url);
  const text = await res.text();
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);

  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    throw new Error(`Non-JSON response (${res.status}). Body: ${text.slice(0, 80)}`);
  }

  if (!data.ok) throw new Error(data.error || "API error");

  const genres = data.genres || [];
  REGION_STATE.cache.set(iso, genres);
  return genres;
}

// -----------------------------
// UI: status line
// -----------------------------
function updateRegionStatus() {
  const clearBtn = document.getElementById("clear_region");
  const status = document.getElementById("region_status");

  const isos = Array.from(REGION_STATE.selected);

  if (isos.length === 0) {
    if (status) status.textContent = "";
    if (clearBtn) clearBtn.textContent = "Clear";
    return;
  }

  const labels = isos.map(i => ISO2_LABEL[i] || i);
  const shown = labels.slice(0, 2);
  const extra = labels.length - shown.length;

  const txt = extra > 0
    ? `${shown.join(", ")} (+${extra})`
    : `${shown.join(", ")}`;

  if (status) status.textContent = `Regions: ${txt}`;
  if (clearBtn) clearBtn.textContent = `Clear (${txt})`;
}




function isRegionMode() {
  return REGION_STATE.selected.size > 0;
}

function writeRegionIsosHidden() {
  const el = document.getElementById("region_isos");
  if (!el) return;
  el.value = Array.from(REGION_STATE.selected).sort().join(",");
}

function readRegionIsosHidden() {
  const el = document.getElementById("region_isos");
  if (!el) return [];
  return (el.value || "")
    .split(",")
    .map(s => s.trim().toUpperCase())
    .filter(Boolean);
}



// -----------------------------
// Map coloring (JS-enforced)
// -----------------------------
function paintCountry(el, active) {
  if (!el) return;
  if (active) {
    el.style.fill = MAP_COLORS.activeFill;
    el.style.stroke = MAP_COLORS.activeStroke;
  } else {
    el.style.fill = MAP_COLORS.defaultFill;
    el.style.stroke = MAP_COLORS.defaultStroke;
  }
}

function syncMapColorsWithSelection() {
  const host = document.getElementById("world_svg_host");
  if (!host) return;

  host.querySelectorAll("[data-iso]").forEach(el => {
    const iso = (el.getAttribute("data-iso") || "").toUpperCase();
    const active = REGION_STATE.selected.has(iso);
    el.classList.toggle("active", active);
    paintCountry(el, active);
  });
}

function bindMapHover() {
  const host = document.getElementById("world_svg_host");
  if (!host) return;

  host.addEventListener("mouseover", (ev) => {
    const el = ev.target.closest("[data-iso]");
    if (!el) return;

    const iso = (el.getAttribute("data-iso") || "").toUpperCase();
    if (REGION_STATE.selected.has(iso)) return; // keep active style

    el.style.fill = MAP_COLORS.hoverFill;
  });

  host.addEventListener("mouseout", (ev) => {
    const el = ev.target.closest("[data-iso]");
    if (!el) return;

    const iso = (el.getAttribute("data-iso") || "").toUpperCase();
    if (REGION_STATE.selected.has(iso)) return;

    el.style.fill = MAP_COLORS.defaultFill;
  });
}

// -----------------------------
// Main toggle handler (multi-select)
// -----------------------------
async function onRegionClick(iso) {
  const status = document.getElementById("region_status");
  iso = String(iso || "").toUpperCase();
  if (!iso) return;

  try {
    if (status) status.textContent = `Loading ${ISO2_LABEL[iso] || iso}…`;

    // -----------------------------
    // toggle OFF
    // -----------------------------
    if (REGION_STATE.selected.has(iso)) {
      REGION_STATE.selected.delete(iso);

      writeRegionIsosHidden();          
      updateRegionStatus();
      toggleGenreUIForRegionMode();     
      syncMapColorsWithSelection();

      if (REGION_STATE.selected.size === 0) {
        setIncludeGenres(REGION_STATE.prevGenres || [], { replace: true, render: true });
        REGION_STATE.prevGenres = null;
        toggleGenreUIForRegionMode();
      }

      if (status) status.textContent = "";
      return;
    }

    // -----------------------------
    // toggle ON
    // -----------------------------
    if (REGION_STATE.selected.size === 0 && REGION_STATE.prevGenres == null) {
      REGION_STATE.prevGenres = getCurrentGenres();
    }

    REGION_STATE.selected.add(iso);
    writeRegionIsosHidden();

    await fetchRegionGenres(iso);

    updateRegionStatus();
    toggleGenreUIForRegionMode();
    syncMapColorsWithSelection();

    if (status) status.textContent = "";
  } catch (e) {
    if (status) status.textContent = "Region load error";
    console.error(e);
    syncMapColorsWithSelection();
  }
}


// -----------------------------
// Bind clear button
// -----------------------------
(function bindClearRegion() {
  const clearBtn = document.getElementById("clear_region");
  if (!clearBtn) return;

  clearBtn.addEventListener("click", () => {
    REGION_STATE.selected.clear();
    REGION_STATE.cache.clear();
    writeRegionIsosHidden();
    updateRegionStatus();

    setIncludeGenres(REGION_STATE.prevGenres || [], { replace: true, render: true });
    REGION_STATE.prevGenres = null;

    toggleGenreUIForRegionMode();
    syncMapColorsWithSelection();
  });
})();



// -----------------------------
// Load SVG map + bind click delegation
// -----------------------------
(async function loadAndBindWorldSVG() {
  const host = document.getElementById("world_svg_host");
  if (!host) return;

  const res = await fetch("/static/maps/world.svg");
  const svgText = await res.text();
  host.innerHTML = svgText;

  // Convert lowercase ISO2 ids (e.g. "it") into data-iso="IT"
  host.querySelectorAll("svg [id]").forEach(el => {
    const id = (el.getAttribute("id") || "").trim();
    if (/^[a-z]{2}$/.test(id)) {
      el.setAttribute("data-iso", id.toUpperCase());
    }
  });

  // Bind hover once
  bindMapHover();

  // Initial paint (default fill + any preselected)
  syncMapColorsWithSelection();

  // Click delegation
  host.addEventListener("click", (ev) => {
    const el = ev.target.closest("[data-iso]");
    if (!el) return;
    const iso = (el.getAttribute("data-iso") || "").toUpperCase();
    if (!iso) return;
    onRegionClick(iso);
  });
})();

function toggleGenreUIForRegionMode() {
  const chips = document.getElementById("genre_chips");
  const input = document.getElementById("genre_query");
  const sugg = document.getElementById("genre_suggestions");

  const on = isRegionMode();

  // Chips manuali SEMPRE visibili (sono l’unica cosa che vogliamo mostrare)
  if (chips) chips.style.display = "";

  // Input SEMPRE utilizzabile: l’utente può aggiungere "salsa" dopo aver cliccato IT
  if (input) {
    input.disabled = false;
    input.placeholder = on ? "Add genre… (regions also active)" : "Add genre…";
  }

  // chiudi suggerimenti quando cambi modalità
  if (sugg) sugg.style.display = "none";
}



(function restoreRegionSelectionFromHidden() {
  const isos = readRegionIsosHidden();
  if (!isos.length) {
    toggleGenreUIForRegionMode();
    return;
  }

  // reimposta stato regioni
  isos.forEach(iso => REGION_STATE.selected.add(iso));
  updateRegionStatus();
  writeRegionIsosHidden();

  // fetch generi per ciascuna regione e applica union
  Promise.all(isos.map(fetchRegionGenres))
    .then(() => {
      toggleGenreUIForRegionMode();
      syncMapColorsWithSelection();
    })
    .catch((e) => {
      console.error(e);
      toggleGenreUIForRegionMode();
      syncMapColorsWithSelection();
    });
})();


// --- Planner weekday toggles (Discovery page) ---
(function () {
  const host = document.getElementById("planner_weekdayToggles");
  const hidden = document.getElementById("planner_weekdays");
  if (!host || !hidden) return;

  function getSelectedSet() {
    return new Set(
      String(hidden.value || "")
        .split(",")
        .map(x => parseInt(x.trim(), 10))
        .filter(Number.isFinite)
    );
  }

  function setSelectedSet(set) {
    // fallback: non permettere set vuoto
    if (set.size === 0) [1,2,3,4,5].forEach(x => set.add(x));

    hidden.value = Array.from(set).sort((a,b)=>a-b).join(",");

    // refresh UI
    host.querySelectorAll(".wd-btn").forEach(btn => {
      const v = parseInt(btn.dataset.wd, 10);
      btn.classList.toggle("active", set.has(v));
    });

    // trigger per download link ecc.
    const form = document.getElementById("playlist_form");
    if (form) form.dispatchEvent(new Event("input", { bubbles: true }));
  }

  // init: se hidden vuoto, usa stato dal template (active) oppure Mon-Fri
  let selected = getSelectedSet();
  if (selected.size === 0) {
    const fromDom = new Set(
      Array.from(host.querySelectorAll(".wd-btn.active"))
        .map(b => parseInt(b.dataset.wd, 10))
        .filter(Number.isFinite)
    );
    selected = fromDom.size ? fromDom : new Set([1,2,3,4,5]);
  }
  setSelectedSet(selected);

  // click delegation
  host.addEventListener("click", (e) => {
    const btn = e.target.closest(".wd-btn");
    if (!btn) return;
    e.preventDefault();

    const wd = parseInt(btn.dataset.wd, 10);
    if (!Number.isFinite(wd)) return;

    const next = getSelectedSet();
    if (next.has(wd)) next.delete(wd);
    else next.add(wd);

    setSelectedSet(next);
  });
})();

// --- Keyword chips (include + exclude) ---
(function () {
  function parseCsvList(raw) {
    return String(raw || "")
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
  }

  function makeKeywordPicker({ inputId, chipsId, hiddenId }) {
    const input  = document.getElementById(inputId);
    const chips  = document.getElementById(chipsId);
    const hidden = document.getElementById(hiddenId);
    if (!input || !chips || !hidden) return;

    const selected = new Map(); // lower -> original

    function syncHidden() {
      hidden.value = Array.from(selected.values()).join(",");
      const form = document.getElementById("playlist_form");
      if (form) form.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function render() {
      const values = Array.from(selected.values());
      if (!values.length) {
        chips.innerHTML = "";
        chips.style.display = "none";
        return;
      }
      chips.style.display = "flex";
      chips.innerHTML = values.map((k) => {
        const safe = escapeHtml(k);
        return `
          <span class="chip" data-name="${safe}">
            ${safe}
            <button type="button" aria-label="Remove ${safe}" title="Remove">×</button>
          </span>`;
      }).join("");
    }

    function addKeyword(v) {
      const cleaned = String(v || "").trim();
      if (!cleaned) return;
      const key = cleaned.toLowerCase();
      if (selected.has(key)) return;
      selected.set(key, cleaned);
      render();
      syncHidden();
      input.value = "";
      input.focus();
    }

    function removeKeyword(v) {
      const key = String(v || "").toLowerCase().trim();
      selected.delete(key);
      render();
      syncHidden();
    }

    input.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      e.stopPropagation();
      addKeyword(input.value);
    });

    chips.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      const chip = e.target.closest(".chip");
      if (!chip) return;
      removeKeyword(chip.getAttribute("data-name") || chip.textContent);
    });

    // init from hidden defaults
    parseCsvList(hidden.value).forEach((k) => {
      selected.set(k.toLowerCase(), k);
    });
    render();
  }

  // 
  makeKeywordPicker({ inputId: "keyword_query", chipsId: "keyword_chips", hiddenId: "keywords" });
  makeKeywordPicker({ inputId: "exclude_keyword_query", chipsId: "exclude_keyword_chips", hiddenId: "exclude_keywords" });
})();
