(() => {
  // ============================================================
  // SISMA Planner — Beta
  // - grid with fused blocks overlay (contiguous runs merged)
  // - translucent slot colors
  // - scroll-synced header
  // - export timetable JSON + Spotify stub
  // Endpoint: POST /planner/api/generate_batch
  // ============================================================

  const START_DAY = "10:00";
  const END_DAY   = "24:00";
  const STEP_MIN  = 30;
  const COLS = 14;
  const LS_PLAN_KEY = "sisma_planner_plan_v1";

  const DEFAULT_K = 50;
  const DEFAULT_MAX_PER_ARTIST = 2;
  const DEFAULT_COOLDOWN_DAYS = 2;

  const WEEKDAY_IT = ["Dom", "Lun", "Mar", "Mer", "Gio", "Ven", "Sab"];

  // DOM
  const daysHead = document.getElementById("daysHead");
  const timeCol  = document.getElementById("timeCol");
  const daysGrid = document.getElementById("daysGrid");
  const gridScroll = document.getElementById("gridScroll");

  const slotInfo = document.getElementById("slotInfo");
  const slotPlaylistList = document.getElementById("slotPlaylistList");
  const slotList = document.getElementById("slotList");
  const summaryStats = document.getElementById("summaryStats");

  const slotColorEdit = document.getElementById("slotColorEdit");

  const slotTracks = document.getElementById("slotTracks");
  const slotEnergy = document.getElementById("slotEnergy");
  const slotMood   = document.getElementById("slotMood");
  const slotDance  = document.getElementById("slotDance");
  const slotBpm    = document.getElementById("slotBpm");
  const slotGenres = document.getElementById("slotGenres");

  const btnExportPlan = document.getElementById("btnExportPlan");
  const btnClearPlan  = document.getElementById("btnClearPlan");
  const btnPrevWindow = document.getElementById("btnPrevWindow");
  const btnNextWindow = document.getElementById("btnNextWindow");
  const windowLabel   = document.getElementById("windowLabel");

  const btnCommitSpotify = document.getElementById("btnCommitSpotify");
  const btnDownloadTimetable = document.getElementById("btnDownloadTimetable");

  const debugBox = document.getElementById("debugBox");

  // State
  let rows = 0;
  let startDate = null; // Monday of current window
  let gridState = [];   // [rows][COLS] -> slotId|null
  let slots = {};       // slotId -> slot
  let selected = { slotId: null, dayISO: null };

  // layers
  let cellsLayer = null;  // .p-cells
  let blocksLayer = null; // .p-blocks

  // ---------------- Utils ----------------
  const pad2 = (n) => String(n).padStart(2, "0");

  function timeToMin(t) {
    const [hh, mm] = String(t).split(":").map(x => parseInt(x, 10));
    return hh * 60 + mm;
  }
  function minToTime(m) {
    const hh = Math.floor(m / 60);
    const mm = m % 60;
    return `${pad2(hh)}:${pad2(mm)}`;
  }
  function clampToStep(mins) {
    return Math.round(mins / STEP_MIN) * STEP_MIN;
  }
  function todayStart() {
    const d = new Date();
    d.setHours(0,0,0,0);
    return d;
  }
  function getStartOfWeekMonday(d) {
    const x = new Date(d);
    x.setHours(0,0,0,0);
    const day = x.getDay(); // 0=Dom ... 6=Sab
    const diff = (day === 0) ? -6 : (1 - day);
    x.setDate(x.getDate() + diff);
    return x;
  }
  function fmtLocalISODate(d) {
    return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
  }

  function computeDayISO(c) {
    const d = new Date(startDate);
    d.setDate(startDate.getDate() + c);
    return fmtLocalISODate(d);
  }


  // deterministic seed from string (fast hash)
  function hashSeed(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return Math.abs(h);
  }

  function safeText(x, fallback="") {
    const s = (x == null) ? "" : String(x);
    return s.trim() ? s.trim() : fallback;
  }


  function ensureTrackEnabledFlags(slot, dayISO) {
    if (!slot || !dayISO) return [];
    if (!slot.playlistsByDay) slot.playlistsByDay = {};

    const playlist = slot.playlistsByDay[dayISO] || [];
    playlist.forEach((t) => {
      if (typeof t.enabled !== "boolean") t.enabled = true;
    });
    slot.playlistsByDay[dayISO] = playlist;
    return playlist;
  }

  function getEnabledPlaylist(slot, dayISO) {
    const playlist = ensureTrackEnabledFlags(slot, dayISO);
    return playlist.filter(t => t.enabled !== false);
  }
    

  function uniq(arr) {
    const out = [];
    const seen = new Set();
    (arr || []).forEach(v => {
      const k = String(v);
      if (!k || seen.has(k)) return;
      seen.add(k);
      out.push(v);
    });
    return out;
  }

  function normalizeWeekdays(wd) {
    const out = [];
    (wd || []).forEach(x => {
      const n = Number(x);
      if (Number.isFinite(n) && n >= 0 && n <= 6) out.push(n);
    });
    const u = uniq(out);
    return u.length ? u : [1,2,3,4,5];
  }

  function hexToRgb(hex) {
    const s = String(hex || "").trim();
    const m = s.match(/^#?([0-9a-f]{6})$/i);
    if (!m) return { r: 255, g: 212, b: 3 };
    const n = parseInt(m[1], 16);
    return { r: (n>>16)&255, g: (n>>8)&255, b: n&255 };
  }

  function setDebug(obj) {
    if (!debugBox) return;
    try {
      debugBox.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
    } catch {
      debugBox.textContent = String(obj);
    }
  }

  function save() {
    try {
      const payload = {
        version: 1,
        startDateISO: startDate ? fmtLocalISODate(startDate) : null,
        slots,
        report: {}
      };
      localStorage.setItem(LS_PLAN_KEY, JSON.stringify(payload));
    } catch {}
  }

  function load() {
    try {
      const raw = localStorage.getItem(LS_PLAN_KEY);
      if (!raw) return false;

      const obj = JSON.parse(raw);
      if (!obj || typeof obj !== "object") return false;

      const sd = obj.startDateISO ? new Date(`${obj.startDateISO}T00:00:00`) : null;
      startDate = sd ? getStartOfWeekMonday(sd) : null;
      slots = obj.slots || {};

      Object.keys(slots).forEach(id => {
        const s = slots[id];
        if (!s) return;
        s.id = s.id || id;
        if (!s.playlistsByDay) s.playlistsByDay = {};
        if (!s.weekdays) s.weekdays = [1,2,3,4,5];
        if (!s.discovery) s.discovery = {};
        if (s.k == null) s.k = DEFAULT_K;
        if (s.max_per_artist == null) s.max_per_artist = DEFAULT_MAX_PER_ARTIST;
        if (s.cooldown_days == null) s.cooldown_days = DEFAULT_COOLDOWN_DAYS;
      });

      return true;
    } catch {
      return false;
    }
  }

  function clearAll() {
    slots = {};
    selected = { slotId: null, dayISO: null };
    localStorage.removeItem(LS_PLAN_KEY);
    rebuildEverything();
    renderSidebarEmpty("Planner resettato.");
    setDebug("—");
  }




  // ---------------- Draft (Discovery -> Planner) ----------------
  // sessionStorage.setItem("sisma_planner_draft", JSON.stringify({
  //   slot: { name, color, start, end, weekdays:[1..5] },
  //   discovery: { ...payload... },
  //   generation: { k, max_per_artist, cooldown_days }
  // }))


  function buildSlotId(slot, discovery) {
    const core = {
      name: safeText(slot?.name, "Slot"),
      start: safeText(slot?.start, "10:00"),
      end: safeText(slot?.end, "11:00"),
      weekdays: normalizeWeekdays(slot?.weekdays),
      discovery: discovery || {}
    };
    const h = hashSeed(JSON.stringify(core)).toString(16).slice(0, 10);
    return `slot_${h}`;
  }

  // ---------------- Grid setup ----------------
  function ensureGridSize() {
    const start = timeToMin(START_DAY);
    const end   = timeToMin(END_DAY);
    rows = Math.floor((end - start) / STEP_MIN);
    if (!startDate) startDate = getStartOfWeekMonday(todayStart());
    gridState = Array.from({ length: rows }, () => Array.from({ length: COLS }, () => null));
  }

  function buildHeads() {
    if (!daysHead) return;
    daysHead.innerHTML = "";
    for (let c = 0; c < COLS; c++) {
      const d = new Date(startDate);
      d.setDate(startDate.getDate() + c);
      const el = document.createElement("div");
      el.className = "cal-day-head";
      el.textContent = `${WEEKDAY_IT[d.getDay()]} ${pad2(d.getDate())}/${pad2(d.getMonth()+1)}`;
      daysHead.appendChild(el);
    }
  }

  function buildTimeColumn() {
    if (!timeCol) return;
    timeCol.innerHTML = "";
    const start = timeToMin(START_DAY);
    const end   = timeToMin(END_DAY);
    for (let m = start; m < end; m += 60) {
      const row = document.createElement("div");
      row.className = "cal-time-row";
      row.textContent = minToTime(m);
      timeCol.appendChild(row);
    }
  }

  function buildGridDOM() {
    if (!daysGrid) return;

    daysGrid.innerHTML = "";

    // base cells
    cellsLayer = document.createElement("div");
    cellsLayer.className = "p-cells";

    // overlay blocks
    blocksLayer = document.createElement("div");
    blocksLayer.className = "p-blocks";

    // create cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < COLS; c++) {
        const cell = document.createElement("div");
        cell.className = "p-cell";
        cell.dataset.r = String(r);
        cell.dataset.c = String(c);
        cell.addEventListener("click", onCellClick);
        cellsLayer.appendChild(cell);
      }
    }

    daysGrid.appendChild(cellsLayer);
    daysGrid.appendChild(blocksLayer);
  }

  function computeCoverageCellsForSlot(slot) {
    const out = [];
    const startMin = timeToMin(START_DAY);
    const a = Math.max(startMin, clampToStep(timeToMin(slot.start)));
    const b = Math.min(timeToMin(END_DAY), clampToStep(timeToMin(slot.end)));
    if (b <= a) return out;

    const r0 = Math.floor((a - startMin) / STEP_MIN);
    const r1 = Math.ceil((b - startMin) / STEP_MIN) - 1;

    const weekdays = new Set(normalizeWeekdays(slot.weekdays || [1,2,3,4,5]));
    const weeks = Math.max(1, Number(slot.weeks) || 1);

    for (let c = 0; c < COLS; c++) {
      const weekIndex = Math.floor(c / 7) + 1;
      if (weekIndex > weeks) continue;

      const d = new Date(startDate);
      d.setDate(startDate.getDate() + c);

      if (!weekdays.has(d.getDay())) continue;

      for (let r = r0; r <= r1; r++) {
        if (r < 0 || r >= rows) continue;
        out.push([r, c]);
      }
    }

    return out;
  }

  function rebuildGridStateFromSlots() {
    gridState = Array.from({ length: rows }, () => Array.from({ length: COLS }, () => null));
    const ids = Object.keys(slots);

    for (const id of ids) {
      const slot = slots[id];
      if (!slot) continue;
      const cells = computeCoverageCellsForSlot(slot);
      for (const [r,c] of cells) {
        gridState[r][c] = id;
      }
    }
  }

  // IMPORTANT: fused rendering using overlay blocks
  function renderBlocksOverlay() {
    if (!blocksLayer) return;
    blocksLayer.innerHTML = "";

    const dayW = cssVarPx("--day-col-w", 112);
    const rowH = cssVarPx("--row-h", 26);

    for (let c = 0; c < COLS; c++) {
      let r = 0;
      while (r < rows) {
        const slotId = gridState?.[r]?.[c] || null;
        if (!slotId) { r++; continue; }

        // scan contiguous run
        let r2 = r;
        while (r2 + 1 < rows && (gridState?.[r2 + 1]?.[c] || null) === slotId) {
          r2++;
        }

        const slot = slots[slotId];
        const rgb = hexToRgb(slot?.color);

        const block = document.createElement("div");
        block.className = "p-block";

        const left = c * dayW + 6; // inner padding so gridlines remain visible
        const top  = r * rowH + 2;
        const height = (r2 - r + 1) * rowH - 4;
        const width  = dayW - 12;

        block.style.left = `${left}px`;
        block.style.top = `${top}px`;
        block.style.height = `${Math.max(8, height)}px`;
        block.style.width = `${Math.max(20, width)}px`;

        // translucent color
        const alpha = getCssNumber("--slot-alpha", 0.18);
        block.style.background = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;

        // selected highlight (by day + slot)
        if (selected.slotId === slotId && selected.dayISO) {
          const dayISO = computeDayISO(c);
          if (dayISO === selected.dayISO) block.classList.add("is-selected");
        }

        blocksLayer.appendChild(block);

        r = r2 + 1;
      }
    }
  }

  function cssVarPx(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const n = Number(String(v).replace("px",""));
    return Number.isFinite(n) ? n : fallback;
  }

  function getCssNumber(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function repaintAll() {
    // cells don't carry color anymore: overlay blocks do.
    renderBlocksOverlay();
  }

  function rebuildEverything() {
    if (!startDate) startDate = getStartOfWeekMonday(todayStart());
    ensureGridSize();
    buildHeads();
    buildTimeColumn();
    buildGridDOM();
    rebuildGridStateFromSlots();
    repaintAll();
    renderSidebarSlots();
    renderSummary();
    updateWindowLabel();
  }

  // ---------------- Sidebar rendering ----------------
  function updateWindowLabel() {
    if (!windowLabel) return;
    const a = fmtLocalISODate(startDate);
    const b = new Date(startDate);
    b.setDate(b.getDate() + (COLS - 1));
    const bb = fmtLocalISODate(b);
    windowLabel.textContent = `${a} → ${bb}`;
  }

  function renderSummary() {
    if (!summaryStats) return;
    const ids = Object.keys(slots);
    const slotCount = ids.length;

    let occurrences = 0;
    for (const id of ids) {
      const s = slots[id];
      if (!s) continue;
      occurrences += occurrencesInViewForSlot(s).length;
    }

    summaryStats.innerHTML = `
      <div><span>Slots</span><span>${slotCount}</span></div>
      <div><span>Occorrenze (2w)</span><span>${occurrences}</span></div>
      <div><span>Window</span><span>${COLS} giorni</span></div>
    `;
  }

  function renderSidebarSlots() {
    if (!slotList) return;

    const ids = Object.keys(slots);
    if (!ids.length) {
      slotList.innerHTML = `<div class="hint">Nessuno slot. Aggiungilo dal Discovery.</div>`;
      return;
    }

    slotList.innerHTML = ids.map(id => {
      const s = slots[id];
      const meta = `${s.start}–${s.end}`;
      return `
        <div class="slot-pill" data-slot="${id}">
          <div class="left">
            <span class="dotc" style="background:${s.color}"></span>
            <div>
              <div class="name">${escapeHtml(s.name)}</div>
              <div class="meta">${meta}</div>
            </div>
          </div>
          <div class="meta">#</div>
        </div>
      `;
    }).join("");

    slotList.querySelectorAll(".slot-pill").forEach(el => {
      el.addEventListener("click", () => {
        const id = el.dataset.slot;
        if (!id || !slots[id]) return;
        selected.slotId = id;
        selected.dayISO = null;
        applySelectedSlotToEditor(slots[id], null);
        if (slotInfo) slotInfo.textContent = `Selezionato: ${slots[id].name} (${slots[id].start}–${slots[id].end})`;
        repaintAll();
      });
    });
  }

  function renderSidebarEmpty(msg) {
    if (slotInfo) slotInfo.textContent = msg || "—";
    if (slotPlaylistList) slotPlaylistList.innerHTML = `<div class="hint">—</div>`;
  }

  function applySelectedSlotToEditor(slot, dayISO = null) {
    try {
      if (slotColorEdit) slotColorEdit.value = slot.color || "#FFD403";
    } catch {}

    const playlist = dayISO ? getEnabledPlaylist(slot, dayISO) : [];
    const stats = computePlaylistStats(playlist);

    if (slotTracks) slotTracks.textContent = String(stats.tracks ?? "—");
    if (slotEnergy) slotEnergy.textContent = stats.energy != null ? stats.energy.toFixed(2) : "—";
    if (slotMood)   slotMood.textContent   = stats.mood != null ? stats.mood.toFixed(2) : "—";
    if (slotDance)  slotDance.textContent  = stats.danceability != null ? stats.danceability.toFixed(2) : "—";
    if (slotBpm)    slotBpm.textContent    = stats.bpm != null ? String(Math.round(stats.bpm)) : "—";

    if (slotGenres) {
      if (stats.topGenres.length) {
        slotGenres.innerHTML = stats.topGenres.map(g => `<li>${escapeHtml(g)}</li>`).join("");
      } else {
        slotGenres.innerHTML = `<li class="hint">—</li>`;
      }
    }
  }

  function escapeHtml(s) {
    return String(s || "")
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
  }

  function computePlaylistStats(playlist) {
    const items = Array.isArray(playlist) ? playlist : [];
    if (!items.length) {
      return {
        tracks: 0,
        energy: null,
        mood: null,
        danceability: null,
        bpm: null,
        topGenres: []
      };
    }

    const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

    const bpms = items
      .map(x => Number(x.bpm))
      .filter(Number.isFinite);

    const energies = items
      .map(x => Number(x.energy))
      .filter(Number.isFinite);

    const moods = items
      .map(x => Number(x.valence ?? x.mood))
      .filter(Number.isFinite);

    const dances = items
      .map(x => Number(x.danceability))
      .filter(Number.isFinite);

    const genreCount = {};
    items.forEach(x => {
      const g = safeText(x.genre ?? x.track_genre, "");
      if (!g) return;
      genreCount[g] = (genreCount[g] || 0) + 1;
    });

    const topGenres = Object.entries(genreCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([g]) => g);

    return {
      tracks: items.length,
      energy: avg(energies),
      mood: avg(moods),
      danceability: avg(dances),
      bpm: avg(bpms),
      topGenres
    };
  }


  // ---------------- Click handling ----------------

  function onCellClick(e) {
    const r = parseInt(e.currentTarget.dataset.r, 10);
    const c = parseInt(e.currentTarget.dataset.c, 10);

    const slotId = gridState?.[r]?.[c] || null;
    if (!slotId) {
      selected = { slotId: null, dayISO: null };
      repaintAll();
      renderSidebarEmpty("Cella vuota: nessuna playlist in questo orario.");
      return;
    }

    const slot = slots[slotId];
    if (!slot) {
      renderSidebarEmpty("Slot mancante (stato corrotto).");
      return;
    }

    const dayISO = computeDayISO(c);
    selected = { slotId, dayISO };
    repaintAll();

    if (slotInfo) slotInfo.textContent = `${slot.name} • ${dayISO} • ${slot.start}–${slot.end}`;

    const playlist = ensureTrackEnabledFlags(slot, dayISO);
    applySelectedSlotToEditor(slot, dayISO);

    if (!playlist.length) {
      if (slotPlaylistList) {
        slotPlaylistList.innerHTML = `<div class="hint">Nessun brano assegnato per questo giorno.</div>`;
      }
      return;
    }

    if (slotPlaylistList) {
      slotPlaylistList.innerHTML = playlist.map((x, i) => {
        const title = safeText(x.title ?? x.track_name ?? x.name, "(untitled)");
        const artist = safeText(x.artist ?? x.artists, "");
        const bpm = (x.bpm != null && x.bpm !== "")
          ? `<span class="muted"> • ${escapeHtml(String(Math.round(Number(x.bpm))))} BPM</span>`
          : "";

        const checked = x.enabled !== false ? "checked" : "";

        return `
          <label class="pl-item pl-item-checkable">
            <input
              type="checkbox"
              class="track-toggle"
              data-track-index="${i}"
              ${checked}
            >
            <span class="pl-item-text">
              ${i + 1}. ${escapeHtml(title)}${artist ? " — " + escapeHtml(artist) : ""}${bpm}
            </span>
          </label>
        `;
      }).join("");

      slotPlaylistList.querySelectorAll(".track-toggle").forEach((el) => {
        el.addEventListener("change", () => {
          const idx = Number(el.dataset.trackIndex);
          if (!Number.isInteger(idx) || !playlist[idx]) return;

          playlist[idx].enabled = !!el.checked;
          slot.playlistsByDay[dayISO] = playlist;
          save();
          applySelectedSlotToEditor(slot, dayISO);
        });
      });
    }
      }


  // ---------------- Slot editor events ----------------
  function onSlotEditorChange() {
    const slotId = selected.slotId;
    if (!slotId || !slots[slotId]) return;

    const s = slots[slotId];

    if (slotColorEdit) s.color = slotColorEdit.value || s.color;

    slots[slotId] = s;
    save();

    renderSidebarSlots();
    renderSummary();
    repaintAll();
  }

  // ---------------- Window navigation ----------------
  function shiftWindow(days) {
    const d = new Date(startDate);
    d.setDate(d.getDate() + days);
    startDate = getStartOfWeekMonday(d);
    save();
    rebuildEverything();
    renderSidebarEmpty("Finestra cambiata. Clicca una cella.");
  }

  // ---------------- Export / Publish ----------------
  function occurrencesInViewForSlot(slot) {
    const weekdays = new Set(normalizeWeekdays(slot.weekdays || [1,2,3,4,5]));
    const weeks = Math.max(1, Number(slot.weeks) || 1);
    const out = [];

    for (let c = 0; c < COLS; c++) {
      const weekIndex = Math.floor(c / 7) + 1;
      if (weekIndex > weeks) continue;

      const d = new Date(startDate);
      d.setDate(startDate.getDate() + c);

      if (!weekdays.has(d.getDay())) continue;
      out.push(computeDayISO(c));
    }

    return out;
  }

  async function exportTimetableJSON() {
    const slotIds = Object.keys(slots);
    if (!slotIds.length) {
      renderSidebarEmpty("Niente da esportare: aggiungi slot dal Discovery.");
      return;
    }

    if (slotInfo) slotInfo.textContent = "Preparo export… (genero playlist mancanti)";

    const items = [];

    for (const slotId of slotIds) {
      const slot = slots[slotId];
      if (!slot) continue;

      const days = occurrencesInViewForSlot(slot);
      if (!days.length) continue;

      // generate missing before export
      const missing = days.filter(d => !(slot.playlistsByDay?.[d]?.length));
      if (missing.length) {
        // opzionale: segnala
        setDebug({ warning: "missing_playlists", slot_id: slotId, days: missing });
      }


      for (const dayISO of days) {
        items.push({
          slot_id: slotId,
          day_iso: dayISO,
          start: slot.start,
          end: slot.end,
          name: slot.name,
          color: slot.color,
          tracks: getEnabledPlaylist(slot, dayISO),
          discovery: slot.discovery || {},
          spotify_playlist_url: null,
        });
      }
    }

    const payload = {
      version: "sisma-planner-export-v1",
      generated_at: new Date().toISOString(),
      window_start: startDate.toISOString().slice(0,10),
      window_days: COLS,
      items
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `sisma_timetable_${startDate.toISOString().slice(0,10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);

    if (slotInfo) slotInfo.textContent = `Export pronto: ${items.length} occorrenze.`;
  }

  async function commitSpotifyStub() {
    // beta behavior: call stub endpoint with current window plan
    const slotIds = Object.keys(slots);
    if (!slotIds.length) return;

    const plan = [];
    for (const slotId of slotIds) {
      const slot = slots[slotId];
      const days = occurrencesInViewForSlot(slot);
      for (const dayISO of days) {
        plan.push({
          slot_id: slotId,
          day_iso: dayISO,
          start: slot.start,
          end: slot.end,
          name: slot.name,
          color: slot.color,
          tracks: getEnabledPlaylist(slot, dayISO),
        });
      }
    }

    try {
      const res = await fetch("/planner/api/commit_plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setDebug({ commit_plan: data });
      if (slotInfo) slotInfo.textContent = "Spotify commit: stub ok (vedi Debug).";
    } catch (e) {
      setDebug({ commit_plan_error: String(e.message || e) });
      if (slotInfo) slotInfo.textContent = `Spotify commit: errore (${e.message})`;
    }
  }

  // ---------------- Scroll sync (header) ----------------
  function bindScrollSync() {
    if (!gridScroll || !daysHead) return;
    gridScroll.addEventListener("scroll", () => {
      // sync header horizontal scroll by translating it
      const x = gridScroll.scrollLeft;
      daysHead.style.transform = `translateX(${-x}px)`;
    }, { passive: true });
  }

  // ---------------- Init ----------------
  startDate = getStartOfWeekMonday(todayStart());
  ensureGridSize();

  load();

  if (!startDate) startDate = getStartOfWeekMonday(todayStart());

  rebuildEverything();
  bindScrollSync();

  // events

  if (slotColorEdit) slotColorEdit.addEventListener("input", onSlotEditorChange);


  if (btnExportPlan) btnExportPlan.addEventListener("click", exportTimetableJSON);
  if (btnDownloadTimetable) btnDownloadTimetable.addEventListener("click", exportTimetableJSON);
  if (btnCommitSpotify) btnCommitSpotify.addEventListener("click", commitSpotifyStub);

  if (btnClearPlan) btnClearPlan.addEventListener("click", clearAll);
  if (btnPrevWindow) btnPrevWindow.addEventListener("click", () => shiftWindow(-14));
  if (btnNextWindow) btnNextWindow.addEventListener("click", () => shiftWindow(+14));
})();
