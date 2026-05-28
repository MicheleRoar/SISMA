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

  const WEEKDAY_IT = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

  // DOM
  const daysHead = document.getElementById("daysHead");
  const timeCol  = document.getElementById("timeCol");
  const daysGrid = document.getElementById("daysGrid");
  const gridScroll = document.getElementById("gridScroll");

  const slotInfo = document.getElementById("slotInfo");
  const slotPeriod = document.getElementById("slotPeriod");
  const slotTimeInfo = document.getElementById("slotTime");
  const slotPlaylistList = document.getElementById("slotPlaylistList");
  const slotList = document.getElementById("slotList");
  const summaryStats = document.getElementById("summaryStats");

  const slotColorEdit = document.getElementById("slotColorEdit");

  const slotTracks = document.getElementById("slotTracks");
  const slotEnergy = document.getElementById("slotEnergy");
  const slotMood   = document.getElementById("slotMood");
  const slotDance  = document.getElementById("slotDance");
  const slotBpm    = document.getElementById("slotBpm");
  
  const btnSortSlotPlaylist = document.getElementById("btnSortSlotPlaylist");
  const slotSortIndicator = document.getElementById("slotSortIndicator");

  const btnLoadPlan = document.getElementById("btnLoadPlan");
  const fileLoadPlan = document.getElementById("fileLoadPlan");
  const btnClearPlan  = document.getElementById("btnClearPlan");
  const btnPrevWindow = document.getElementById("btnPrevWindow");
  const btnNextWindow = document.getElementById("btnNextWindow");
  const windowLabel   = document.getElementById("windowLabel");

  const candidatePoolPanel = document.getElementById("candidatePoolPanel");
  const candidatePoolTitle = document.getElementById("candidatePoolTitle");

  const candidateSearch = document.getElementById("candidateSearch");
  const candidateSort = document.getElementById("candidateSort");
  const candidateBpmMin = document.getElementById("candidateBpmMin");
  const candidateBpmMax = document.getElementById("candidateBpmMax");
  const candidateEnergyMin = document.getElementById("candidateEnergyMin");
  const candidateEnergyMax = document.getElementById("candidateEnergyMax");
  const candidateMoodMin = document.getElementById("candidateMoodMin");
  const candidateMoodMax = document.getElementById("candidateMoodMax");
  const candidateDanceMin = document.getElementById("candidateDanceMin");
  const candidateDanceMax = document.getElementById("candidateDanceMax");

  const candidateResultsBody = document.getElementById("candidateResultsBody");
  const candidateResultsCount = document.getElementById("candidateResultsCount");

  const btnResetCandidateFilters = document.getElementById("btnResetCandidateFilters");
  const btnHideUsedTracks = document.getElementById("btnHideUsedTracks");

  const btnDownloadTimetable = document.getElementById("btnDownloadTimetable");

  const debugBox = document.getElementById("debugBox");

  // State
  let rows = 0;
  let startDate = null; // Monday of current window
  let gridState = [];   // [rows][COLS] -> slotId|null
  let slots = {};       // slotId -> slot
  let selected = { slotId: null, dayISO: null };
  let slotPlaylistSortMode = "bpm_asc"; // bpm_asc | bpm_desc | random
  window.isSlotEditMode = false;
  let candidateRows = [];
  let hideUsedCandidates = false;

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
      startDate = sd && !Number.isNaN(sd.getTime()) ? sd : null;
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
        if (slotPeriod) slotPeriod.textContent = safeText(slots[id].name, "—");
        if (slotTimeInfo) slotTimeInfo.textContent = `${slots[id].start}–${slots[id].end}`;
        repaintAll();
      });
    });
  }

  function renderSidebarEmpty(msg) {
    if (slotPeriod) slotPeriod.textContent = "—";
    if (slotTimeInfo) slotTimeInfo.textContent = msg || "—";
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
      const raw = safeText(x.genre ?? x.track_genre, "");
      if (!raw) return;

      const genres = raw
        .split(",")
        .map(g => g.trim())
        .filter(Boolean);

      genres.forEach(g => {
        const key = g.toLowerCase();
        genreCount[key] = (genreCount[key] || 0) + 1;
      });
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

  function getTrackIdsForSelectedDay(slot, dayISO) {
    const playlist = ensureTrackEnabledFlags(slot, dayISO);
    return playlist
      .map(t => safeText(t.track_id, ""))
      .filter(Boolean);
  }

  function getAllUsedTrackIds() {
    const used = new Set();

    Object.values(slots).forEach((slot) => {
      const byDay = slot?.playlistsByDay || {};
      Object.values(byDay).forEach((playlist) => {
        (playlist || []).forEach((t) => {
          const id = safeText(t.track_id, "");
          if (id) used.add(id);
        });
      });
    });

    return used;
  }  

async function loadCandidatesForSelectedSlot() {
  const slotId = selected?.slotId || null;
  const dayISO = selected?.dayISO || null;

  if (!slotId || !dayISO || !slots[slotId]) {
    candidateRows = [];
    renderCandidateResults();
    return;
  }

  const slot = slots[slotId];
  const discovery = slot.discovery || {};
  const excludeTrackIds = Array.from(getAllUsedTrackIds());

  if (candidatePoolTitle) {
    candidatePoolTitle.textContent = `Candidates for ${safeText(slot.name, "Slot")} • ${dayISO}`;
  }

  try {
    const resp = await fetch("/planner/api/candidates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slot_id: slotId,
        day_iso: dayISO,
        discovery,
        k: 100,
        exclude_track_ids: excludeTrackIds
      })
    });

    const data = await resp.json();

    if (!resp.ok || !data?.ok) {
      throw new Error(data?.error || `HTTP ${resp.status}`);
    }

    candidateRows = Array.isArray(data.tracks) ? data.tracks : [];
    setDebug(data.report || {});
    renderCandidateResults();

  } catch (err) {
    console.error("Candidates load failed:", err);
    candidateRows = [];
    renderCandidateResults(`Unable to load candidates: ${err.message}`);
  }
}

  function renderCandidateResults(errorMsg = "") {
    if (!candidateResultsBody || !candidateResultsCount) return;

    const slotId = selected?.slotId || null;
    const dayISO = selected?.dayISO || null;
    const slot = slotId ? slots[slotId] : null;

    if (!slotId || !dayISO || !slot) {
      candidateResultsBody.innerHTML = `
        <tr class="candidate-empty-row">
          <td colspan="9">No slot selected yet.</td>
        </tr>
      `;
      candidateResultsCount.textContent = "0 candidates";
      return;
    }

    if (errorMsg) {
      candidateResultsBody.innerHTML = `
        <tr class="candidate-empty-row">
          <td colspan="9">${escapeHtml(errorMsg)}</td>
        </tr>
      `;
      candidateResultsCount.textContent = "0 candidates";
      return;
    }

    const q = safeText(candidateSearch?.value, "").toLowerCase();
    const sortBy = safeText(candidateSort?.value, "match");

    const bpmMin = candidateBpmMin?.value !== "" ? Number(candidateBpmMin.value) : null;
    const bpmMax = candidateBpmMax?.value !== "" ? Number(candidateBpmMax.value) : null;

    const energyMin = candidateEnergyMin?.value !== "" ? Number(candidateEnergyMin.value) : null;
    const energyMax = candidateEnergyMax?.value !== "" ? Number(candidateEnergyMax.value) : null;

    const moodMin = candidateMoodMin?.value !== "" ? Number(candidateMoodMin.value) : null;
    const moodMax = candidateMoodMax?.value !== "" ? Number(candidateMoodMax.value) : null;

    const danceMin = candidateDanceMin?.value !== "" ? Number(candidateDanceMin.value) : null;
    const danceMax = candidateDanceMax?.value !== "" ? Number(candidateDanceMax.value) : null;

    const usedIds = getAllUsedTrackIds();
    const dayTrackIds = new Set(getTrackIdsForSelectedDay(slot, dayISO));

    let rows = [...candidateRows];

    if (hideUsedCandidates) {
      rows = rows.filter((r) => {
        const id = safeText(r.track_id, "");
        return !usedIds.has(id);
      });
    }

    rows = rows.filter((r) => {
      const textBlob = [
        safeText(r.title, ""),
        safeText(r.artist, ""),
        safeText(r.genre, "")
      ].join(" ").toLowerCase();

      if (q && !textBlob.includes(q)) return false;

      const bpm = Number(r.bpm);
      const energy = Number(r.energy);
      const mood = Number(r.valence ?? r.mood);
      const dance = Number(r.danceability);

      if (bpmMin != null && (!Number.isFinite(bpm) || bpm < bpmMin)) return false;
      if (bpmMax != null && (!Number.isFinite(bpm) || bpm > bpmMax)) return false;

      if (energyMin != null && (!Number.isFinite(energy) || energy < energyMin)) return false;
      if (energyMax != null && (!Number.isFinite(energy) || energy > energyMax)) return false;

      if (moodMin != null && (!Number.isFinite(mood) || mood < moodMin)) return false;
      if (moodMax != null && (!Number.isFinite(mood) || mood > moodMax)) return false;

      if (danceMin != null && (!Number.isFinite(dance) || dance < danceMin)) return false;
      if (danceMax != null && (!Number.isFinite(dance) || dance > danceMax)) return false;

      return true;
    });

    rows.sort((a, b) => {
      const av = Number(a[sortBy] ?? (sortBy === "mood" ? a.valence : null));
      const bv = Number(b[sortBy] ?? (sortBy === "mood" ? b.valence : null));

      if (sortBy === "match" || sortBy === "popularity") {
        return (Number.isFinite(bv) ? bv : -Infinity) - (Number.isFinite(av) ? av : -Infinity);
      }

      return (Number.isFinite(av) ? av : Infinity) - (Number.isFinite(bv) ? bv : Infinity);
    });

    if (!rows.length) {
      candidateResultsBody.innerHTML = `
        <tr class="candidate-empty-row">
          <td colspan="9">No candidates match the current filters.</td>
        </tr>
      `;
      candidateResultsCount.textContent = "0 candidates";
      return;
    }

    candidateResultsBody.innerHTML = rows.map((row) => {
      const id = safeText(row.track_id, "");
      const alreadyInDay = dayTrackIds.has(id);

      return `
        <tr>
          <td>${escapeHtml(safeText(row.title, "—"))}</td>
          <td>${escapeHtml(safeText(row.artist, "—"))}</td>
          <td>${Number.isFinite(Number(row.bpm)) ? Math.round(Number(row.bpm)) : "—"}</td>
          <td>${Number.isFinite(Number(row.match)) ? Math.round(Number(row.match)) : "—"}</td>
          <td>${Number.isFinite(Number(row.energy)) ? Number(row.energy).toFixed(2) : "—"}</td>
          <td>${Number.isFinite(Number(row.valence ?? row.mood)) ? Number(row.valence ?? row.mood).toFixed(2) : "—"}</td>
          <td>${Number.isFinite(Number(row.danceability)) ? Number(row.danceability).toFixed(2) : "—"}</td>
          <td>${escapeHtml(safeText(row.genre, "—"))}</td>
          <td>
            ${
              alreadyInDay
                ? `<span class="hint">Added</span>`
                : `<button class="candidate-add-btn" type="button" data-track-id="${escapeHtml(id)}">+ Add</button>`
            }
          </td>
        </tr>
      `;
    }).join("");

    candidateResultsCount.textContent = `${rows.length} candidates`;

    candidateResultsBody.querySelectorAll(".candidate-add-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const trackId = safeText(btn.dataset.trackId, "");
        const picked = candidateRows.find((r) => safeText(r.track_id, "") === trackId);
        if (!picked) return;

        const playlist = ensureTrackEnabledFlags(slot, dayISO);

        if (!playlist.some((t) => safeText(t.track_id, "") === trackId)) {
          playlist.push({
            ...picked,
            enabled: true
          });

          candidateRows = candidateRows.filter((r) => safeText(r.track_id, "") !== trackId);

          slot.playlistsByDay[dayISO] = playlist;
          save();
          applySelectedSlotToEditor(slot, dayISO);
          renderSelectedSlotPlaylist();

          if (window.isSlotEditMode) {
            loadCandidatesForSelectedSlot();
          }

          renderCandidateResults();
        }
      });
    });
  }


  function getSortedSlotPlaylist(playlist) {
    const arr = Array.isArray(playlist) ? [...playlist] : [];

    if (slotPlaylistSortMode === "random") {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      return arr;
    }

    const bpmValue = (x) => {
      const n = Number(x?.bpm);
      return Number.isFinite(n) ? n : Number.POSITIVE_INFINITY;
    };

    arr.sort((a, b) => bpmValue(a) - bpmValue(b));

    if (slotPlaylistSortMode === "bpm_desc") {
      arr.reverse();
    }

    return arr;
  }

  function updateSlotSortIndicator() {
    if (!slotSortIndicator) return;

    if (slotPlaylistSortMode === "bpm_asc") {
      slotSortIndicator.textContent = "▲";
      slotSortIndicator.title = "BPM ascending";
      return;
    }

    if (slotPlaylistSortMode === "bpm_desc") {
      slotSortIndicator.textContent = "▼";
      slotSortIndicator.title = "BPM descending";
      return;
    }

    slotSortIndicator.textContent = "•";
    slotSortIndicator.title = "Random order";
  }  



  function renderSelectedSlotPlaylist() {
    const slotId = selected?.slotId || null;
    const dayISO = selected?.dayISO || null;

    if (!slotId || !dayISO || !slots[slotId]) {
      if (slotPlaylistList) {
        slotPlaylistList.innerHTML = `<div class="hint">—</div>`;
      }
      return;
    }

    const slot = slots[slotId];
    const playlist = ensureTrackEnabledFlags(slot, dayISO);
    const sortedPlaylist = getSortedSlotPlaylist(playlist);

    if (!playlist.length) {
      if (slotPlaylistList) {
        slotPlaylistList.innerHTML = `<div class="hint">No songs assigned for this day.</div>`;
      }
      return;
    }

    slotPlaylistList.innerHTML = sortedPlaylist.map((x, i) => {
      const title = safeText(x.title ?? x.track_name ?? x.name, "(untitled)");
      const artist = safeText(x.artist ?? x.artists, "");
      const bpm = (x.bpm != null && x.bpm !== "")
        ? `<span class="muted"> • ${escapeHtml(String(Math.round(Number(x.bpm))))} BPM</span>`
        : "";

      return `
        <div class="pl-item">
          <div class="pl-item-row">
            <button
              type="button"
              class="pl-remove-btn ${window.isSlotEditMode ? "" : "is-hidden"}"
              data-track-index="${i}"
              aria-label="Remove track"
              title="Remove track"
            >×</button>

            <span class="pl-item-text">
              ${i + 1}. ${escapeHtml(title)}${artist ? " — " + escapeHtml(artist) : ""}${bpm}
            </span>
          </div>
        </div>
      `;
    }).join("");

    slotPlaylistList.querySelectorAll(".pl-remove-btn").forEach((btn) => {
      btn.addEventListener("click", () => {

        const idx = Number(btn.dataset.trackIndex);
        if (!Number.isInteger(idx) || !sortedPlaylist[idx]) return;

        const picked = sortedPlaylist[idx];
        const pickedId = safeText(picked.track_id, "");
        const originalIdx = playlist.findIndex((t) => safeText(t.track_id, "") === pickedId);

        if (originalIdx < 0) return;

        playlist.splice(originalIdx, 1);
        slot.playlistsByDay[dayISO] = playlist;

        save();
        applySelectedSlotToEditor(slot, dayISO);
        renderSelectedSlotPlaylist();
      });
    });
  }

  // ---------------- Click handling ----------------

  function onCellClick(e) {
    const r = parseInt(e.currentTarget.dataset.r, 10);
    const c = parseInt(e.currentTarget.dataset.c, 10);

    const slotId = gridState?.[r]?.[c] || null;
    if (!slotId) {
      selected = { slotId: null, dayISO: null };
      repaintAll();
      renderSidebarEmpty("Empty cell: no playlist at this time.");
      return;
    }

    const slot = slots[slotId];
    if (!slot) {
      selected = { slotId: null, dayISO: null };
      repaintAll();
      renderSidebarEmpty("Missing slot (corrupted state).");
      return;
    }

    const dayISO = computeDayISO(c);
    selected = { slotId, dayISO };
    repaintAll();

    if (slotPeriod) slotPeriod.textContent = safeText(slot.name, "—");
    if (slotTimeInfo) slotTimeInfo.textContent = `${slot.start}–${slot.end}`;

    ensureTrackEnabledFlags(slot, dayISO);
    applySelectedSlotToEditor(slot, dayISO);
    renderSelectedSlotPlaylist();

    if (window.isSlotEditMode) {
      loadCandidatesForSelectedSlot();
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
    startDate = d;
    save();
    rebuildEverything();
    renderSidebarEmpty("Changed window. Click a cell.");
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
      renderSidebarEmpty("Nothing to export: add slots from Discovery.");
      return;
    }

    setDebug("Preparing JSON export...");

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
      window_start: fmtLocalISODate(startDate),
      window_days: COLS,
      items
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `sisma_timetable_${fmtLocalISODate(startDate)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);

    if (slotInfo) slotInfo.textContent = `Export ready: ${items.length} occurrencies.`;
  }


  function findConflictingSlotIds(candidateSlot, candidateSlotId = null) {
    const conflicts = new Set();
    const cells = computeCoverageCellsForSlot(candidateSlot);

    for (const [r, c] of cells) {
      const existingId = gridState?.[r]?.[c] || null;
      if (!existingId) continue;
      if (candidateSlotId && existingId === candidateSlotId) continue;
      conflicts.add(existingId);
    }

    return Array.from(conflicts);
  }

  function confirmOverwriteConflicts(conflictIds) {
    if (!conflictIds || !conflictIds.length) return true;

    const names = conflictIds
      .map(id => {
        const s = slots[id];
        if (!s) return id;
        return `${s.name} (${s.start}–${s.end})`;
      })
      .join(", ");

    return window.confirm(
      `This slot overlaps with existing slots: ${names}. Do you want to overwrite them?`
    );
  }


  function normalizeImportedTrack(track) {
    if (!track || typeof track !== "object") return null;

    return {
      ...track,
      title: safeText(track.title ?? track.track_name ?? track.name, ""),
      artist: safeText(track.artist ?? track.artists, ""),
      genre: safeText(track.genre ?? track.track_genre, ""),
      bpm: Number.isFinite(Number(track.bpm)) ? Number(track.bpm) : null,
      energy: Number.isFinite(Number(track.energy)) ? Number(track.energy) : null,
      danceability: Number.isFinite(Number(track.danceability)) ? Number(track.danceability) : null,
      valence: Number.isFinite(Number(track.valence ?? track.mood))
        ? Number(track.valence ?? track.mood)
        : null,
      enabled: typeof track.enabled === "boolean" ? track.enabled : true,
    };
  }

  function inferWeekdaysFromItems(items) {
    const set = new Set();
    (items || []).forEach(item => {
      const dayISO = safeText(item.day_iso, "");
      if (!dayISO) return;
      const d = new Date(`${dayISO}T00:00:00`);
      if (Number.isNaN(d.getTime())) return;
      set.add(d.getDay());
    });
    const out = Array.from(set).sort((a, b) => a - b);
    return out.length ? out : [1, 2, 3, 4, 5];
  }

  function inferWeeksFromItems(items, importedStartDate) {
    if (!items?.length || !importedStartDate) return 2;

    let maxOffset = 0;
    items.forEach(item => {
      const dayISO = safeText(item.day_iso, "");
      if (!dayISO) return;
      const d = new Date(`${dayISO}T00:00:00`);
      if (Number.isNaN(d.getTime())) return;

      const diffDays = Math.floor((d - importedStartDate) / (1000 * 60 * 60 * 24));
      if (diffDays > maxOffset) maxOffset = diffDays;
    });

    return Math.max(1, Math.floor(maxOffset / 7) + 1);
  }

  function buildSlotIdFromImportedItems(slotItems) {
    if (!slotItems?.length) return `slot_${Date.now()}`;

    const first = slotItems[0];
    const core = {
      name: safeText(first.name, "Slot"),
      start: safeText(first.start, "10:00"),
      end: safeText(first.end, "11:00"),
      color: safeText(first.color, "#FFD403"),
      discovery: first.discovery || {},
      days: slotItems.map(x => safeText(x.day_iso, "")).filter(Boolean).sort()
    };

    const h = hashSeed(JSON.stringify(core)).toString(16).slice(0, 10);
    return `slot_${h}`;
  }

  function importPlanFromTimetablePayload(payload) {
    if (!payload || typeof payload !== "object") {
      throw new Error("JSON not valid.");
    }

    const items = Array.isArray(payload.items) ? payload.items : null;
    if (!items || !items.length) {
      throw new Error("The file contains no items to import.");
    }

    const importedStartDateISO = safeText(payload.window_start, "");
    const importedStartDate = importedStartDateISO
      ? new Date(`${importedStartDateISO}T00:00:00`)
      : getStartOfWeekMonday(todayStart());

    if (Number.isNaN(importedStartDate.getTime())) {
      throw new Error("Invalid window_start in JSON.");
    }

    const grouped = {};
    items.forEach(item => {
      const key = [
        safeText(item.slot_id, ""),
        safeText(item.name, ""),
        safeText(item.start, ""),
        safeText(item.end, "")
      ].join("|");

      if (!grouped[key]) grouped[key] = [];
      grouped[key].push(item);
    });

    const importedSlots = {};

    Object.values(grouped).forEach((slotItems) => {
      if (!slotItems.length) return;

      const first = slotItems[0];
      const slotId = safeText(first.slot_id, "") || buildSlotIdFromImportedItems(slotItems);

      const playlistsByDay = {};
      slotItems.forEach(item => {
        const dayISO = safeText(item.day_iso, "");
        if (!dayISO) return;

        const tracks = Array.isArray(item.tracks)
          ? item.tracks.map(normalizeImportedTrack).filter(Boolean)
          : [];

        playlistsByDay[dayISO] = tracks;
      });

      importedSlots[slotId] = {
        id: slotId,
        name: safeText(first.name, "Slot"),
        color: safeText(first.color, "#FFD403"),
        start: safeText(first.start, "10:00"),
        end: safeText(first.end, "11:00"),
        weekdays: inferWeekdaysFromItems(slotItems),
        weeks: inferWeeksFromItems(slotItems, importedStartDate),
        discovery: first.discovery || {},
        playlistsByDay,
        k: DEFAULT_K,
        max_per_artist: DEFAULT_MAX_PER_ARTIST,
        cooldown_days: DEFAULT_COOLDOWN_DAYS,
      };
    });

    startDate = getStartOfWeekMonday(importedStartDate);
    slots = importedSlots;
    selected = { slotId: null, dayISO: null };

    save();
    rebuildEverything();
    renderSidebarEmpty("Planner caricato dal JSON.");
    setDebug({
      imported_slots: Object.keys(importedSlots).length,
      imported_items: items.length,
      window_start: fmtLocalISODate(startDate)
    });
  }

  async function handleLoadPlanFile(file) {
    if (!file) return;

    const text = await file.text();
    let payload;

    try {
      payload = JSON.parse(text);
    } catch {
      window.alert("The selected file is not valid JSON.");
      return;
    }

    try {
      importPlanFromTimetablePayload(payload);
    } catch (err) {
      console.error(err);
      window.alert(err?.message || "Unable to load planner from this file.");
    }
  }


  function consumePlannerDraftFromSession() {
    const raw = sessionStorage.getItem("sisma_planner_draft");
    if (!raw) return false;

    let draft;
    try {
      draft = JSON.parse(raw);
    } catch (e) {
      console.error("Invalid sisma_planner_draft:", e);
      sessionStorage.removeItem("sisma_planner_draft");
      return false;
    }

    const slotDraft = draft?.slot || {};
    const discoveryDraft = draft?.discovery || {};
    const generationDraft = draft?.generation || {};
    const generatedSlot = draft?.generated_slot || null;
    const generatedStartDateISO = draft?.generated_startDateISO || null;

    if (generatedStartDateISO) {
      const parsed = new Date(`${generatedStartDateISO}T00:00:00`);
      if (!Number.isNaN(parsed.getTime())) {
        startDate = parsed;
      }
    }

    const newSlot =
      generatedSlot && typeof generatedSlot === "object"
        ? {
            ...generatedSlot,
            id: null,
            name: safeText(generatedSlot.name ?? slotDraft.name, "Slot"),
            color: safeText(generatedSlot.color ?? slotDraft.color, "#FFD403"),
            start: safeText(generatedSlot.start ?? slotDraft.start, "10:00"),
            end: safeText(generatedSlot.end ?? slotDraft.end, "11:00"),
            weekdays: normalizeWeekdays(generatedSlot.weekdays ?? slotDraft.weekdays),
            weeks: Math.max(1, Number(generatedSlot.weeks ?? slotDraft.weeks) || 2),

            discovery: generatedSlot.discovery || discoveryDraft || {},

            playlistsByDay:
              generatedSlot.playlistsByDay &&
              typeof generatedSlot.playlistsByDay === "object"
                ? generatedSlot.playlistsByDay
                : {},

            k: Number.isFinite(Number(generatedSlot.k))
              ? Number(generatedSlot.k)
              : (Number.isFinite(Number(generationDraft.k))
                  ? Number(generationDraft.k)
                  : DEFAULT_K),

            max_per_artist: Number.isFinite(Number(generatedSlot.max_per_artist))
              ? Number(generatedSlot.max_per_artist)
              : (Number.isFinite(Number(generationDraft.max_per_artist))
                  ? Number(generationDraft.max_per_artist)
                  : DEFAULT_MAX_PER_ARTIST),

            cooldown_days: Number.isFinite(Number(generatedSlot.cooldown_days))
              ? Number(generatedSlot.cooldown_days)
              : (Number.isFinite(Number(generationDraft.cooldown_days))
                  ? Number(generationDraft.cooldown_days)
                  : DEFAULT_COOLDOWN_DAYS),
          }
        : {
            id: null,
            name: safeText(slotDraft.name, "Slot"),
            color: safeText(slotDraft.color, "#FFD403"),
            start: safeText(slotDraft.start, "10:00"),
            end: safeText(slotDraft.end, "11:00"),
            weekdays: normalizeWeekdays(slotDraft.weekdays),
            weeks: Math.max(1, Number(slotDraft.weeks) || 2),

            discovery: discoveryDraft || {},
            playlistsByDay: {},

            k: Number.isFinite(Number(generationDraft.k))
              ? Number(generationDraft.k)
              : DEFAULT_K,

            max_per_artist: Number.isFinite(Number(generationDraft.max_per_artist))
              ? Number(generationDraft.max_per_artist)
              : DEFAULT_MAX_PER_ARTIST,

            cooldown_days: Number.isFinite(Number(generationDraft.cooldown_days))
              ? Number(generationDraft.cooldown_days)
              : DEFAULT_COOLDOWN_DAYS,
          };

    const slotId = buildSlotId(newSlot, newSlot.discovery || discoveryDraft);
    newSlot.id = slotId;

    rebuildGridStateFromSlots();

    const conflictIds = findConflictingSlotIds(newSlot, slotId);
    if (conflictIds.length) {
      const ok = confirmOverwriteConflicts(conflictIds);
      if (!ok) {
        sessionStorage.removeItem("sisma_planner_draft");
        return false;
      }

      conflictIds.forEach((id) => {
        delete slots[id];
      });

      rebuildGridStateFromSlots();
    }

    slots[slotId] = newSlot;
    save();
    sessionStorage.removeItem("sisma_planner_draft");
    return true;
  }

  function initCandidatePoolEditMode() {
    const btnEdit = document.getElementById("btnEditSlotPlaylist");
    const panel = document.getElementById("candidatePoolPanel");
    const title = document.getElementById("candidatePoolTitle");
    const subtitle = document.getElementById("candidatePoolSubtitle");

    if (!btnEdit || !panel) return;

    btnEdit.addEventListener("click", () => {
      const isHidden = panel.classList.contains("is-hidden");

      if (isHidden) {
        panel.classList.remove("is-hidden");
        window.isSlotEditMode = true;

        if (title) {
          title.textContent = "Select a slot to inspect candidates";
        }

        if (subtitle) {
          subtitle.textContent = "Click a slot in the calendar to inspect candidates and refine the playlist manually.";
        }

        btnEdit.textContent = "Close";
      } else {
        panel.classList.add("is-hidden");
        window.isSlotEditMode = false;
        btnEdit.textContent = "Edit";
      }

      renderSelectedSlotPlaylist();

      if (window.isSlotEditMode) {
        loadCandidatesForSelectedSlot();
      } else {
        candidateRows = [];
        renderCandidateResults();
      }

    });
  }

  function initSlotPlaylistSort() {
    if (!btnSortSlotPlaylist) return;

    updateSlotSortIndicator();

    btnSortSlotPlaylist.addEventListener("click", () => {
      if (slotPlaylistSortMode === "bpm_asc") {
        slotPlaylistSortMode = "bpm_desc";
      } else if (slotPlaylistSortMode === "bpm_desc") {
        slotPlaylistSortMode = "random";
      } else {
        slotPlaylistSortMode = "bpm_asc";
      }

      updateSlotSortIndicator();
      renderSelectedSlotPlaylist();
    });
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

  // import any draft coming from Discovery
  consumePlannerDraftFromSession();

  rebuildEverything();
  bindScrollSync();
  initCandidatePoolEditMode();
  initSlotPlaylistSort();

  // events
  if (slotColorEdit) slotColorEdit.addEventListener("input", onSlotEditorChange);

  if (btnLoadPlan && fileLoadPlan) {
    btnLoadPlan.addEventListener("click", () => fileLoadPlan.click());

    fileLoadPlan.addEventListener("change", async (e) => {
      const file = e.target.files?.[0] || null;
      await handleLoadPlanFile(file);

      // reset value so same file can be selected again later
      e.target.value = "";
    });
  }

  if (btnDownloadTimetable) btnDownloadTimetable.addEventListener("click", exportTimetableJSON);
  // if (btnCommitSpotify) btnCommitSpotify.addEventListener("click", commitSpotifyStub);

  if (btnClearPlan) btnClearPlan.addEventListener("click", clearAll);
  if (btnPrevWindow) btnPrevWindow.addEventListener("click", () => shiftWindow(-14));
  if (btnNextWindow) btnNextWindow.addEventListener("click", () => shiftWindow(+14));

  [
    candidateSearch,
    candidateSort,
    candidateBpmMin,
    candidateBpmMax,
    candidateEnergyMin,
    candidateEnergyMax,
    candidateMoodMin,
    candidateMoodMax,
    candidateDanceMin,
    candidateDanceMax
  ].forEach((el) => {
    if (!el) return;
    el.addEventListener("input", () => renderCandidateResults());
    el.addEventListener("change", () => renderCandidateResults());
  });

  if (btnResetCandidateFilters) {
    btnResetCandidateFilters.addEventListener("click", () => {
      if (candidateSearch) candidateSearch.value = "";
      if (candidateSort) candidateSort.value = "match";

      if (candidateBpmMin) candidateBpmMin.value = "";
      if (candidateBpmMax) candidateBpmMax.value = "";

      if (candidateEnergyMin) candidateEnergyMin.value = "";
      if (candidateEnergyMax) candidateEnergyMax.value = "";

      if (candidateMoodMin) candidateMoodMin.value = "";
      if (candidateMoodMax) candidateMoodMax.value = "";

      if (candidateDanceMin) candidateDanceMin.value = "";
      if (candidateDanceMax) candidateDanceMax.value = "";

      renderCandidateResults();
    });
  }

  if (btnHideUsedTracks) {
    btnHideUsedTracks.addEventListener("click", () => {
      hideUsedCandidates = !hideUsedCandidates;
      btnHideUsedTracks.textContent = hideUsedCandidates ? "Show used" : "Hide used";
      renderCandidateResults();
    });
  }
})();

