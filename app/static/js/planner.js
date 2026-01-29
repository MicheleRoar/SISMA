// app/static/js/planner.js
(() => {
  // ----------------------------
  // Config
  // ----------------------------
  const START_DAY = "10:00";
  const END_DAY   = "24:00";
  const STEP_MIN  = 30;
  const COLS      = 14;

  // localStorage key
  const LS_KEY    = "sisma_planner_rules_v1";

  // Planner policy: avoid repeats for the previous N days (per rule)
  const COOLDOWN_DAYS = 2;

  const WEEKDAY_IT = ["Dom", "Lun", "Mar", "Mer", "Gio", "Ven", "Sab"];

  // ----------------------------
  // DOM (calendar)
  // ----------------------------
  const daysHead = document.getElementById("daysHead");
  const timeCol  = document.getElementById("timeCol");
  const daysGrid = document.getElementById("daysGrid");

  // ----------------------------
  // DOM (left controls)
  // ----------------------------
  const slotPreset = document.getElementById("slotPreset");
  const slotColor  = document.getElementById("slotColor");
  const swatchPreview = document.getElementById("swatchPreview");

  const slotStart = document.getElementById("slotStart"); // <input type="time">
  const slotEnd   = document.getElementById("slotEnd");   // <input type="time">

  const weeksCount = document.getElementById("weeksCount"); // <select> 1..8 (default 2)
  const weekdayToggles = document.getElementById("weekdayToggles"); // container buttons
  const btnGenerate = document.getElementById("btnGenerate");
  const btnResetPlan = document.getElementById("btnResetPlan");

  // ----------------------------
  // DOM (sidebar viewer)
  // ----------------------------
  const slotInfo = document.getElementById("slotInfo");
  const slotPlaylistList = document.getElementById("slotPlaylistList");

  // ----------------------------
  // State
  // ----------------------------
  let rows = 0;

  // gridState[r][c] = null | { slotId: "..." }
  let gridState = [];

  // rules map: slotId -> rule object
  // rule = { preset, color, start, end, weeks, weekdays:[...], playlistsByDay:{ [c]: tracks[] } }
  let rules = {};

  // selection (drag)
  let isMouseDown = false;
  let selection = new Set(); // set of "r,c"
  let selectionMode = null;  // "add" | "remove"

  // 14-day view start
  let startDate = null;

  // ----------------------------
  // Utils
  // ----------------------------
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

  function keyRC(r, c){ return `${r},${c}`; }

  function clampToStep(mins) {
    return Math.round(mins / STEP_MIN) * STEP_MIN;
  }

  function safeVal(el, fallback){
    if (!el) return fallback;
    const v = el.value;
    return (v === undefined || v === null || v === "") ? fallback : v;
  }

  // Stable integer seed from a string (fast hash)
  function hashSeed(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return Math.abs(h);
  }

  function save() {
    localStorage.setItem(LS_KEY, JSON.stringify({
      rows,
      startDateISO: startDate ? startDate.toISOString() : null,
      grid: gridState,
      rules
    }));
  }

  function load() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (!raw) return false;
      const obj = JSON.parse(raw);
      if (!obj || !Array.isArray(obj.grid) || typeof obj.rows !== "number") return false;

      rows = obj.rows;
      gridState = obj.grid;
      rules = obj.rules || {};
      startDate = obj.startDateISO ? new Date(obj.startDateISO) : null;

      // Back-compat: migrate old rule.playlist -> playlistsByDay[0]
      Object.keys(rules).forEach(slotId => {
        const r = rules[slotId];
        if (!r) return;
        if (!r.playlistsByDay) r.playlistsByDay = {};
        if (r.playlist && Array.isArray(r.playlist) && r.playlist.length) {
          if (!r.playlistsByDay[0] || !r.playlistsByDay[0].length) {
            r.playlistsByDay[0] = r.playlist;
          }
          delete r.playlist;
        }
      });

      return true;
    } catch {
      return false;
    }
  }

  function resetState() {
    gridState = Array.from({ length: rows }, () => Array.from({ length: COLS }, () => null));
    rules = {};
    selection.clear();
    save();
  }

  // ----------------------------
  // Rendering: heads/time/grid
  // ----------------------------
  function buildHeads() {
    if (!daysHead) return;
    daysHead.innerHTML = "";
    for (let c = 0; c < COLS; c++) {
      const d = new Date(startDate);
      d.setDate(startDate.getDate() + c);

      const el = document.createElement("div");
      el.className = "day-head";
      el.textContent = WEEKDAY_IT[d.getDay()];
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
      row.className = "time-row";
      row.textContent = minToTime(m);
      timeCol.appendChild(row);
    }
  }

  function ensureGridSize() {
    const start = timeToMin(START_DAY);
    const end   = timeToMin(END_DAY);
    rows = Math.floor((end - start) / STEP_MIN);

    if (!startDate) startDate = getStartOfWeekMonday(todayStart());

    const ok =
      Array.isArray(gridState) &&
      gridState.length === rows &&
      Array.isArray(gridState[0]) &&
      gridState[0].length === COLS;

    if (!ok) {
      gridState = Array.from({ length: rows }, () => Array.from({ length: COLS }, () => null));
    }
  }

  function buildGrid() {
    ensureGridSize();
    if (!daysGrid) return;

    daysGrid.innerHTML = "";
    const grid = document.createElement("div");
    grid.className = "grid";
    daysGrid.appendChild(grid);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < COLS; c++) {
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.dataset.r = String(r);
        cell.dataset.c = String(c);

        const entry = gridState[r][c];
        if (entry?.slotId && rules[entry.slotId]?.color) {
          cell.style.background = rules[entry.slotId].color;
        }

        if (selection.has(keyRC(r,c))) {
          cell.classList.add("selected");
        }

        cell.addEventListener("mousedown", onCellMouseDown);
        cell.addEventListener("mouseenter", onCellMouseEnter);
        cell.addEventListener("click", onCellClick);

        grid.appendChild(cell);
      }
    }
  }

  function setSwatch() {
    if (!swatchPreview || !slotColor) return;
    swatchPreview.style.background = slotColor.value;
  }

  function findCell(r, c) {
    if (!daysGrid) return null;
    return daysGrid.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
  }

  function setCellSelected(r,c,on){
    const k = keyRC(r,c);
    const cell = findCell(r,c);
    if (!cell) return;

    if (on) {
      selection.add(k);
      cell.classList.add("selected");
    } else {
      selection.delete(k);
      cell.classList.remove("selected");
    }
  }

  // ----------------------------
  // Selection (drag)
  // ----------------------------
  function onCellMouseDown(e) {
    e.preventDefault();
    const r = parseInt(e.currentTarget.dataset.r, 10);
    const c = parseInt(e.currentTarget.dataset.c, 10);

    const k = keyRC(r,c);
    const already = selection.has(k);

    selectionMode = already ? "remove" : "add";
    isMouseDown = true;
    if (daysGrid) daysGrid.classList.add("painting");

    setCellSelected(r,c, selectionMode === "add");
  }

  function onCellMouseEnter(e) {
    if (!isMouseDown) return;
    const r = parseInt(e.currentTarget.dataset.r, 10);
    const c = parseInt(e.currentTarget.dataset.c, 10);

    setCellSelected(r,c, selectionMode === "add");
  }

  function stopSelecting() {
    if (!isMouseDown) return;
    isMouseDown = false;
    selectionMode = null;
    if (daysGrid) daysGrid.classList.remove("painting");
  }

  // ----------------------------
  // Sidebar
  // ----------------------------
  function renderSidebarEmpty() {
    if (slotInfo) slotInfo.textContent = "Cella vuota: nessuna playlist associata.";
    if (slotPlaylistList) slotPlaylistList.innerHTML = "";
  }

  async function onCellClick(e) {
    if (isMouseDown) return;

    const r = parseInt(e.currentTarget.dataset.r, 10);
    const c = parseInt(e.currentTarget.dataset.c, 10);

    const entry = gridState[r][c];
    if (!entry?.slotId) {
      renderSidebarEmpty();
      return;
    }
    await renderSidebar(entry.slotId, r, c);
  }

  // ----------------------------
  // Backend fetch
  // ----------------------------
  async function fetchPlaylistFromBackend(preset, seed, k = 50, excludeTrackIds = []) {
    const res = await fetch("/planner/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        preset,
        k,
        seed,
        exclude_track_ids: excludeTrackIds
      })
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data.tracks || [];
  }

  function computeDayISO(c) {
    const d = new Date(startDate);
    d.setDate(startDate.getDate() + c);
    return d.toISOString().slice(0,10);
  }

  // Ensure playlist exists for a given slotId + column day c.
  // Generates it on demand, with cooldown exclusion for previous days.
  async function ensurePlaylistForDay(slotId, c) {
    const rule = rules[slotId];
    if (!rule) return [];

    if (!rule.playlistsByDay) rule.playlistsByDay = {};

    const existing = rule.playlistsByDay[c];
    if (existing && Array.isArray(existing) && existing.length) return existing;

    // Build exclude list from the previous N days (same rule)
    const exclude = [];
    for (let back = 1; back <= COOLDOWN_DAYS; back++) {
      const prev = rule.playlistsByDay[c - back];
      if (prev && Array.isArray(prev)) {
        for (const t of prev) {
          if (t && t.track_id) exclude.push(String(t.track_id));
        }
      }
    }

    const dayISO = computeDayISO(c);
    const seed = hashSeed(`${slotId}_${dayISO}`);

    // Generate and store
    const tracks = await fetchPlaylistFromBackend(rule.preset || "", seed, 50, exclude);
    rule.playlistsByDay[c] = tracks;

    save();
    return tracks;
  }

  async function renderSidebar(slotId, r, c) {
    const rule = rules[slotId];
    if (!rule) {
      renderSidebarEmpty();
      return;
    }

    const day = new Date(startDate);
    day.setDate(startDate.getDate() + c);
    const dayLabel = WEEKDAY_IT[day.getDay()];

    if (slotInfo) {
      slotInfo.textContent = `${dayLabel} • ${rule.preset || "Custom"} • ${rule.start}–${rule.end}`;
    }

    if (!slotPlaylistList) return;

    slotPlaylistList.innerHTML = `<div class="hint">Carico playlist…</div>`;

    let playlist = [];
    try {
      playlist = await ensurePlaylistForDay(slotId, c);
    } catch (err) {
      slotPlaylistList.innerHTML = `<div class="hint">Errore: ${err.message}</div>`;
      return;
    }

    if (!playlist.length) {
      slotPlaylistList.innerHTML = `<div class="hint">Nessun brano trovato per questo slot.</div>`;
      return;
    }

    slotPlaylistList.innerHTML = playlist.map((x, i) => {
      const artist = x.artist ? ` — ${x.artist}` : "";
      return `<div class="pl-item">${i+1}. ${x.title}${artist}</div>`;
    }).join("");
  }

  // ----------------------------
  // Rule application ("Generate")
  // ----------------------------
  function getSelectedWeekdays() {
    // Default: Mon-Fri
    const fallback = new Set([1,2,3,4,5]);
    if (!weekdayToggles) return fallback;

    const active = new Set();
    weekdayToggles.querySelectorAll("[data-wd].active").forEach(btn => {
      active.add(parseInt(btn.dataset.wd,10));
    });

    return active.size ? active : fallback;
  }

  function buildSlotId(rule) {
    // stable-ish id: preset + start-end + weekdays + weeks
    const p = (rule.preset || "custom").toLowerCase().replace(/\s+/g,"_").slice(0,40);
    const wd = Array.from(rule.weekdays).sort().join("");
    return `${p}_${rule.start.replace(":","")}_${rule.end.replace(":","")}_w${rule.weeks}_d${wd}`;
  }

  function applyRuleToGrid(rule, slotId) {
    const startMin = timeToMin(START_DAY);
    const dayStart = timeToMin(rule.start);
    const dayEnd = timeToMin(rule.end);

    const a = Math.max(startMin, clampToStep(dayStart));
    const b = Math.min(timeToMin(END_DAY), clampToStep(dayEnd));
    if (b <= a) return;

    const r0 = Math.floor((a - startMin) / STEP_MIN);
    const r1 = Math.ceil((b - startMin) / STEP_MIN) - 1;

    for (let c = 0; c < COLS; c++) {
      const d = new Date(startDate);
      d.setDate(startDate.getDate() + c);

      const weekIndex = Math.floor(c / 7) + 1;
      if (weekIndex > rule.weeks) continue;

      if (!rule.weekdays.has(d.getDay())) continue;

      for (let r = r0; r <= r1; r++) {
        if (r < 0 || r >= rows) continue;
        gridState[r][c] = { slotId };
      }
    }
  }

  function repaintFromState() {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < COLS; c++) {
        const cell = findCell(r,c);
        if (!cell) continue;

        const entry = gridState[r][c];
        if (entry?.slotId && rules[entry.slotId]?.color) {
          cell.style.background = rules[entry.slotId].color;
        } else {
          cell.style.background = "rgba(255,255,255,0.02)";
        }
      }
    }
  }

  async function onGenerate() {
    const preset = safeVal(slotPreset, "");
    const color  = safeVal(slotColor, "#77dd77");
    const start  = safeVal(slotStart, "10:00");
    const end    = safeVal(slotEnd,   "11:00");

    const weeks = parseInt(safeVal(weeksCount, "2"), 10) || 2;
    const weekdays = getSelectedWeekdays();

    const ruleForId = { preset, start, end, weeks, weekdays };
    const slotId = buildSlotId(ruleForId);

    // create/update rule
    if (!rules[slotId]) {
      rules[slotId] = {
        preset, color, start, end, weeks,
        weekdays: Array.from(weekdays),
        playlistsByDay: {}
      };
    } else {
      rules[slotId].preset = preset;
      rules[slotId].color  = color;
      rules[slotId].start  = start;
      rules[slotId].end    = end;
      rules[slotId].weeks  = weeks;
      rules[slotId].weekdays = Array.from(weekdays);
      if (!rules[slotId].playlistsByDay) rules[slotId].playlistsByDay = {};
    }

    // Apply to calendar immediately
    applyRuleToGrid({ preset, color, start, end, weeks, weekdays }, slotId);

    // Optional eager generation:
    // We DO NOT pre-generate all days here to avoid 14 backend calls.
    // Playlists are generated lazily on click (ensurePlaylistForDay).
    // But we can generate for the first matching day to avoid "empty" feeling:
    try {
      // find first day in view that matches this rule and pre-warm it
      for (let c = 0; c < COLS; c++) {
        const d = new Date(startDate);
        d.setDate(startDate.getDate() + c);

        const weekIndex = Math.floor(c / 7) + 1;
        if (weekIndex > weeks) continue;
        if (!weekdays.has(d.getDay())) continue;

        // pre-generate only one day (the first)
        await ensurePlaylistForDay(slotId, c);
        break;
      }
    } catch (err) {
      if (slotInfo) slotInfo.textContent = `Errore generazione playlist: ${err.message}`;
      // still keep the rule applied
    }

    selection.clear();
    buildGrid();
    repaintFromState();
    save();
  }

  // ----------------------------
  // Weekday toggles
  // ----------------------------
  function ensureWeekdayToggles() {
    if (!weekdayToggles) return;
    if (weekdayToggles.children.length) return;

    const order = [1,2,3,4,5,6,0]; // Monday..Sunday
    order.forEach(wd => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "wd-btn";
      b.dataset.wd = String(wd);
      b.textContent = WEEKDAY_IT[wd];

      // default active Mon-Fri
      if ([1,2,3,4,5].includes(wd)) b.classList.add("active");

      b.addEventListener("click", () => b.classList.toggle("active"));
      weekdayToggles.appendChild(b);
    });
  }

  // ----------------------------
  // Events
  // ----------------------------
  if (slotColor) slotColor.addEventListener("input", setSwatch);
  if (btnGenerate) btnGenerate.addEventListener("click", onGenerate);

  if (btnResetPlan) {
    btnResetPlan.addEventListener("click", () => {
      resetState();
      buildGrid();
      buildHeads();
      repaintFromState();
      renderSidebarEmpty();
    });
  }

  document.addEventListener("mouseup", stopSelecting);
  document.addEventListener("mouseleave", stopSelecting);

  // ----------------------------
  // Init
  // ----------------------------
  startDate = getStartOfWeekMonday(todayStart());

  ensureGridSize();
  if (load()) {
    if (!startDate) startDate = getStartOfWeekMonday(todayStart());
    ensureGridSize();
  }

  setSwatch();
  ensureWeekdayToggles();
  buildHeads();
  buildTimeColumn();
  buildGrid();
  repaintFromState();
  renderSidebarEmpty();
})();