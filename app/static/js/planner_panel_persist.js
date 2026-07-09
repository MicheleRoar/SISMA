// app/static/js/planner_panel_persist.js
//
// The Discovery page's "Generate 50-song playlist" button submits a GET
// form and reloads the whole page. The "Planner mode" panel fields
// (slot name, color, start/end time, weeks, track order, weekdays) have
// hardcoded HTML defaults and aren't echoed back by the server, so every
// reload silently resets them (e.g. start time back to 10:00) even if the
// user had set something else before generating a preview.
//
// This persists those fields to localStorage across reloads, independent
// of the server-rendered defaults. Must load BEFORE main.js, since
// main.js's weekday-toggle init reads #planner_weekdays synchronously.
(function () {
  const STORAGE_KEY = "sisma_discovery_planner_panel_v1";

  const FIELD_IDS = [
    "planner_slot_name",
    "planner_color",
    "planner_start",
    "planner_end",
    "planner_window_start",
    "planner_weeks",
    "planner_track_order",
    "planner_weekdays",
  ];

  function getFields() {
    const fields = {};
    for (const id of FIELD_IDS) {
      const el = document.getElementById(id);
      if (el) fields[id] = el;
    }
    return fields;
  }

  function restore() {
    let saved;
    try {
      saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch (e) {
      saved = {};
    }

    const fields = getFields();
    for (const id of Object.keys(fields)) {
      if (Object.prototype.hasOwnProperty.call(saved, id) && saved[id]) {
        fields[id].value = saved[id];
      }
    }
  }

  function persist() {
    const fields = getFields();
    const out = {};
    for (const id of Object.keys(fields)) {
      out[id] = fields[id].value;
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(out));
    } catch (e) {
      // ignore quota/security errors (e.g. private browsing)
    }
  }

  restore();

  const form = document.getElementById("playlist_form");
  if (form) {
    form.addEventListener("input", persist);
    form.addEventListener("change", persist);
  }
})();
