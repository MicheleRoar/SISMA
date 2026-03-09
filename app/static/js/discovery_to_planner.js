// app/static/js/discovery_to_planner.js
(function () {
  const btn = document.getElementById("btn_add_to_planner");
  const form = document.getElementById("playlist_form");
  if (!btn || !form) return;

  const LS_PLAN_KEY = "sisma_planner_plan_v1";

  function loadStoredPlan() {
    try {
      const raw = localStorage.getItem(LS_PLAN_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return null;
      return parsed;
    } catch (e) {
      console.warn("Invalid stored planner JSON:", e);
      return null;
    }
  }

  function saveStoredPlan(plan) {
    localStorage.setItem(LS_PLAN_KEY, JSON.stringify(plan));
  }

  function mergePlans(existingPlan, incomingPlan) {
    const base =
      existingPlan && typeof existingPlan === "object"
        ? existingPlan
        : { version: 1, startDateISO: null, slots: {}, report: {} };

    const incoming =
      incomingPlan && typeof incomingPlan === "object"
        ? incomingPlan
        : { version: 1, startDateISO: null, slots: {}, report: {} };

    if (!base.version) base.version = 1;
    if (!base.slots || typeof base.slots !== "object") base.slots = {};
    if (!base.report || typeof base.report !== "object") base.report = {};

    // Mantieni la finestra del piano già esistente.
    // Se non esiste ancora, usa quella in arrivo.
    if (!base.startDateISO && incoming.startDateISO) {
      base.startDateISO = incoming.startDateISO;
    }

    const incomingSlots =
      incoming.slots && typeof incoming.slots === "object"
        ? incoming.slots
        : {};

    for (const [slotId, slot] of Object.entries(incomingSlots)) {
      base.slots[slotId] = slot;
    }

    if (incoming.report && typeof incoming.report === "object") {
      base.report = incoming.report;
    }

    return base;
  }

  function buildDiscoveryPayloadFromForm(formEl) {
    const fd = new FormData(formEl);
    const obj = {};
    const EXCLUDE_PREFIXES = ["planner_"];
    const EXCLUDE_KEYS = new Set(["csrf_token", "submit", "btn", "btn_reset"]);

    for (const [k, v] of fd.entries()) {
      const key = String(k);
      if (EXCLUDE_KEYS.has(key)) continue;
      if (EXCLUDE_PREFIXES.some((p) => key.startsWith(p))) continue;

      const val = typeof v === "string" ? v.trim() : v;
      if (val === "" || val == null) continue;

      if (obj[key] == null) obj[key] = val;
      else if (Array.isArray(obj[key])) obj[key].push(val);
      else obj[key] = [obj[key], val];
    }

    return obj;
  }

  function getSelectedWeekdaysFromHidden() {
    const h = document.getElementById("planner_weekdays");
    const raw = h ? String(h.value || "") : "";
    const arr = raw
      .split(",")
      .map((x) => parseInt(x.trim(), 10))
      .filter(Number.isFinite);

    return arr.length ? arr : [1, 2, 3, 4, 5];
  }

  function setBusy(on) {
    btn.disabled = on;
    btn.textContent = on ? "Generating…" : "Add to planner";
  }

  async function sendToPlanner() {
    setBusy(true);

    try {
      const name =
        (document.getElementById("planner_slot_name")?.value || "").trim() ||
        "Slot";
      const color = (
        document.getElementById("planner_color")?.value || "#77dd77"
      ).trim();
      const start = (
        document.getElementById("planner_start")?.value || "10:00"
      ).trim();
      const end = (
        document.getElementById("planner_end")?.value || "11:00"
      ).trim();
      const weeks =
        parseInt(
          document.getElementById("planner_weeks")?.value || "2",
          10
        ) || 2;
      const weekdays = getSelectedWeekdaysFromHidden();

      const discovery = buildDiscoveryPayloadFromForm(form);

      const payload = {
        discovery,
        rule: { name, color, start, end, weeks, weekdays },
        k: 50,
        max_per_artist: 2,
        cooldown_days: 2,
      };

      const res = await fetch("/planner/api/prepare_plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const existingPlan = loadStoredPlan();
      const mergedPlan = mergePlans(existingPlan, data.plan);

      saveStoredPlan(mergedPlan);
      window.location.href = "/planner/";
    } catch (e) {
      console.error(e);
      setBusy(false);

      const errBox = document.getElementById("planner_error");
      if (errBox) {
        errBox.style.display = "block";
        errBox.textContent = `Error: ${e.message}`;
      } else {
        alert(`Planner error: ${e.message}`);
      }
    }
  }

  btn.addEventListener("click", (e) => {
    e.preventDefault();
    sendToPlanner();
  });
})();