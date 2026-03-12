// app/static/js/discovery_to_planner.js
(function () {
  const btn = document.getElementById("btn_add_to_planner");
  const form = document.getElementById("playlist_form");
  if (!btn || !form) return;

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
      .filter((x) => Number.isFinite(x) && x >= 0 && x <= 6);

    return arr.length ? arr : [1, 2, 3, 4, 5];
  }

  function setBusy(on) {
    btn.disabled = on;
    btn.textContent = on ? "Generating..." : "Add to planner";
  }

  function showError(message) {
    const errBox = document.getElementById("planner_error");
    if (errBox) {
      errBox.style.display = "block";
      errBox.textContent = `Error: ${message}`;
    } else {
      alert(`Planner error: ${message}`);
    }
  }

  async function sendToPlanner() {
    setBusy(true);

    try {
      const name =
        (document.getElementById("planner_slot_name")?.value || "").trim() ||
        "Slot";

      const color =
        (document.getElementById("planner_color")?.value || "#77dd77").trim();

      const start =
        (document.getElementById("planner_start")?.value || "10:00").trim();

      const end =
        (document.getElementById("planner_end")?.value || "11:00").trim();

      const weeks =
        parseInt(document.getElementById("planner_weeks")?.value || "2", 10) || 2;

      const weekdays = getSelectedWeekdaysFromHidden();
      const discovery = buildDiscoveryPayloadFromForm(form);

      const payload = {
        discovery,
        rule: {
          name,
          color,
          start,
          end,
          weeks,
          weekdays,
        },
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

      const generatedPlan =
        data.plan && typeof data.plan === "object" ? data.plan : null;

      if (!generatedPlan || !generatedPlan.slots || typeof generatedPlan.slots !== "object") {
        throw new Error("Planner payload invalido: slots mancanti.");
      }

      const slotEntries = Object.entries(generatedPlan.slots);
      if (!slotEntries.length) {
        throw new Error("Nessuno slot generato dal planner.");
      }

      // In questo flusso ci aspettiamo un singolo slot generato
      const [, generatedSlot] = slotEntries[0];

      sessionStorage.setItem(
        "sisma_planner_draft",
        JSON.stringify({
          slot: {
            name,
            color,
            start,
            end,
            weeks,
            weekdays,
          },
          discovery,
          generation: {
            k: 50,
            max_per_artist: 2,
            cooldown_days: 2,
          },
          generated_slot: generatedSlot,
          generated_startDateISO: generatedPlan.startDateISO || null,
        })
      );

      window.location.href = "/planner/";
    } catch (e) {
      console.error(e);
      setBusy(false);
      showError(e.message || String(e));
    }
  }

  btn.addEventListener("click", (e) => {
    e.preventDefault();
    sendToPlanner();
  });
})();