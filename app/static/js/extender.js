// app/static/js/extender.js
(function () {
  const tokenInput = document.getElementById("extenderToken");
  const includeAllPlaylistsInput = document.getElementById("extenderIncludeAllPlaylists");
  const btnFetch = document.getElementById("btnExtenderFetch");
  const fetchStatus = document.getElementById("extenderFetchStatus");
  const summaryBox = document.getElementById("extenderSummary");

  const resultsSection = document.getElementById("extenderResults");
  const tracksBody = document.getElementById("extenderTracksBody");
  const artistsBody = document.getElementById("extenderArtistsBody");
  const tracksCountEl = document.getElementById("extenderTracksCount");
  const artistsCountEl = document.getElementById("extenderArtistsCount");
  const tracksSelectAll = document.getElementById("extenderTracksSelectAll");
  const artistsSelectAll = document.getElementById("extenderArtistsSelectAll");

  const btnApply = document.getElementById("btnExtenderApply");
  const applyStatus = document.getElementById("extenderApplyStatus");

  if (!btnFetch) return;

  // Keep the full row objects (as returned by /api/fetch) keyed by id,
  // so "Apply" can resend exactly what the backend needs without re-fetching.
  let tracksById = {};
  let artistsById = {};

  function setStatus(el, text, kind) {
    if (!el) return;
    el.textContent = text || "";
    el.className = "extender-status" + (kind ? ` extender-status-${kind}` : "");
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));
  }

  function joinList(v) {
    if (Array.isArray(v)) return v.join(", ");
    return v == null ? "" : String(v);
  }

  function parseGenresInput(raw) {
    return String(raw || "")
      .split(",")
      .map((g) => g.trim())
      .filter((g) => g.length > 0);
  }

  function renderTracksTable(rows) {
    tracksById = {};
    tracksCountEl.textContent = String(rows.length);

    if (!rows.length) {
      tracksBody.innerHTML = `<tr class="extender-empty-row"><td colspan="6">Nessuna traccia nuova trovata.</td></tr>`;
      return;
    }

    const html = rows.map((t) => {
      tracksById[t.id] = t;
      const invisible = !!t.invisible;
      const note = invisible
        ? `<span class="extender-badge extender-badge-warn" title="Nessun genere trovato per gli artisti: la traccia non comparirà in SISMA finché non avrà un genere.">⚠ nessun genere</span>`
        : "";

      return `
        <tr data-id="${escapeHtml(t.id)}" data-kind="track">
          <td class="col-check"><input type="checkbox" class="extender-row-check" ${invisible ? "" : "checked"} /></td>
          <td>${escapeHtml(t.name)}</td>
          <td>${escapeHtml(joinList(t.artists))}</td>
          <td>${escapeHtml(t.release_date)}</td>
          <td>${escapeHtml(t.popularity)}</td>
          <td>${note}</td>
        </tr>
      `;
    }).join("");

    tracksBody.innerHTML = html;
  }

  function renderArtistsTable(rows) {
    artistsById = {};
    artistsCountEl.textContent = String(rows.length);

    if (!rows.length) {
      artistsBody.innerHTML = `<tr class="extender-empty-row"><td colspan="5">Nessun artista nuovo trovato.</td></tr>`;
      return;
    }

    const html = rows.map((a) => {
      artistsById[a.id] = a;
      const hasGenres = Array.isArray(a.genres) && a.genres.length > 0;
      return `
        <tr data-id="${escapeHtml(a.id)}" data-kind="artist">
          <td class="col-check"><input type="checkbox" class="extender-row-check" checked /></td>
          <td>${escapeHtml(a.name)}</td>
          <td>
            <input
              type="text"
              class="extender-genre-input ${hasGenres ? "" : "extender-genre-input-empty"}"
              value="${escapeHtml(joinList(a.genres))}"
              placeholder="nessun genere - aggiungine uno"
            />
          </td>
          <td>${escapeHtml(a.popularity)}</td>
          <td>${escapeHtml(Math.round(a.followers || 0))}</td>
        </tr>
      `;
    }).join("");

    artistsBody.innerHTML = html;
  }

  function renderSummary(summary) {
    const parts = [
      `Playlist trovate: <strong>${summary.playlists}</strong>`,
      `Tracce trovate: <strong>${summary.tracks_fetched}</strong> (${summary.tracks_skipped_existing} già presenti, ${summary.tracks_new} nuove)`,
      `Artisti trovati: <strong>${summary.artists_fetched}</strong> (${summary.artists_skipped_existing} già presenti, ${summary.artists_new} nuovi)`,
    ];

    if (summary.invisible_count) {
      parts.push(`<span class="extender-badge extender-badge-warn">${summary.invisible_count} nuove tracce senza genere (non visibili in SISMA finché non hanno un genere)</span>`);
    }

    if (summary.audio_features_warning) {
      parts.push(`<span class="extender-badge extender-badge-warn">${escapeHtml(summary.audio_features_warning)}</span>`);
    }

    summaryBox.innerHTML = parts.join("<br/>");
    summaryBox.classList.remove("is-hidden");
  }

  async function doFetch() {
    const token = (tokenInput.value || "").trim();
    if (!token) {
      setStatus(fetchStatus, "Incolla un token prima di continuare.", "error");
      return;
    }

    btnFetch.disabled = true;
    btnApply.disabled = true;
    setStatus(fetchStatus, "Fetching da Spotify...");
    summaryBox.classList.add("is-hidden");
    resultsSection.classList.add("is-hidden");

    try {
      const res = await fetch("/extender/api/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          token,
          include_all_playlists: includeAllPlaylistsInput ? includeAllPlaylistsInput.checked : false,
        }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      renderTracksTable(data.new_tracks || []);
      renderArtistsTable(data.new_artists || []);
      renderSummary(data.summary || {});

      resultsSection.classList.remove("is-hidden");
      setStatus(fetchStatus, "Fatto.", "ok");
      btnApply.disabled = false;
    } catch (e) {
      console.error(e);
      setStatus(fetchStatus, `Errore: ${e.message || e}`, "error");
    } finally {
      btnFetch.disabled = false;
    }
  }

  function bindSelectAll(selectAllEl, tbody) {
    if (!selectAllEl) return;
    selectAllEl.addEventListener("change", () => {
      tbody.querySelectorAll(".extender-row-check").forEach((cb) => {
        cb.checked = selectAllEl.checked;
      });
    });
  }

  bindSelectAll(tracksSelectAll, tracksBody);
  bindSelectAll(artistsSelectAll, artistsBody);

  document.querySelectorAll("[data-select]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const kind = btn.dataset.select;
      const mode = btn.dataset.mode;
      const tbody = kind === "tracks" ? tracksBody : artistsBody;
      tbody.querySelectorAll(".extender-row-check").forEach((cb) => {
        cb.checked = (mode === "all");
      });
    });
  });

  function collectSelected(tbody, byId) {
    const out = [];
    tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
      const cb = tr.querySelector(".extender-row-check");
      if (cb && cb.checked) {
        const row = byId[tr.dataset.id];
        if (row) out.push(row);
      }
    });
    return out;
  }

  function collectSelectedArtists(tbody, byId) {
    const out = [];
    tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
      const cb = tr.querySelector(".extender-row-check");
      if (!cb || !cb.checked) return;

      const row = byId[tr.dataset.id];
      if (!row) return;

      // genres may have been edited/added by hand since the fetch, so read
      // the live input value rather than the original fetched row.
      const genreInput = tr.querySelector(".extender-genre-input");
      const genres = genreInput ? parseGenresInput(genreInput.value) : row.genres;

      out.push({ ...row, genres });
    });
    return out;
  }

  async function doApply() {
    const selectedTracks = collectSelected(tracksBody, tracksById);
    const selectedArtists = collectSelectedArtists(artistsBody, artistsById);

    if (!selectedTracks.length && !selectedArtists.length) {
      setStatus(applyStatus, "Seleziona almeno una riga.", "error");
      return;
    }

    btnApply.disabled = true;
    setStatus(applyStatus, "Applico al dataset...");

    try {
      const res = await fetch("/extender/api/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tracks: selectedTracks, artists: selectedArtists }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      setStatus(
        applyStatus,
        `Fatto — tracce: ${data.tracks_before} → ${data.tracks_after}, ` +
        `artisti: ${data.artists_before} → ${data.artists_after}. ` +
        `Backup in ${data.backup_dir}. Riavvia l'app (python run.py) per vedere i nuovi dati in Discovery/Planner.`,
        "ok"
      );
    } catch (e) {
      console.error(e);
      setStatus(applyStatus, `Errore: ${e.message || e}`, "error");
      btnApply.disabled = false;
    }
  }

  btnFetch.addEventListener("click", doFetch);
  btnApply.addEventListener("click", doApply);
})();
