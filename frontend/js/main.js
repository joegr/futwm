/**
 * main.js — application entry-point, data flow, and transport controls.
 */

/* global PitchVis, Charts, d3 */

const App = (() => {
  const API = "";  // same origin

  // ── state ─────────────────────────────────────────────────────────────
  let events       = [];
  let teams        = [];
  let pitchMeta    = null;
  let currentIndex = 0;
  let playTimer    = null;
  let heatmapData  = null;

  // ── DOM refs ──────────────────────────────────────────────────────────
  const $slider       = document.getElementById("slider");
  const $counter      = document.getElementById("event-counter");
  const $matchLabel   = document.getElementById("match-label");
  const $eventInfo    = document.getElementById("event-info");
  const $eventList    = document.getElementById("event-list");
  const $predDetail   = document.getElementById("prediction-detail");
  const $teamFilter   = document.getElementById("team-filter");
  const $passnetTeam  = document.getElementById("passnet-team");
  const $speedSelect  = document.getElementById("speed-select");

  // ── boot ──────────────────────────────────────────────────────────────
  async function boot() {
    PitchVis.init("pitch-container");

    // fetch pitch geometry
    try {
      const r = await fetch(`${API}/api/pitch`);
      pitchMeta = await r.json();
      PitchVis.setPitchData(pitchMeta);
    } catch (_) {
      pitchMeta = { length: 105, width: 68, goal_width: 7.32, penalty_depth: 16.5, goal_area_depth: 5.5, centre_radius: 9.15 };
    }

    _bindControls();
  }

  // ── data loading ──────────────────────────────────────────────────────
  async function loadFromUpload(file) {
    const form = new FormData();
    form.append("file", file);
    const r = await fetch(`${API}/api/upload`, { method: "POST", body: form });
    const data = await r.json();
    if (data.error) { alert(data.error); return; }
    _ingestMatch(data);
  }

  async function loadSample() {
    // fetch sample CSV text and POST it
    try {
      const csvResp = await fetch("/examples/sample_match.csv");
      if (!csvResp.ok) throw new Error("not served");
      const csvText = await csvResp.text();
      const r = await fetch(`${API}/api/upload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ csv_text: csvText }),
      });
      const data = await r.json();
      if (data.error) { alert(data.error); return; }
      _ingestMatch(data);
    } catch (e) {
      // fallback: try to load the sample from disk via the examples static route
      alert("Could not load sample: " + e.message);
    }
  }

  function _ingestMatch(data) {
    events = data.events || [];
    teams  = data.teams  || [];
    currentIndex = 0;

    $matchLabel.textContent = `${teams[0] || "?"} vs ${teams[1] || "?"} — ${events.length} events`;

    // slider
    $slider.max   = Math.max(events.length - 1, 0);
    $slider.value = 0;

    // team dropdowns
    _populateTeamDropdowns();

    // render initial frame
    _renderFrame();
    _buildEventList();
    _fetchXgTimeline();
    _fetchPassNetwork();
  }

  function _populateTeamDropdowns() {
    [$teamFilter, $passnetTeam].forEach(sel => {
      sel.innerHTML = "";
      if (sel === $teamFilter) {
        const o = document.createElement("option");
        o.value = ""; o.textContent = "All teams"; sel.appendChild(o);
      }
      teams.forEach(t => {
        const o = document.createElement("option");
        o.value = t; o.textContent = t; sel.appendChild(o);
      });
    });
  }

  // ── rendering ─────────────────────────────────────────────────────────
  function _renderFrame() {
    if (events.length === 0) return;

    const opts = {
      passes: document.getElementById("chk-passes").checked,
      shots:  document.getElementById("chk-shots").checked,
      team:   $teamFilter.value,
      onEventClick: goToEvent,
    };
    PitchVis.renderEvents(events, currentIndex, opts);

    $counter.textContent = `${currentIndex + 1} / ${events.length}`;
    $slider.value = currentIndex;

    _renderEventInfo(events[currentIndex]);
    _highlightEventRow(currentIndex);

    // predictions
    if (document.getElementById("chk-predictions").checked) {
      _fetchPrediction(currentIndex);
    } else {
      PitchVis.clearPredictions();
      Charts.renderPredictionBars("prediction-chart", null);
      $predDetail.textContent = "";
    }

    // heatmap
    if (document.getElementById("chk-heatmap").checked) {
      _fetchHeatmap();
    } else {
      PitchVis.clearHeatmap();
    }
  }

  function _renderEventInfo(ev) {
    if (!ev) { $eventInfo.textContent = "—"; return; }
    const min = Math.floor(ev.timestamp / 60);
    const sec = Math.floor(ev.timestamp % 60);
    let html = `<span class="ev-label">${ev.event_type.replace("_", " ")}</span>`;
    html += ` <span class="ev-muted">${min}′${String(sec).padStart(2, "0")}″</span><br/>`;
    html += `<b>${ev.player}</b> (${ev.team})`;
    html += `<br/><span class="ev-muted">Pos: (${ev.x.toFixed(1)}, ${ev.y.toFixed(1)})</span>`;
    if (ev.to_player)   html += `<br/>→ ${ev.to_player}`;
    if (ev.outcome)     html += `<br/>Outcome: <b>${ev.outcome}</b>`;
    if (ev.xg != null)  html += `<br/>xG: <b>${ev.xg.toFixed(2)}</b>`;
    if (ev.pass_type)   html += `<br/>Type: ${ev.pass_type}`;
    $eventInfo.innerHTML = html;
  }

  // ── event list ────────────────────────────────────────────────────────
  function _buildEventList() {
    $eventList.innerHTML = "";
    events.forEach((ev, i) => {
      const row = document.createElement("div");
      row.className = "ev-row";
      row.dataset.index = i;

      const min = Math.floor(ev.timestamp / 60);
      const sec = Math.floor(ev.timestamp % 60);
      const teamClass = (ev.team === teams[0]) ? "home" : "away";

      let detail = ev.player;
      if (ev.to_player) detail += ` → ${ev.to_player}`;
      if (ev.outcome) detail += ` [${ev.outcome}]`;

      row.innerHTML =
        `<span class="ev-time">${min}′${String(sec).padStart(2, "0")}″</span>` +
        `<span class="ev-team ${teamClass}">${ev.team}</span>` +
        `<span class="ev-type">${ev.event_type.replace("_", " ")}</span>` +
        `<span class="ev-detail">${detail}</span>`;

      row.addEventListener("click", () => goToEvent(i));
      $eventList.appendChild(row);
    });
  }

  function _highlightEventRow(idx) {
    $eventList.querySelectorAll(".ev-row").forEach(r => r.classList.remove("active"));
    const row = $eventList.querySelector(`.ev-row[data-index="${idx}"]`);
    if (row) {
      row.classList.add("active");
      row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  // ── API fetchers ──────────────────────────────────────────────────────
  async function _fetchPrediction(idx) {
    try {
      const r = await fetch(`${API}/api/predict/${idx}`);
      const data = await r.json();
      if (data.error) return;

      Charts.renderPredictionBars("prediction-chart", data.full_distribution);
      PitchVis.renderPredictions(data, events[idx]);

      $predDetail.textContent = `Expected xG: ${data.expected_xg.toFixed(4)}`;
    } catch (_) { /* server may be down */ }
  }

  async function _fetchXgTimeline() {
    try {
      const r = await fetch(`${API}/api/xg_timeline`);
      const data = await r.json();
      if (data.error) return;
      Charts.renderXgTimeline("xg-chart", data.timeline, data.teams);
    } catch (_) {}
  }

  async function _fetchPassNetwork() {
    try {
      const team = $passnetTeam.value || teams[0] || "";
      const r = await fetch(`${API}/api/pass_network?team=${encodeURIComponent(team)}`);
      const data = await r.json();
      if (data.error) return;
      Charts.renderPassNetwork(
        "passnet-chart", data.nodes, data.links,
        pitchMeta.length, pitchMeta.width,
      );
    } catch (_) {}
  }

  async function _fetchHeatmap() {
    try {
      const team = $teamFilter.value;
      const r = await fetch(`${API}/api/heatmap?team=${encodeURIComponent(team)}`);
      const data = await r.json();
      if (data.error) return;
      const maxC = d3.max(data.cells, d => d.count) || 1;
      PitchVis.renderHeatmap(data.cells, maxC);
    } catch (_) {}
  }

  // ── transport ─────────────────────────────────────────────────────────
  function goToEvent(idx) {
    currentIndex = Math.max(0, Math.min(events.length - 1, idx));
    _renderFrame();
  }

  function play() {
    if (playTimer) { pause(); return; }
    const speed = parseInt($speedSelect.value, 10) || 500;
    document.getElementById("btn-play").textContent = "⏸";
    playTimer = setInterval(() => {
      if (currentIndex >= events.length - 1) { pause(); return; }
      currentIndex++;
      _renderFrame();
    }, speed);
  }

  function pause() {
    clearInterval(playTimer);
    playTimer = null;
    document.getElementById("btn-play").textContent = "▶";
  }

  // ── bind controls ─────────────────────────────────────────────────────
  function _bindControls() {
    document.getElementById("csv-upload").addEventListener("change", e => {
      if (e.target.files[0]) loadFromUpload(e.target.files[0]);
    });
    document.getElementById("btn-load-sample").addEventListener("click", loadSample);

    document.getElementById("btn-start").addEventListener("click", () => goToEvent(0));
    document.getElementById("btn-prev").addEventListener("click",  () => goToEvent(currentIndex - 1));
    document.getElementById("btn-play").addEventListener("click",  play);
    document.getElementById("btn-next").addEventListener("click",  () => goToEvent(currentIndex + 1));
    document.getElementById("btn-end").addEventListener("click",   () => goToEvent(events.length - 1));

    $slider.addEventListener("input", () => goToEvent(parseInt($slider.value, 10)));

    $speedSelect.addEventListener("change", () => { if (playTimer) { pause(); play(); } });

    // overlay toggles
    ["chk-heatmap", "chk-passes", "chk-shots", "chk-predictions"].forEach(id => {
      document.getElementById(id).addEventListener("change", () => _renderFrame());
    });
    $teamFilter.addEventListener("change", () => { _renderFrame(); _fetchHeatmap(); });
    $passnetTeam.addEventListener("change", () => _fetchPassNetwork());
  }

  // ── init ──────────────────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", boot);

  return { goToEvent, loadSample };
})();
