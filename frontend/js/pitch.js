/**
 * pitch.js — D3 pitch renderer and event overlay.
 *
 * Exposes a global `PitchVis` object.
 */

/* global d3 */

const PitchVis = (() => {
  // ── state ────────────────────────────────────────────────────────────
  let svg, gPitch, gHeatmap, gEvents, gPredictions;
  let scaleX, scaleY;
  let pitchData = { length: 105, width: 68, goal_width: 7.32, penalty_depth: 16.5, goal_area_depth: 5.5, centre_radius: 9.15 };

  const MARGIN = { top: 12, right: 12, bottom: 12, left: 12 };

  // ── init ──────────────────────────────────────────────────────────────
  function init(containerId) {
    const container = document.getElementById(containerId);
    const rect = container.getBoundingClientRect();
    const W = rect.width  || 700;
    const H = rect.height || 460;

    svg = d3.select(`#${containerId}`)
      .append("svg")
      .attr("viewBox", `0 0 ${W} ${H}`)
      .attr("preserveAspectRatio", "xMidYMid meet");

    _addDefs(svg);

    gPitch       = svg.append("g").attr("class", "g-pitch");
    gHeatmap     = svg.append("g").attr("class", "g-heatmap");
    gEvents      = svg.append("g").attr("class", "g-events");
    gPredictions = svg.append("g").attr("class", "g-predictions");

    _updateScales(W, H);
    _drawPitch();

    window.addEventListener("resize", () => {
      const r2 = container.getBoundingClientRect();
      svg.attr("viewBox", `0 0 ${r2.width} ${r2.height}`);
      _updateScales(r2.width, r2.height);
      _drawPitch();
    });
  }

  function setPitchData(d) {
    pitchData = d;
    const vb = svg.attr("viewBox").split(" ");
    _updateScales(+vb[2], +vb[3]);
    _drawPitch();
  }

  // ── scales ────────────────────────────────────────────────────────────
  function _updateScales(W, H) {
    const innerW = W - MARGIN.left - MARGIN.right;
    const innerH = H - MARGIN.top  - MARGIN.bottom;
    const pitchRatio = pitchData.length / pitchData.width;
    const boxRatio   = innerW / innerH;

    let pW, pH;
    if (boxRatio > pitchRatio) {
      pH = innerH; pW = pH * pitchRatio;
    } else {
      pW = innerW; pH = pW / pitchRatio;
    }

    const offX = MARGIN.left + (innerW - pW) / 2;
    const offY = MARGIN.top  + (innerH - pH) / 2;

    scaleX = d3.scaleLinear().domain([0, pitchData.length]).range([offX, offX + pW]);
    scaleY = d3.scaleLinear().domain([0, pitchData.width ]).range([offY, offY + pH]);
  }

  // ── draw pitch markings ───────────────────────────────────────────────
  function _drawPitch() {
    gPitch.selectAll("*").remove();
    const L = pitchData.length, W = pitchData.width;
    const gw = pitchData.goal_width, pd = pitchData.penalty_depth;
    const gd = pitchData.goal_area_depth, cr = pitchData.centre_radius;
    const hw = W / 2, hg = gw / 2;

    // background
    gPitch.append("rect")
      .attr("class", "pitch-bg")
      .attr("x", scaleX(0)).attr("y", scaleY(0))
      .attr("width",  scaleX(L) - scaleX(0))
      .attr("height", scaleY(W) - scaleY(0))
      .attr("rx", 4);

    const line = (x1, y1, x2, y2) => gPitch.append("line")
      .attr("class", "pitch-line")
      .attr("x1", scaleX(x1)).attr("y1", scaleY(y1))
      .attr("x2", scaleX(x2)).attr("y2", scaleY(y2));

    const rect = (x, y, w, h) => gPitch.append("rect")
      .attr("class", "pitch-line")
      .attr("x", scaleX(x)).attr("y", scaleY(y))
      .attr("width", scaleX(x + w) - scaleX(x))
      .attr("height", scaleY(y + h) - scaleY(y));

    // outline
    rect(0, 0, L, W);

    // halfway
    line(L / 2, 0, L / 2, W);

    // centre circle
    const cxPx = scaleX(L / 2), cyPx = scaleY(hw);
    const rPx  = scaleX(cr) - scaleX(0);
    gPitch.append("circle").attr("class", "pitch-line")
      .attr("cx", cxPx).attr("cy", cyPx).attr("r", rPx);

    // centre spot
    gPitch.append("circle").attr("class", "pitch-spot")
      .attr("cx", cxPx).attr("cy", cyPx).attr("r", 3);

    // penalty areas
    const paW = pd, paYoff = hw - hg - 11;
    rect(0, paYoff, paW, gw + 22);
    rect(L - paW, paYoff, paW, gw + 22);

    // goal areas
    const gaYoff = hw - hg - 5.5;
    rect(0, gaYoff, gd, gw + 11);
    rect(L - gd, gaYoff, gd, gw + 11);

    // penalty spots
    const penDot = (x) => gPitch.append("circle").attr("class", "pitch-spot")
      .attr("cx", scaleX(x)).attr("cy", scaleY(hw)).attr("r", 3);
    penDot(11);
    penDot(L - 11);

    // goals
    const goalRect = (x, dir) => {
      const gDepth = 2.5;
      const yTop = hw - hg;
      gPitch.append("rect").attr("class", "pitch-goal")
        .attr("x", scaleX(dir > 0 ? x : x - gDepth))
        .attr("y", scaleY(yTop))
        .attr("width",  Math.abs(scaleX(gDepth) - scaleX(0)))
        .attr("height", scaleY(yTop + gw) - scaleY(yTop));
    };
    goalRect(0, -1);
    goalRect(L, 1);

    // penalty arcs
    const arcGen = d3.arc();
    [11, L - 11].forEach((px, i) => {
      const cx2 = scaleX(px), cy2 = scaleY(hw);
      const r2  = scaleX(cr + px) - scaleX(px);
      // only draw outside penalty area
      const startAngle = i === 0 ? -0.65 : Math.PI - 0.65;
      const endAngle   = i === 0 ?  0.65 : Math.PI + 0.65;
      gPitch.append("path")
        .attr("class", "pitch-line")
        .attr("d", arcGen({ innerRadius: r2, outerRadius: r2, startAngle, endAngle }))
        .attr("transform", `translate(${cx2},${cy2})`);
    });
  }

  // ── defs (arrowheads) ─────────────────────────────────────────────────
  function _addDefs(svg) {
    const defs = svg.append("defs");
    // event arrowhead
    defs.append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 5).attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", "#fff").attr("opacity", .7);

    // prediction arrowhead
    defs.append("marker")
      .attr("id", "arrowhead-pred")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 5).attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", "var(--accent)").attr("opacity", .7);
  }

  // ── render events on pitch ────────────────────────────────────────────
  function renderEvents(events, currentIndex, opts = {}) {
    gEvents.selectAll("*").remove();
    if (!events || events.length === 0) return;

    const showPasses = opts.passes !== false;
    const showShots  = opts.shots  !== false;
    const teamFilter = opts.team || "";
    const windowSize = opts.window || 12;

    // show events in a window around current index
    const lo = Math.max(0, currentIndex - windowSize);
    const hi = Math.min(events.length - 1, currentIndex);
    const visible = events.slice(lo, hi + 1);

    // pass arrows
    if (showPasses) {
      gEvents.selectAll(".pass-arrow")
        .data(visible.filter(e => e.event_type === "pass" && e.end_x != null && (!teamFilter || e.team === teamFilter)))
        .join("line")
        .attr("class", e => `pass-arrow ${e.team === (events[0] || {}).team ? "home" : "away"}`)
        .attr("x1", e => scaleX(e.x)).attr("y1", e => scaleY(e.y))
        .attr("x2", e => scaleX(e.end_x)).attr("y2", e => scaleY(e.end_y))
        .attr("opacity", (e, i) => 0.2 + 0.8 * (i / visible.length));
    }

    // shot arrows
    if (showShots) {
      gEvents.selectAll(".shot-arrow")
        .data(visible.filter(e => e.event_type === "shot" && e.end_x != null && (!teamFilter || e.team === teamFilter)))
        .join("line")
        .attr("class", e => `shot-arrow ${e.outcome === "goal" ? "goal" : ""} ${e.team === (events[0] || {}).team ? "home" : "away"}`)
        .attr("x1", e => scaleX(e.x)).attr("y1", e => scaleY(e.y))
        .attr("x2", e => scaleX(e.end_x)).attr("y2", e => scaleY(e.end_y))
        .attr("stroke", e => e.outcome === "goal" ? "var(--green)" : "var(--yellow)");
    }

    // event dots
    gEvents.selectAll(".event-dot")
      .data(visible.filter(e => !teamFilter || e.team === teamFilter))
      .join("circle")
      .attr("class", e => `event-dot ${e.team === (events[0] || {}).team ? "home" : "away"}`)
      .attr("cx", e => scaleX(e.x))
      .attr("cy", e => scaleY(e.y))
      .attr("r",  (e, i) => i === visible.length - 1 ? 6 : 3.5)
      .attr("opacity", (e, i) => 0.3 + 0.7 * (i / visible.length))
      .on("click", (ev, e) => {
        if (opts.onEventClick) opts.onEventClick(e.index);
      });

    // highlight current event
    const cur = events[currentIndex];
    if (cur) {
      gEvents.append("circle")
        .attr("cx", scaleX(cur.x)).attr("cy", scaleY(cur.y))
        .attr("r", 8)
        .attr("fill", "none")
        .attr("stroke", "var(--accent)")
        .attr("stroke-width", 2.5)
        .attr("stroke-dasharray", "4 3");
    }
  }

  // ── heatmap ───────────────────────────────────────────────────────────
  function renderHeatmap(cells, maxCount) {
    gHeatmap.selectAll("*").remove();
    if (!cells || cells.length === 0) return;

    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxCount || 1]);
    const cellW = (scaleX(pitchData.length) - scaleX(0)) / 20;
    const cellH = (scaleY(pitchData.width)  - scaleY(0)) / 13;

    gHeatmap.selectAll(".heatmap-cell")
      .data(cells)
      .join("rect")
      .attr("class", "heatmap-cell")
      .attr("x", d => scaleX(d.x) - cellW / 2)
      .attr("y", d => scaleY(d.y) - cellH / 2)
      .attr("width",  cellW)
      .attr("height", cellH)
      .attr("rx", 2)
      .attr("fill", d => colorScale(d.count));
  }

  function clearHeatmap() { gHeatmap.selectAll("*").remove(); }

  // ── prediction overlay ─────────────────────────────────────────────────
  function renderPredictions(preds, currentEvent) {
    gPredictions.selectAll("*").remove();
    if (!preds || !preds.top_predictions) return;

    // dest mean marker
    if (preds.dest_mean) {
      gPredictions.append("circle")
        .attr("class", "pred-dest-marker")
        .attr("cx", scaleX(preds.dest_mean[0]))
        .attr("cy", scaleY(preds.dest_mean[1]))
        .attr("r", 10);
    }

    // arrows from current position to predicted destinations
    if (currentEvent) {
      preds.top_predictions.forEach(p => {
        if (p.end_x != null) {
          gPredictions.append("line")
            .attr("class", "pred-arrow")
            .attr("x1", scaleX(currentEvent.x))
            .attr("y1", scaleY(currentEvent.y))
            .attr("x2", scaleX(p.end_x))
            .attr("y2", scaleY(p.end_y));
        }
      });
    }
  }

  function clearPredictions() { gPredictions.selectAll("*").remove(); }

  // ── public ────────────────────────────────────────────────────────────
  return { init, setPitchData, renderEvents, renderHeatmap, clearHeatmap, renderPredictions, clearPredictions };
})();
