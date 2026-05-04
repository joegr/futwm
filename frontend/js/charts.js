/**
 * charts.js — D3 sidebar charts: prediction bars, xG timeline, pass network.
 *
 * Exposes a global `Charts` object.
 */

/* global d3 */

const Charts = (() => {

  // ════════════════════════════════════════════════════════════════════════
  //  Prediction bar chart
  // ════════════════════════════════════════════════════════════════════════

  function renderPredictionBars(containerId, distribution) {
    const el = document.getElementById(containerId);
    el.innerHTML = "";
    if (!distribution) return;

    const entries = Object.entries(distribution)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 7);

    const W = el.clientWidth || 300;
    const barH = 18, gap = 4, labelW = 100, valW = 44;
    const H = entries.length * (barH + gap);

    const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);

    const maxP = d3.max(entries, d => d[1]) || 1;
    const barScale = d3.scaleLinear().domain([0, maxP]).range([0, W - labelW - valW - 8]);

    const colors = {
      pass: "#3b82f6", touch: "#6366f1", shot: "#ef4444", dribble: "#f59e0b",
      tackle: "#10b981", header: "#8b5cf6", foul: "#f43f5e",
      goalkeeper_action: "#06b6d4", set_piece: "#a3a3a3",
    };

    entries.forEach(([type, prob], i) => {
      const y = i * (barH + gap);
      // bg
      svg.append("rect").attr("class", "pred-bar-bg")
        .attr("x", labelW).attr("y", y)
        .attr("width", W - labelW - valW - 8).attr("height", barH);
      // bar
      svg.append("rect").attr("class", "pred-bar")
        .attr("x", labelW).attr("y", y)
        .attr("width", barScale(prob)).attr("height", barH)
        .attr("fill", colors[type] || "#64748b");
      // label
      svg.append("text").attr("class", "pred-label")
        .attr("x", labelW - 6).attr("y", y + barH / 2 + 4)
        .attr("text-anchor", "end")
        .text(type.replace("_", " "));
      // value
      svg.append("text").attr("class", "pred-value")
        .attr("x", W - valW + 4).attr("y", y + barH / 2 + 4)
        .text((prob * 100).toFixed(1) + "%");
    });
  }

  // ════════════════════════════════════════════════════════════════════════
  //  xG timeline
  // ════════════════════════════════════════════════════════════════════════

  function renderXgTimeline(containerId, data, teams) {
    const el = document.getElementById(containerId);
    el.innerHTML = "";
    if (!data || data.length === 0) return;

    const W = el.clientWidth || 300, H = 100;
    const M = { top: 8, right: 10, bottom: 22, left: 32 };

    const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);

    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.timestamp) || 1])
      .range([M.left, W - M.right]);

    const allVals = [];
    teams.forEach(t => data.forEach(d => allVals.push(d[`xg_${t}`] || 0)));
    const yMax = Math.max(d3.max(allVals) || 0.5, 0.5);

    const yScale = d3.scaleLinear()
      .domain([0, yMax])
      .range([H - M.bottom, M.top]);

    // axes
    svg.append("g").attr("class", "xg-axis")
      .attr("transform", `translate(0,${H - M.bottom})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => `${Math.floor(d / 60)}'`));

    svg.append("g").attr("class", "xg-axis")
      .attr("transform", `translate(${M.left},0)`)
      .call(d3.axisLeft(yScale).ticks(3).tickFormat(d3.format(".1f")));

    const teamColors = ["var(--home-color)", "var(--away-color)"];

    teams.forEach((t, ti) => {
      const lineGen = d3.line()
        .x(d => xScale(d.timestamp))
        .y(d => yScale(d[`xg_${t}`] || 0))
        .curve(d3.curveStepAfter);

      svg.append("path")
        .datum(data)
        .attr("class", "xg-line")
        .attr("d", lineGen)
        .attr("stroke", teamColors[ti]);
    });

    // legend
    teams.forEach((t, ti) => {
      svg.append("circle").attr("cx", M.left + 8 + ti * 80).attr("cy", M.top + 2).attr("r", 4).attr("fill", teamColors[ti]);
      svg.append("text").attr("x", M.left + 16 + ti * 80).attr("y", M.top + 6)
        .attr("fill", "var(--text-muted)").attr("font-size", 9).text(t);
    });
  }

  // ════════════════════════════════════════════════════════════════════════
  //  Pass network
  // ════════════════════════════════════════════════════════════════════════

  function renderPassNetwork(containerId, nodes, links, pitchLength, pitchWidth) {
    const el = document.getElementById(containerId);
    el.innerHTML = "";
    if (!nodes || nodes.length === 0) return;

    const W = el.clientWidth || 300, H = 140;
    const M = { top: 6, right: 6, bottom: 6, left: 6 };

    const svg = d3.select(el).append("svg").attr("width", W).attr("height", H);

    const xS = d3.scaleLinear().domain([0, pitchLength]).range([M.left + 10, W - M.right - 10]);
    const yS = d3.scaleLinear().domain([0, pitchWidth ]).range([M.top + 10, H - M.bottom - 10]);

    const maxCount = d3.max(links, d => d.count) || 1;
    const maxTouches = d3.max(nodes, d => d.touches) || 1;
    const wScale = d3.scaleLinear().domain([0, maxCount]).range([0.5, 5]);
    const rScale = d3.scaleLinear().domain([0, maxTouches]).range([4, 14]);

    // node index
    const nodeIdx = {};
    nodes.forEach(n => nodeIdx[n.player] = n);

    // links
    svg.selectAll(".pn-link")
      .data(links.filter(l => nodeIdx[l.source] && nodeIdx[l.target]))
      .join("line")
      .attr("class", "pn-link")
      .attr("x1", l => xS(nodeIdx[l.source].avg_x))
      .attr("y1", l => yS(nodeIdx[l.source].avg_y))
      .attr("x2", l => xS(nodeIdx[l.target].avg_x))
      .attr("y2", l => yS(nodeIdx[l.target].avg_y))
      .attr("stroke", "var(--accent)")
      .attr("stroke-width", l => wScale(l.count));

    // nodes
    svg.selectAll(".pn-node")
      .data(nodes)
      .join("circle")
      .attr("class", "pn-node")
      .attr("cx", d => xS(d.avg_x))
      .attr("cy", d => yS(d.avg_y))
      .attr("r",  d => rScale(d.touches))
      .attr("fill", "var(--accent)");

    // labels
    svg.selectAll(".pn-label")
      .data(nodes)
      .join("text")
      .attr("class", "pn-label")
      .attr("x", d => xS(d.avg_x))
      .attr("y", d => yS(d.avg_y) - rScale(d.touches) - 3)
      .text(d => d.player);
  }

  // ════════════════════════════════════════════════════════════════════════
  //  public
  // ════════════════════════════════════════════════════════════════════════
  return { renderPredictionBars, renderXgTimeline, renderPassNetwork };
})();
