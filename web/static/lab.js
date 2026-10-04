"use strict";

const $ = id => document.getElementById(id);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const setText = (id, value) => {
  const el = $(id);
  if (el) el.textContent = value;
};
const STEP_MS = 750;

const towerImage = new Image();
towerImage.src = "/static/icons/tower.svg";
const icons = Object.fromEntries(["car", "drone", "robot"].map(type => {
  const icon = new Image();
  icon.src = `/static/icons/${type}.svg`;
  return [type, icon];
}));

const app = {
  token: null, snapshot: null, previous: null, scenarios: null,
  playing: true, busy: false, generation: 0, timer: null,
  frameStart: performance.now(), pausedProgress: 0, pauseWall: null,
  selectedCell: 0, view: {zoom: 1, cx: 50, cy: 50}, drag: null,
  outcome: null, outcomeStep: null, pulse: null,
  hover: null, trail: [], sliceHighlight: null, autoTrack: false,
  selectedUEId: null, spotlightLine: null,
};

async function requestJSON(url, payload) {
  const response = await fetch(url, payload ? {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
  } : {});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function setPlaybackButton() {
  setText("playIcon", app.playing ? "Ⅱ" : "▶");
  setText("playText", app.playing ? "Pause" : "Resume");
  const btn = $("playBtn");
  if (btn) btn.setAttribute("aria-pressed", String(!app.playing));
}

function showError(error) {
  app.playing = false;
  clearTimeout(app.timer);
  setPlaybackButton();
  setText("phaseLabel", "SIMULATION ERROR");
  setText("phaseTitle", "The local simulator stopped.");
  setText("phaseText", String(error));
}

function canvasSize() {
  const canvas = $("networkCanvas");
  const rect = canvas.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * pixelRatio));
  const height = Math.max(1, Math.round(rect.height * pixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  return {ctx, width: rect.width, height: rect.height};
}

function projection(width, height) {
  // Sizing the coordinate space so that the map fills up the screen generously
  const baseDim = Math.min(width, height);
  // 1.55 scale ensures the 7-cell honeycomb cluster area is expansive and fills the canvas
  const side = (baseDim - 10) * 1.55 * app.view.zoom;
  const x = value => width / 2 + (value - app.view.cx) * side / 100;
  const y = value => height / 2 + (value - app.view.cy) * side / 100;
  return {
    side, x, y,
    worldX: px => app.view.cx + (px - width / 2) * 100 / side,
    worldY: py => app.view.cy + (py - height / 2) * 100 / side,
  };
}

function clampView() {
  // Free roaming bounds allowing user to move around anywhere across the city
  const panLimit = 75;
  app.view.cx = clamp(app.view.cx, 50 - panLimit, 50 + panLimit);
  app.view.cy = clamp(app.view.cy, 50 - panLimit, 50 + panLimit);
  setText("zoomLevel", `${Math.round(app.view.zoom * 100)}%`);
  $("zoomOut").disabled = app.view.zoom <= 1;
  $("zoomIn").disabled = app.view.zoom >= 3.8;
  const trackBtn = $("trackCar");
  if (trackBtn) {
    trackBtn.classList.toggle("active", Boolean(app.autoTrack));
  }
}

function zoomTo(next, anchor) {
  const rect = $("networkCanvas").getBoundingClientRect();
  const before = projection(rect.width, rect.height);
  const px = anchor?.x ?? rect.width / 2, py = anchor?.y ?? rect.height / 2;
  const worldX = before.worldX(px), worldY = before.worldY(py);
  app.view.zoom = clamp(Math.round(next * 20) / 20, 1, 3.8);
  const after = projection(rect.width, rect.height);
  app.view.cx = worldX - (px - rect.width / 2) * 100 / after.side;
  app.view.cy = worldY - (py - rect.height / 2) * 100 / after.side;
  clampView();
}

function pointFor(ue, previous, alpha, p) {
  const prior = previous?.ues.find(item => item.id === ue.id) || ue;
  return {
    x: p.x(prior.x + (ue.x - prior.x) * alpha),
    y: p.y(prior.y + (ue.y - prior.y) * alpha),
  };
}

function roundedLabel(ctx, text, x, y, background, foreground = "#fff", border = null) {
  ctx.save();
  ctx.font = "700 11.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  const metrics = ctx.measureText(text);
  const width = metrics.width + 20;
  const height = 26;
  const canvasW = ctx.canvas.width / (window.devicePixelRatio || 1);
  const clampedX = clamp(x, width / 2 + 12, canvasW - width / 2 - 12);
  const rx = clampedX - width / 2;
  const ry = y - height / 2;

  // Shadow
  ctx.shadowColor = "rgba(15, 23, 42, 0.15)";
  ctx.shadowBlur = 8;
  ctx.shadowOffsetY = 2;

  ctx.fillStyle = background;
  ctx.beginPath();
  ctx.roundRect(rx, ry, width, height, 7);
  ctx.fill();

  ctx.shadowColor = "transparent";

  if (border) {
    ctx.strokeStyle = border;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  ctx.fillStyle = foreground;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, clampedX, y);
  ctx.restore();
}

function drawBackhaulMesh(ctx, network, p, visualNow, contract, selectedUE, width, height) {
  const isBackhaulSpot = app.spotlightLine === "backhaul";
  const isBackhaulDimmed = app.spotlightLine && !isBackhaulSpot;
  const isHoneycomb7 = network.cells.length === 7;
  const maxBackhaulDist = isHoneycomb7 ? 31.0 : (network.cells.length === 5 ? 36.5 : 52.0);

  // Discover adjacent cell pairs in the backhaul network graph
  const pairs = [];
  for (let i = 0; i < network.cells.length; i++) {
    for (let j = i + 1; j < network.cells.length; j++) {
      const c1 = network.cells[i], c2 = network.cells[j];
      const d = Math.hypot(c1.x - c2.x, c1.y - c2.y);
      if (d <= maxBackhaulDist) {
        pairs.push({c1, c2, d});
      }
    }
  }

  ctx.save();
  if (isBackhaulDimmed) ctx.globalAlpha = 0.16;

  // 1. Draw physical fiber optic conduits between adjacent base stations
  for (const {c1, c2} of pairs) {
    const x1 = p.x(c1.x), y1 = p.y(c1.y);
    const x2 = p.x(c2.x), y2 = p.y(c2.y);
    const selectedId = selectedUE?.id;

    const isContractRoute = contract && ((c1.id === selectedUE?.cell && c2.id === contract.target) || (c2.id === selectedUE?.cell && c1.id === contract.target));
    const isHoRoute = app.pulse && app.pulse.ue === selectedId && ((c1.id === app.pulse.from && c2.id === app.pulse.to) || (c2.id === app.pulse.from && c1.id === app.pulse.to));
    const isSelectedRoute = (c1.id === app.selectedCell || c2.id === app.selectedCell);

    // Outer conduit sheath
    ctx.strokeStyle = isBackhaulSpot
      ? "rgba(2, 132, 199, 0.45)"
      : isContractRoute
      ? "rgba(234, 88, 12, 0.40)"
      : isHoRoute
      ? "rgba(16, 185, 129, 0.45)"
      : isSelectedRoute
      ? "rgba(2, 132, 199, 0.28)"
      : "rgba(148, 163, 184, 0.26)";
    ctx.lineWidth = isBackhaulSpot ? 4.5 : (isContractRoute || isHoRoute) ? 3.5 : 2.2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    // Inner optical core line
    ctx.strokeStyle = isBackhaulSpot
      ? "#0284c7"
      : isContractRoute
      ? "#ea580c"
      : isHoRoute
      ? "#10b981"
      : isSelectedRoute
      ? "#38bdf8"
      : "rgba(100, 116, 139, 0.52)";
    ctx.lineWidth = isBackhaulSpot ? 2.2 : 1.4;
    ctx.setLineDash(isBackhaulSpot ? [6, 4] : [4, 4]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Animated optical photon packets on active signaling routes (strictly for selected device)
    if (isContractRoute || isHoRoute || isBackhaulSpot) {
      const packetCount = (isContractRoute || isHoRoute) ? 3 : 1;
      const speed = isHoRoute ? 0.0014 : 0.0008;
      const pColor = isHoRoute ? "#10b981" : isContractRoute ? "#ea580c" : "#38bdf8";

      for (let k = 0; k < packetCount; k++) {
        const frac = ((visualNow * speed) + k / packetCount) % 1;
        const fromX = (c1.id === (selectedUE?.cell ?? app.pulse?.from)) ? x1 : x2;
        const fromY = (c1.id === (selectedUE?.cell ?? app.pulse?.from)) ? y1 : y2;
        const toX = (fromX === x1) ? x2 : x1;
        const toY = (fromY === y1) ? y2 : y1;
        const px = fromX + (toX - fromX) * frac;
        const py = fromY + (toY - fromY) * frac;

        ctx.save();
        ctx.fillStyle = pColor;
        ctx.shadowColor = pColor;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(px, py, isBackhaulSpot ? 4.5 : 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // Mid-link explanatory label on active route
      if ((isContractRoute || isHoRoute) && (visualNow % 4200 < 3400)) {
        const midX = (x1 + x2) / 2, midY = (y1 + y2) / 2;
        const backhaulLabel = isHoRoute
          ? `⚡ 3GPP Xn FIBER: Transferring UE${selectedId} Data Plane C${app.pulse.from} ➔ C${app.pulse.to} (0ms Drop)`
          : `⚡ 3GPP Xn FIBER: Inter-gNB Handover Prep & PRB Reservation (TS 38.423)`;
        roundedLabel(ctx, backhaulLabel, midX, midY, "#0f172a", isHoRoute ? "#a7f3d0" : "#ffedd5", isHoRoute ? "#10b981" : "#ea580c");
      }
    }
  }

  // 2. Optical Distribution Frame (ODF) Junction Rings at Base Stations
  for (const cell of network.cells) {
    const cx = p.x(cell.x), cy = p.y(cell.y);
    const isConn = (cell.id === selectedUE?.cell || (contract && cell.id === contract.target) || (app.pulse && (cell.id === app.pulse.from || cell.id === app.pulse.to)));
    ctx.fillStyle = isBackhaulSpot ? "#0284c7" : isConn ? "#38bdf8" : "#64748b";
    ctx.beginPath();
    ctx.arc(cx, cy, isBackhaulSpot ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(cx, cy, isBackhaulSpot ? 2.5 : 1.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // 3. Central Backhaul Tag if Spotlighted
  if (isBackhaulSpot) {
    const c0 = network.cells[0];
    if (c0) {
      roundedLabel(ctx, "⚡ 3GPP Xn-C OPTICAL BACKHAUL MESH (TS 38.423)", p.x(c0.x), p.y(c0.y) + 38, "#0369a1", "#e0f2fe", "#0284c7");
    }
  }
  ctx.restore();
}

function drawMap(now) {
  const {ctx, width, height} = canvasSize();
  if (!app.snapshot) return;
  const p = projection(width, height);
  const network = app.snapshot.proposed;
  const previous = app.previous?.proposed;
  const scenarioFocusId = app.snapshot.focus_id;
  const selectedId = app.selectedUEId ?? scenarioFocusId;
  const selectedUE = network.ues.find(ue => ue.id === selectedId) || network.ues[0];
  const scenarioFocus = network.ues.find(ue => ue.id === scenarioFocusId);
  const contract = network.contracts.find(item => item.ue === selectedId);
  const contractUE = contract ? network.ues.find(item => item.id === contract.ue) : null;
  const visualNow = app.playing ? now : (app.pauseWall ?? now);
  const alpha = app.playing ? clamp((now - app.frameStart) / STEP_MS, 0, 1) : app.pausedProgress;

  // Auto-track selected target if enabled
  const trackTarget = selectedUE;
  if (app.autoTrack && trackTarget) {
    const prior = previous?.ues.find(item => item.id === trackTarget.id) || trackTarget;
    const targetX = prior.x + (trackTarget.x - prior.x) * alpha;
    const targetY = prior.y + (trackTarget.y - prior.y) * alpha;
    app.view.cx += (targetX - app.view.cx) * 0.08;
    app.view.cy += (targetY - app.view.cy) * 0.08;
    clampView();
  }

  ctx.clearRect(0, 0, width, height);

  // 1. Engineering Background Grounds
  const splitX = p.x(50);
  const splitY = p.y(50);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, splitX, splitY);
  ctx.fillStyle = "#f1f5f9";
  ctx.fillRect(splitX, 0, width - splitX, splitY);
  ctx.fillStyle = "#f1f5f9";
  ctx.fillRect(0, splitY, splitX, height - splitY);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(splitX, splitY, width - splitX, height - splitY);

  // Engineering Dot Grid Pattern
  ctx.save();
  ctx.fillStyle = "#cbd5e1";
  const stepWorld = 10;
  for (let wx = -20; wx <= 120; wx += stepWorld) {
    for (let wy = -20; wy <= 120; wy += stepWorld) {
      const px = p.x(wx), py = p.y(wy);
      if (px >= 0 && px <= width && py >= 0 && py <= height) {
        ctx.fillRect(px - 1, py - 1, 2, 2);
      }
    }
  }
  ctx.restore();

  // 1.5 Hexagonal Cellular Cell Geometry ("Cells shape" for 5G Base Stations)
  const isHexSpot = app.spotlightLine === "hex";
  const isHoneycomb7 = network.cells.length === 7;
  for (const cell of network.cells) {
    const cx = p.x(cell.x), cy = p.y(cell.y);
    const hexRadius = isHoneycomb7 ? 17.0 : (network.cells.length === 5 ? 21.5 : 28.8);
    const r = hexRadius * (p.side / 100);
    const isOverloaded = (cell.load >= 0.82) || (cell.free < 4);
    const isSelected = cell.id === app.selectedCell;
    const isDimmed = app.spotlightLine && !isHexSpot;

    ctx.save();
    if (isDimmed) ctx.globalAlpha = 0.22;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i;
      const hx = cx + r * Math.cos(angle);
      const hy = cy + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(hx, hy);
      else ctx.lineTo(hx, hy);
    }
    ctx.closePath();

    // Subtle translucent hexagonal cell fill
    ctx.fillStyle = isOverloaded
      ? "rgba(239, 68, 68, 0.08)"
      : isSelected
      ? "rgba(2, 132, 199, 0.09)"
      : (cell.id % 2 === 0 ? "rgba(241, 245, 249, 0.70)" : "rgba(248, 250, 252, 0.70)");
    ctx.fill();

    // Hexagonal cell perimeter border
    ctx.strokeStyle = isHexSpot
      ? "#0284c7"
      : isOverloaded
      ? "rgba(239, 68, 68, 0.75)"
      : isSelected
      ? "#0284c7"
      : "rgba(100, 116, 139, 0.65)";
    ctx.lineWidth = isHexSpot ? 3.0 : isSelected ? 2.5 : isOverloaded ? 2.2 : 1.8;
    if (isOverloaded) ctx.setLineDash([6, 4]);
    else ctx.setLineDash([]);
    ctx.stroke();

    // Hexagon corner vertex pins
    ctx.fillStyle = isOverloaded ? "#ef4444" : isSelected ? "#0284c7" : "#64748b";
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i;
      const hx = cx + r * Math.cos(angle);
      const hy = cy + r * Math.sin(angle);
      ctx.beginPath();
      ctx.arc(hx, hy, isHexSpot ? 3.5 : 2.2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Hexagon header label
    ctx.font = "800 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillStyle = isOverloaded ? "#dc2626" : isSelected ? "#0284c7" : "#64748b";
    ctx.textAlign = "left";
    const hexLabel = isOverloaded ? `CELL C${cell.id} · OVERLOADED (${cell.free} FREE)` : `CELL C${cell.id}`;
    ctx.fillText(hexLabel, cx - r * 0.72, cy - r * 0.72);
    ctx.restore();
  }

  // Cell Boundaries & Handover Frontier (drawn for 4-cell grid; 5-cell clusters use hexagonal perimeters)
  if (network.cells.length <= 4) {
    const isFrontierSpot = app.spotlightLine === "frontier";
    ctx.save();
    if (app.spotlightLine && !isFrontierSpot) ctx.globalAlpha = 0.22;
    ctx.strokeStyle = isFrontierSpot ? "#f59e0b" : "#64748b";
    ctx.lineWidth = isFrontierSpot ? 3.0 : 1.8;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    ctx.moveTo(p.x(50), 0);
    ctx.lineTo(p.x(50), height);
    ctx.moveTo(0, p.y(50));
    ctx.lineTo(width, p.y(50));
    ctx.stroke();
    ctx.setLineDash([]);

    // Boundary corridor band at y = 50
    ctx.fillStyle = isFrontierSpot ? "rgba(245, 158, 11, 0.14)" : "rgba(226, 232, 240, 0.4)";
    ctx.fillRect(0, p.y(54), width, p.y(46) - p.y(54));

    // Boundary Label
    ctx.font = "800 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillStyle = isFrontierSpot ? "#b45309" : "#64748b";
    ctx.textAlign = "center";
    ctx.fillText("CELL BOUNDARY · 3GPP A3 HANDOVER FRONTIER (y = 50m)", p.x(50), p.y(50) - 10);
    ctx.restore();
  }

  // 2.5 3GPP Xn-C Inter-gNB Optical Fiber Backhaul Mesh (TS 38.423)
  drawBackhaulMesh(ctx, network, p, visualNow, contract, selectedUE, width, height);

  // 3. Background Radio Links for other devices (Filtered to eliminate visual clutter)
  for (const ue of network.ues) {
    if (ue.id === selectedId) continue;
    const pos = pointFor(ue, previous, alpha, p);
    const cell = network.cells[ue.cell];
    const isCongested = ue.allocated === 0;
    const isSpot = isCongested ? (app.spotlightLine === "congested") : (app.spotlightLine === "active-link");
    const isHovered = app.hover?.data?.id === ue.id;

    // Only render background links when hovered or explicitly spotlighted in the line guide
    if (!isSpot && !isHovered) continue;

    ctx.save();
    if (isCongested) {
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = isSpot ? 3.0 : 1.8;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(p.x(cell.x), p.y(cell.y));
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();

      const midX = (p.x(cell.x) + pos.x) / 2;
      const midY = (p.y(cell.y) + pos.y) / 2;
      roundedLabel(ctx, "0 RB · CONGESTED", midX, midY, "#450a0a", "#fca5a5", "#ef4444", 5, 2);
    } else {
      ctx.strokeStyle = isSpot ? "#0284c7" : "rgba(100, 116, 139, 0.45)";
      ctx.lineWidth = isSpot ? 2.5 : 1.4;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(p.x(cell.x), p.y(cell.y));
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();

      const midX = (p.x(cell.x) + pos.x) / 2;
      const midY = (p.y(cell.y) + pos.y) / 2;
      roundedLabel(ctx, `${ue.slice.toUpperCase()} (${ue.allocated} RB)`, midX, midY, "rgba(15, 23, 42, 0.92)", "#93c5fd", "#0284c7", 5, 2);
    }
    ctx.restore();
  }

  // 4. Selected UE Active Serving Carrier Beam with 5G Massive MIMO Beamforming
  if (selectedUE) {
    const pos = pointFor(selectedUE, previous, alpha, p);
    const cell = network.cells[selectedUE.cell];
    const sx = p.x(cell.x), sy = p.y(cell.y);
    const connected = selectedUE.allocated > 0;
    const prior = previous?.ues.find(item => item.id === selectedUE.id) || selectedUE;
    const quality = prior.signal_quality + (selectedUE.signal_quality - prior.signal_quality) * alpha;

    const isBeamSpot = app.spotlightLine === "beam";
    const isDimmed = app.spotlightLine && !isBeamSpot;

    ctx.save();
    if (isDimmed) ctx.globalAlpha = 0.22;
    const beamColor = quality < 0.48 ? "#f59e0b" : "#0284c7";
    const beamGlow = quality < 0.48 ? "rgba(245, 158, 11, 0.32)" : "rgba(2, 132, 199, 0.28)";

    // Directional 5G Massive MIMO Beamforming Cone
    const dist = Math.hypot(pos.x - sx, pos.y - sy);
    if (dist > 5 && connected) {
      const nx = (pos.x - sx) / dist;
      const ny = (pos.y - sy) / dist;
      const px = -ny;
      const py = nx;
      const beamSpread = clamp(dist * 0.11, 10, 28);

      const coneGrad = ctx.createLinearGradient(sx, sy, pos.x, pos.y);
      coneGrad.addColorStop(0, quality < 0.48 ? "rgba(245, 158, 11, 0.28)" : "rgba(2, 132, 199, 0.24)");
      coneGrad.addColorStop(0.7, quality < 0.48 ? "rgba(245, 158, 11, 0.08)" : "rgba(2, 132, 199, 0.07)");
      coneGrad.addColorStop(1, "rgba(2, 132, 199, 0.0)");

      ctx.fillStyle = coneGrad;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(pos.x + px * beamSpread, pos.y + py * beamSpread);
      ctx.lineTo(pos.x - px * beamSpread, pos.y - py * beamSpread);
      ctx.closePath();
      ctx.fill();
    }

    // High-contrast core carrier laser beam
    ctx.strokeStyle = connected ? beamGlow : "rgba(239, 68, 68, 0.2)";
    ctx.lineWidth = isBeamSpot ? 14 : 10;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    ctx.strokeStyle = connected ? beamColor : "#ef4444";
    ctx.lineWidth = isBeamSpot ? 4.5 : 3.5;
    ctx.setLineDash(connected ? [] : [7, 6]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    // Inline Mid-Beam Label with rich explanatory status
    const beamMidX = (sx + pos.x) / 2;
    const beamMidY = (sy + pos.y) / 2;
    let beamLabel, beamBadgeBg, beamBadgeBorder, beamBadgeText;
    if (!connected) {
      beamLabel = `⛔ 5G BEAM CONGESTED: Tower C${cell.id} ➔ UE${selectedUE.id} · 0 RBs (Queueing)`;
      beamBadgeBg = "#450a0a";
      beamBadgeBorder = "#ef4444";
      beamBadgeText = "#fca5a5";
    } else if (quality < 0.48) {
      beamLabel = `⚠️ 5G SERVING BEAM: Tower C${cell.id} ➔ UE${selectedUE.id} · Fading (${Math.round(quality * 100)}% RSRP) · 3GPP A3 Handover Triggered`;
      beamBadgeBg = "#0f172a";
      beamBadgeBorder = "#f59e0b";
      beamBadgeText = "#f59e0b";
    } else {
      beamLabel = `📡 5G SERVING BEAM: Tower C${cell.id} ➔ UE${selectedUE.id} · Active Link · ${selectedUE.allocated}/${selectedUE.demand} RBs (${Math.round(quality * 100)}% RSRP)`;
      beamBadgeBg = "rgba(15, 23, 42, 0.94)";
      beamBadgeBorder = "#0284c7";
      beamBadgeText = "#38bdf8";
    }
    roundedLabel(ctx, beamLabel, beamMidX, beamMidY, beamBadgeBg, beamBadgeText, beamBadgeBorder, 5.5, 2.2);

    // Flowing 5G Radio Packets / OFDM Symbols
    if (connected) {
      for (let packet = 0; packet < 4; packet++) {
        const fraction = (visualNow * 0.00045 + packet * 0.25) % 1;
        const px = sx + (pos.x - sx) * fraction;
        const py = sy + (pos.y - sy) * fraction;
        ctx.save();
        ctx.fillStyle = beamColor;
        ctx.shadowColor = beamColor;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = beamColor;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.55;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px - (pos.x - sx) * 0.035, py - (pos.y - sy) * 0.035);
        ctx.stroke();
        ctx.restore();
      }
    }

    // 3GPP A3 Handover Trigger Alert Tag when approaching cell boundary
    if (quality < 0.48 && (!app.pulse || (visualNow - app.pulse.at >= 3200))) {
      roundedLabel(
        ctx,
        "⚠️ 3GPP A3 EVENT: RSRP FADING · HO TRIGGERED",
        clamp(pos.x, 140, width - 140),
        clamp(pos.y - 48, 28, height - 28),
        "#0f172a",
        "#f59e0b",
        "#f59e0b"
      );
    }
    ctx.restore();
  }

  // 4.5 Forward Trajectory Heading & Target Forecast
  const isTrailSpot = app.spotlightLine === "trail";
  if (selectedUE && (selectedUE.vx !== 0 || selectedUE.vy !== 0)) {
    const horizonSteps = 2.8;
    const pos = pointFor(selectedUE, previous, alpha, p);
    const predEndX = p.x(selectedUE.x + selectedUE.vx * horizonSteps);
    const predEndY = p.y(selectedUE.y + selectedUE.vy * horizonSteps);

    ctx.save();
    if (app.spotlightLine && !isTrailSpot) ctx.globalAlpha = 0.22;

    // Solid directional heading vector (no dotted dashes)
    ctx.strokeStyle = isTrailSpot ? "#38bdf8" : "rgba(2, 132, 199, 0.75)";
    ctx.lineWidth = isTrailSpot ? 3.0 : 2.0;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.lineTo(predEndX, predEndY);
    ctx.stroke();

    // Directional arrowhead
    const ang = Math.atan2(predEndY - pos.y, predEndX - pos.x);
    ctx.fillStyle = isTrailSpot ? "#38bdf8" : "#0284c7";
    ctx.beginPath();
    ctx.moveTo(predEndX, predEndY);
    ctx.lineTo(predEndX - 9 * Math.cos(ang - 0.45), predEndY - 9 * Math.sin(ang - 0.45));
    ctx.lineTo(predEndX - 9 * Math.cos(ang + 0.45), predEndY - 9 * Math.sin(ang + 0.45));
    ctx.closePath();
    ctx.fill();

    // Predicted target base station closest to forward horizon
    const predWorldX = selectedUE.x + selectedUE.vx * horizonSteps;
    const predWorldY = selectedUE.y + selectedUE.vy * horizonSteps;
    let targetCell = null, minDist = Infinity;
    for (const c of network.cells) {
      const d = Math.hypot(c.x - predWorldX, c.y - predWorldY);
      if (d < minDist) { minDist = d; targetCell = c; }
    }
    const targetTag = (targetCell && targetCell.id !== selectedUE.cell)
      ? ` ➔ APPROACHING TOWER C${targetCell.id}`
      : ` ➔ IN TOWER C${selectedUE.cell} COVERAGE`;

    // Forecast label with clear explanatory text
    roundedLabel(ctx, `🧭 PREDICTED HEADING: Vector${targetTag} (Speed: ${selectedUE.speed.toFixed(1)} m/s)`, predEndX, predEndY - 16, "#0f172a", "#38bdf8", "#0284c7", 4.5, 1.8);
    ctx.restore();
  }

  // 5. Anticipatory Reservation Arc (PPO Target Curve)
  const isResSpot = app.spotlightLine === "reservation";

  if (contract && contractUE) {
    const pos = pointFor(contractUE, previous, alpha, p);
    const target = network.cells[contract.target];
    const tx = p.x(target.x), ty = p.y(target.y);
    const active = contract.active;
    const midX = (pos.x + tx) / 2;
    const midY = (pos.y + ty) / 2 - (p.side * 0.12);

    ctx.save();
    if (app.spotlightLine && !isResSpot) ctx.globalAlpha = 0.22;
    ctx.strokeStyle = active ? "rgba(228, 43, 12, 0.28)" : "rgba(245, 158, 11, 0.22)";
    ctx.lineWidth = isResSpot ? 14 : 10;
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.quadraticCurveTo(midX, midY, tx, ty);
    ctx.stroke();

    ctx.strokeStyle = active ? "#e42b0c" : "#f59e0b";
    ctx.lineWidth = isResSpot ? 4.5 : 3.5;
    ctx.setLineDash(active ? [10, 7] : [5, 7]);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.quadraticCurveTo(midX, midY, tx, ty);
    ctx.stroke();

    // Animated particle stream
    for (let pIdx = 0; pIdx < 2; pIdx++) {
      const tFrac = ((visualNow * 0.00045) + pIdx * 0.5) % 1;
      const p1x = (1 - tFrac) * pos.x + tFrac * midX;
      const p1y = (1 - tFrac) * pos.y + tFrac * midY;
      const p2x = (1 - tFrac) * midX + tFrac * tx;
      const p2y = (1 - tFrac) * midY + tFrac * ty;
      const qx = (1 - tFrac) * p1x + tFrac * p2x;
      const qy = (1 - tFrac) * p1y + tFrac * p2y;

      ctx.save();
      ctx.fillStyle = active ? "#e42b0c" : "#f59e0b";
      ctx.shadowColor = active ? "rgba(228, 43, 12, 0.9)" : "rgba(245, 158, 11, 0.9)";
      ctx.shadowBlur = 9;
      ctx.beginPath();
      ctx.arc(qx, qy, 4.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Floating Badge for Reservation at Target Tower
    const badgeX = tx < width / 2 ? tx + 120 : tx - 120;
    const badgeText = active
      ? `🔒 TOWER C${contract.target}: ${contract.amount} RBs LOCKED FOR UE${selectedId}`
      : `🔮 TOWER C${contract.target}: ${contract.amount} RBs REQUESTED FOR UE${selectedId}`;
    roundedLabel(
      ctx, badgeText,
      clamp(badgeX, 100, width - 100),
      clamp(ty - 88, 30, height - 30),
      active ? "#0f172a" : "#1e293b",
      "#ffffff",
      active ? "#e42b0c" : "#f59e0b"
    );

    // Mid-arc Reservation Formula Badge
    roundedLabel(
      ctx,
      active
        ? `🔒 PPO PRE-ALLOCATION ARC: Tower C${contract.target} has locked ${contract.amount} RBs for UE${selectedId} (Eq. 21)`
        : `🔮 PPO PRE-ALLOCATION ARC: Requesting ${contract.amount} RBs at Tower C${contract.target} (Eq. 21)`,
      clamp(midX, 100, width - 100),
      clamp(midY - 14, 25, height - 25),
      "rgba(15, 23, 42, 0.94)",
      active ? "#ff7a59" : "#fbbf24",
      active ? "#e42b0c" : "#f59e0b"
    );
    ctx.restore();
  }

  // 6. Cell Base Stations (Towers & Segmented Capacity Meters)
  // Scaled compactly so the hexagonal cell coverage area is expansive and prominent
  for (const cell of network.cells) {
    const x = p.x(cell.x), y = p.y(cell.y);
    const selected = cell.id === app.selectedCell;
    const isOverloaded = (cell.load >= 0.82) || (cell.free < 4);

    ctx.save();
    if (app.spotlightLine && app.spotlightLine !== "hex") ctx.globalAlpha = 0.3;

    // Compact scaling: Cell coverage area is expansive, while base station is a sleek antenna node
    const towerScale = isHoneycomb7 ? 0.36 : (network.cells.length === 5 ? 0.50 : 0.72);
    const tw = 110 * towerScale, th = 121 * towerScale;

    // Ground platform shadow & pad
    ctx.fillStyle = "rgba(15, 23, 42, 0.06)";
    ctx.beginPath();
    ctx.ellipse(x, y + 18, 16, 5, 0, 0, Math.PI * 2);
    ctx.fill();

    // Selection Glow Ring
    if (selected) {
      const selR = 34;
      ctx.fillStyle = "rgba(228, 43, 12, 0.08)";
      ctx.beginPath();
      ctx.arc(x, y, selR, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = "#e42b0c";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(x, y, selR, 0, Math.PI * 2);
      ctx.stroke();

      const ripplePhase = (visualNow * 0.001) % 1;
      ctx.strokeStyle = `rgba(228, 43, 12, ${0.4 * (1 - ripplePhase)})`;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(x, y, selR + ripplePhase * 14, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Overload pulsing warning halo
    if (isOverloaded) {
      const warnPhase = (visualNow * 0.002) % 1;
      ctx.strokeStyle = `rgba(239, 68, 68, ${0.7 * (1 - warnPhase)})`;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(x, y, 26 + warnPhase * 12, 0, Math.PI * 2);
      ctx.stroke();

      roundedLabel(ctx, `⚠️ C${cell.id} FULL (${cell.free} RB)`, x, y - 46, "#450a0a", "#fca5a5", "#ef4444", 4.5, 1.8);
    }

    // Tower graphic
    if (towerImage.complete && towerImage.naturalWidth) {
      ctx.drawImage(towerImage, x - tw / 2, y - th * 0.56, tw, th);
    } else {
      ctx.fillStyle = "#334155";
      ctx.fillRect(x - 6, y - 20, 12, 40);
    }

    // Tower Identifier Badge: "C0", "C1", etc.
    const badgeW = 34, badgeH = 15;
    const badgeY = y + 15;
    ctx.save();
    ctx.shadowColor = "rgba(15, 23, 42, 0.12)";
    ctx.shadowBlur = 4;
    ctx.shadowOffsetY = 1;
    ctx.fillStyle = selected ? "#0f172a" : "#ffffff";
    ctx.beginPath();
    ctx.roundRect(x - badgeW / 2, badgeY, badgeW, badgeH, 4);
    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.strokeStyle = selected ? "#e42b0c" : (isOverloaded ? "#ef4444" : "#cbd5e1");
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.fillStyle = selected ? "#ffffff" : (isOverloaded ? "#dc2626" : "#0f172a");
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "800 9.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillText(`C${cell.id}`, x, badgeY + badgeH / 2);
    ctx.restore();

    // Segmented Capacity Gauge HUD
    const barW = 34, barH = 4, barX = x - barW / 2, barY = badgeY + badgeH + 3;
    ctx.save();
    ctx.fillStyle = "#e2e8f0";
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, 2);
    ctx.fill();

    let currentX = barX;
    if (cell.slices) {
      for (const s of cell.slices) {
        if (s.allocated > 0) {
          const segW = (barW * s.allocated) / cell.capacity;
          ctx.fillStyle = s.name.toLowerCase().includes("urllc")
            ? "#e42b0c"
            : s.name.toLowerCase().includes("embb")
            ? "#0284c7"
            : "#059669";
          ctx.beginPath();
          ctx.roundRect(currentX, barY, segW, barH, 1.5);
          ctx.fill();
          currentX += segW;
        }
      }
    } else if (cell.allocated > 0) {
      const allocW = (barW * cell.allocated) / cell.capacity;
      ctx.fillStyle = "#334155";
      ctx.beginPath();
      ctx.roundRect(barX, barY, allocW, barH, 2);
      ctx.fill();
      currentX += allocW;
    }

    if (cell.reserved > 0) {
      const heldW = (barW * cell.reserved) / cell.capacity;
      ctx.fillStyle = "#ea580c";
      ctx.shadowColor = "rgba(234, 88, 12, 0.6)";
      ctx.shadowBlur = 3;
      ctx.beginPath();
      ctx.roundRect(currentX, barY, heldW, barH, 1.5);
      ctx.fill();
    }
    ctx.restore();

    // Capacity text label below gauge
    ctx.save();
    ctx.font = "700 8px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillStyle = selected ? "#e42b0c" : (isOverloaded ? "#dc2626" : "#64748b");
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`${cell.allocated + cell.reserved}/${cell.capacity} RB`, x, barY + barH + 2);
    ctx.restore();
    ctx.restore();
    ctx.restore();
  }

  // 6.5 Saturated Cell Avoidance Indicator (Drawn when 'overloaded' line is spotlighted)
  const isOverloadSpot = app.spotlightLine === "overloaded";
  if (selectedUE && isOverloadSpot) {
    const uPos = pointFor(selectedUE, previous, alpha, p);
    let nearestOverloaded = null, minODist = Infinity;
    for (const cell of network.cells) {
      const isOverloaded = (cell.load >= 0.82) || (cell.free < 4);
      if (!isOverloaded || cell.id === selectedUE.cell) continue;
      const cx = p.x(cell.x), cy = p.y(cell.y);
      const d = Math.hypot(uPos.x - cx, uPos.y - cy);
      if (d < minODist) {
        minODist = d;
        nearestOverloaded = {cell, cx, cy, d};
      }
    }

    if (nearestOverloaded) {
      const {cell, cx, cy} = nearestOverloaded;
      ctx.save();
      ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(uPos.x, uPos.y);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      ctx.setLineDash([]);

      const midAx = (uPos.x + cx) / 2;
      const midAy = (uPos.y + cy) / 2;
      roundedLabel(ctx, `⛔ C${cell.id} SATURATED · AVOIDED BY PPO`, midAx, midAy, "#450a0a", "#fca5a5", "#ef4444", 5, 2);
      ctx.restore();
    }
  }

  // 7. Handover Admitted Transition Animation (Shockwave, Transfer Arc & Celebration)
  // Strictly scoped to the currently selected device
  if (app.pulse && app.pulse.ue === selectedId && visualNow - app.pulse.at < 3200) {
    const elapsed = visualNow - app.pulse.at;
    const progress = clamp(elapsed / 3200, 0, 1);
    const source = network.cells[app.pulse.from], target = network.cells[app.pulse.to];
    const sx = p.x(source.x), sy = p.y(source.y) - 10;
    const tx = p.x(target.x), ty = p.y(target.y) - 10;
    const midX = (sx + tx) / 2, midY = (sy + ty) / 2 - 45;

    ctx.save();
    ctx.globalAlpha = 1 - progress * 0.6;
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 4;
    ctx.shadowColor = "rgba(16, 185, 129, 0.7)";
    ctx.shadowBlur = 10;

    // Glowing handoff transfer arc
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.quadraticCurveTo(midX, midY - 20, tx, ty);
    ctx.stroke();

    // Flowing handoff transfer photon particles
    for (let hIdx = 0; hIdx < 3; hIdx++) {
      const hFrac = ((visualNow * 0.0008) + hIdx * 0.33) % 1;
      const h1x = (1 - hFrac) * sx + hFrac * midX;
      const h1y = (1 - hFrac) * sy + hFrac * (midY - 20);
      const h2x = (1 - hFrac) * midX + hFrac * tx;
      const h2y = (1 - hFrac) * (midY - 20) + hFrac * ty;
      const hx = (1 - hFrac) * h1x + hFrac * h2x;
      const hy = (1 - hFrac) * h1y + hFrac * h2y;
      ctx.fillStyle = "#34d399";
      ctx.beginPath();
      ctx.arc(hx, hy, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Arrowhead at target tower
    const angle = Math.atan2(ty - (midY - 20), tx - midX);
    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - 16 * Math.cos(angle - 0.42), ty - 16 * Math.sin(angle - 0.42));
    ctx.lineTo(tx - 16 * Math.cos(angle + 0.42), ty - 16 * Math.sin(angle + 0.42));
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Mid-arc Handover Transfer Label
    roundedLabel(ctx, `🔄 5G HANDOVER: Transferring user session C${app.pulse.from} ➔ C${app.pulse.to} (Admission Confirmed · 0ms Drop)`, midX, midY - 32, "#065f46", "#a7f3d0", "#10b981", 6, 2.5);

    // Expanding Boundary Shockwave Wave centered at the target base station tower
    const waveR = progress * p.side * 0.45;
    ctx.save();
    ctx.strokeStyle = `rgba(16, 185, 129, ${0.9 * (1 - progress)})`;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(tx, ty, waveR, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = `rgba(52, 211, 153, ${0.7 * (1 - progress)})`;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(tx, ty, waveR * 0.58, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // Target Tower Admitted Beacon Aura
    ctx.save();
    ctx.fillStyle = `rgba(16, 185, 129, ${0.15 * (1 - progress)})`;
    ctx.beginPath();
    ctx.arc(tx, ty, 50, 0, Math.PI * 2);
    ctx.fill();
    roundedLabel(ctx, "✅ 0ms ADMITTED · ONLINE", tx, ty - 68, "#065f46", "#a7f3d0", "#10b981", 5, 2);
    roundedLabel(ctx, "RELEASED", sx, sy - 68, "#1e293b", "#94a3b8", "#475569", 4.5, 1.8);
    ctx.restore();

    // Top Celebration Banner
    roundedLabel(ctx, `⚡ 5G PPO HANDOVER: UE${selectedId} Migrated C${app.pulse.from} ➜ C${app.pulse.to} · 0ms ADMISSION · ZERO PACKET LOSS (Eq. 22)`, width / 2, 38, "#047857", "#ffffff", "#10b981", 7, 3);

    // Real-time handover indicator on active device
    if (selectedUE) {
      const uPos = pointFor(selectedUE, previous, alpha, p);
      roundedLabel(ctx, `🔄 0ms HANDOVER: C${app.pulse.from} ➔ C${app.pulse.to}`, uPos.x, uPos.y - 42, "#065f46", "#a7f3d0", "#10b981", 5.5, 2.2);
    }
  }

  // 8. Focus Vehicle Motion Trail (Only shown when explicitly spotlighted via Line Guide)
  if (app.spotlightLine === "trail" && app.trail && app.trail.length > 1) {
    ctx.save();
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < app.trail.length; i++) {
      const pt = app.trail[i];
      if (i > 0) {
        const prev = app.trail[i - 1];
        if (Math.hypot(pt.x - prev.x, pt.y - prev.y) > 15) {
          started = false;
        }
      }
      const sx = p.x(pt.x), sy = p.y(pt.y);
      if (!started) { ctx.moveTo(sx, sy); started = true; }
      else { ctx.lineTo(sx, sy); }
    }
    ctx.strokeStyle = "rgba(2, 132, 199, 0.45)";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    ctx.stroke();
    ctx.restore();
  }

  // 9. Devices Rendering (Vehicles, Drones, Robots)
  const deviceOrder = [...network.ues].sort((a, b) => Number(a.id === selectedId) - Number(b.id === selectedId));

  for (const ue of deviceOrder) {
    const pos = pointFor(ue, previous, alpha, p);
    const prior = previous?.ues.find(item => item.id === ue.id) || ue;
    const quality = prior.signal_quality + (ue.signal_quality - prior.signal_quality) * alpha;

    // Quality color gradient
    const ringColor = quality < 0.48 ? "#ef4444" : quality < 0.68 ? "#f59e0b" : "#10b981";
    const isSelected = ue.id === selectedId;
    const isScenarioFocus = ue.id === scenarioFocusId;
    const iconSize = isSelected ? 48 : 25;
    const radius = isSelected ? 34 : 18;

    ctx.save();
    if (app.sliceHighlight) {
      const match = ue.slice.toLowerCase().includes(app.sliceHighlight);
      if (!match) {
        ctx.globalAlpha = 0.22;
      } else {
        ctx.shadowColor = "#38bdf8";
        ctx.shadowBlur = 14;
      }
    }

    // Slice Glow Aura
    const sliceAura = ue.slice.toLowerCase().includes("urllc")
      ? "rgba(228, 43, 12, 0.18)"
      : ue.slice.toLowerCase().includes("embb")
      ? "rgba(37, 99, 235, 0.16)"
      : "rgba(5, 150, 105, 0.16)";
    ctx.fillStyle = sliceAura;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius + 8, 0, Math.PI * 2);
    ctx.fill();

    // Signal Quality halo for selected vehicle when RSRP fading
    if (isSelected && quality < 0.48) {
      const pulsePhase = (visualNow * 0.003) % 1;
      ctx.strokeStyle = `rgba(239, 68, 68, ${0.7 * (1 - pulsePhase)})`;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius + 10 + pulsePhase * 18, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Directional velocity heading vector
    if (isSelected && (ue.vx !== 0 || ue.vy !== 0)) {
      const angle = Math.atan2(ue.vy, ue.vx);
      const headX = pos.x + Math.cos(angle) * (radius + 12);
      const headY = pos.y + Math.sin(angle) * (radius + 12);
      ctx.fillStyle = ringColor;
      ctx.beginPath();
      ctx.moveTo(headX, headY);
      ctx.lineTo(headX - 10 * Math.cos(angle - 0.5), headY - 10 * Math.sin(angle - 0.5));
      ctx.lineTo(headX - 10 * Math.cos(angle + 0.5), headY - 10 * Math.sin(angle + 0.5));
      ctx.closePath();
      ctx.fill();
    }

    // Outer Vehicle Disk
    ctx.shadowColor = isSelected ? "rgba(15, 23, 42, 0.2)" : "rgba(15, 23, 42, 0.08)";
    ctx.shadowBlur = isSelected ? 12 : 5;
    ctx.shadowOffsetY = isSelected ? 3 : 1;

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowColor = "transparent";

    // Signal Quality Stroke
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = isSelected ? 4 : 2.5;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
    ctx.stroke();

    if (isSelected) {
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius + 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Icon render: side-view car stays upright (flips on horizontal movement), drones/robots rotate with heading
    const icon = icons[ue.entity_type];
    if (icon && icon.complete && icon.naturalWidth) {
      ctx.save();
      ctx.translate(pos.x, pos.y);
      if (ue.entity_type === "car") {
        if (ue.vx < -0.05) {
          ctx.scale(-1, 1);
        }
      } else {
        const heading = (ue.vx !== 0 || ue.vy !== 0) ? Math.atan2(ue.vy, ue.vx) : 0;
        ctx.rotate(heading);
      }
      ctx.drawImage(icon, -iconSize / 2, -iconSize / 2, iconSize, iconSize);
      ctx.restore();
    } else {
      ctx.fillStyle = "#334155";
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 9, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // Floating Target HUD Tag
    if (isSelected) {
      roundedLabel(
        ctx,
        `UE${ue.id} · ${ue.slice.toUpperCase()} (${ue.allocated}/${ue.demand} RB · C${ue.cell})`,
        clamp(pos.x, 80, width - 80),
        clamp(pos.y + 38, 20, height - 20),
        "#0f172a",
        "#ffffff",
        "#38bdf8",
        5, 2
      );
    } else {
      ctx.save();
      ctx.fillStyle = "#64748b";
      ctx.textAlign = "center";
      ctx.font = "700 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      ctx.fillText(`UE${ue.id}`, pos.x, pos.y + 22);
      ctx.restore();
    }
  }

  // 10. Interactive Inspection HUD Tooltip on Canvas
  if (app.hover) {
    const h = app.hover;
    const cardW = 248, cardH = 82;
    const cardX = clamp(h.px + 14, 12, width - cardW - 12);
    const cardY = clamp(h.py - cardH / 2, 12, height - cardH - 12);

    ctx.save();
    ctx.shadowColor = "rgba(15, 23, 42, 0.28)";
    ctx.shadowBlur = 16;
    ctx.shadowOffsetY = 4;
    ctx.fillStyle = "rgba(15, 23, 42, 0.94)";
    ctx.beginPath();
    ctx.roundRect(cardX, cardY, cardW, cardH, 9);
    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.16)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.textAlign = "left";
    ctx.textBaseline = "middle";

    if (h.type === "ue") {
      const ue = h.data;
      const isUrllc = ue.slice.toLowerCase().includes("urllc");
      ctx.font = "800 11.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      ctx.fillStyle = isUrllc ? "#ff7a59" : "#38bdf8";
      ctx.fillText(`UE${ue.id} · ${ue.slice.toUpperCase()} (${ue.entity_type.toUpperCase()})`, cardX + 12, cardY + 18);

      ctx.font = "600 10.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      ctx.fillStyle = "#f1f5f9";
      ctx.fillText(`Serving: Tower C${ue.cell}  ·  RSRP: ${Math.round(ue.signal_quality * 100)}%`, cardX + 12, cardY + 39);

      const qosMet = ue.allocated >= ue.demand;
      ctx.fillStyle = qosMet ? "#34d399" : "#fbbf24";
      ctx.fillText(`Demand: ${ue.demand} RB · Alloc: ${ue.allocated} RB (${qosMet ? "SLA Met" : "Degraded"}) · ${ue.speed.toFixed(1)}m/s`, cardX + 12, cardY + 60);
    } else if (h.type === "cell") {
      const cell = h.data;
      ctx.font = "800 11.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(`BASE STATION TOWER C${cell.id}`, cardX + 12, cardY + 18);

      ctx.font = "600 10.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      ctx.fillStyle = "#f1f5f9";
      ctx.fillText(`Capacity: ${cell.capacity} RB  (${cell.allocated} used, ${cell.reserved} held, ${cell.free} free)`, cardX + 12, cardY + 39);

      ctx.fillStyle = "#94a3b8";
      const sSum = cell.slices.map(s => `${s.name.slice(0, 4)}: ${s.allocated}RB`).join(" · ");
      ctx.fillText(`Load: ${Math.round(cell.load * 100)}%  ·  ${sSum}`, cardX + 12, cardY + 60);
    }
    ctx.restore();
  }

  // 11. Real-Time Simulation Telemetry HUD on Canvas (Docked in top-left open space)
  if (width >= 420 && selectedUE) {
    const hudW = 236, hudH = 92;
    const hudX = 16, hudY = 16;
    ctx.save();
    ctx.shadowColor = "rgba(15, 23, 42, 0.25)";
    ctx.shadowBlur = 12;
    ctx.shadowOffsetY = 3;
    ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
    ctx.beginPath();
    ctx.roundRect(hudX, hudY, hudW, hudH, 8);
    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.textAlign = "left";
    ctx.textBaseline = "middle";

    ctx.font = "800 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillStyle = "#38bdf8";
    ctx.fillText(`5G PPO RAN TELEMETRY · UE${selectedUE.id} (${selectedUE.slice.toUpperCase()})`, hudX + 12, hudY + 16);

    ctx.font = "600 9.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillStyle = "#f1f5f9";
    const prior = previous?.ues.find(item => item.id === selectedUE.id) || selectedUE;
    const quality = prior.signal_quality + (selectedUE.signal_quality - prior.signal_quality) * alpha;
    ctx.fillText(`Serving: Tower C${selectedUE.cell}  ·  RSRP: ${Math.round(quality * 100)}%`, hudX + 12, hudY + 34);

    const ueContract = network.contracts.find(item => item.ue === selectedUE.id) || network.contracts.find(item => item.ue === scenarioFocusId);
    ctx.fillStyle = ueContract?.active ? "#ff7a59" : "#cbd5e1";
    const holdStr = ueContract ? `${ueContract.amount} RB @ C${ueContract.target} (${ueContract.active ? "LOCKED" : "QUEUED"})` : "None (In Coverage)";
    ctx.fillText(`PPO Hold: ${holdStr}`, hudX + 12, hudY + 52);

    const hoStatus = app.pulse && (visualNow - app.pulse.at < 3200)
      ? "HO ADMITTED (0ms Drop)"
      : quality < 0.48 ? "A3 EVENT TRIGGERED" : "SLA SATISFIED";
    ctx.fillStyle = hoStatus.includes("ADMITTED") ? "#34d399" : hoStatus.includes("TRIGGERED") ? "#fbbf24" : "#94a3b8";
    ctx.fillText(`Handover: ${hoStatus}  ·  ${selectedUE.allocated}/${selectedUE.demand} RB`, hudX + 12, hudY + 70);

    ctx.font = "500 8.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillStyle = "#64748b";
    ctx.fillText(`Speed: ${selectedUE.speed.toFixed(1)} m/s · Entity: ${selectedUE.entity_type.toUpperCase()}`, hudX + 12, hudY + 86);
    ctx.restore();
  }

  // 12. Dynamic Physical Scale Bar on Canvas (Google Maps / OpenStreetMap style)
  ctx.save();
  const scaleMetersList = [50, 100, 200, 500, 1000];
  const pxPerMeter = p.side / 1000;
  let chosenMeters = 200;
  for (const m of scaleMetersList) {
    const pxLen = m * pxPerMeter;
    if (pxLen >= 45 && pxLen <= 120) {
      chosenMeters = m;
      break;
    }
  }
  const scaleBarPx = chosenMeters * pxPerMeter;
  const barX = 16, barY = height - 16;
  ctx.strokeStyle = "#475569";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(barX, barY - 4);
  ctx.lineTo(barX, barY);
  ctx.lineTo(barX + scaleBarPx, barY);
  ctx.lineTo(barX + scaleBarPx, barY - 4);
  ctx.stroke();

  ctx.font = "700 9.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace";
  ctx.fillStyle = "#475569";
  ctx.textAlign = "center";
  ctx.fillText(`${chosenMeters} m`, barX + scaleBarPx / 2, barY - 5);
  ctx.restore();

  // 13. High-Precision Navigation Compass Rose in Top-Right
  if (width >= 480) {
    const compassX = width - 262, compassY = 46;
    ctx.save();
    ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
    ctx.strokeStyle = "rgba(203, 213, 225, 0.8)";
    ctx.lineWidth = 1;
    ctx.shadowColor = "rgba(15, 23, 42, 0.1)";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(compassX, compassY, 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.shadowColor = "transparent";

    // North Needle (Vermillion)
    ctx.fillStyle = "#e42b0c";
    ctx.beginPath();
    ctx.moveTo(compassX, compassY - 13);
    ctx.lineTo(compassX - 3.5, compassY);
    ctx.lineTo(compassX + 3.5, compassY);
    ctx.closePath();
    ctx.fill();

    // South Needle (Slate)
    ctx.fillStyle = "#94a3b8";
    ctx.beginPath();
    ctx.moveTo(compassX, compassY + 13);
    ctx.lineTo(compassX - 3.5, compassY);
    ctx.lineTo(compassX + 3.5, compassY);
    ctx.closePath();
    ctx.fill();

    // North Label
    ctx.font = "800 8px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillStyle = "#e42b0c";
    ctx.textAlign = "center";
    ctx.fillText("N", compassX, compassY - 14);
    ctx.restore();
  }

  // 14. Target Lock Telemetry Reticle on Focus Car when Auto-Tracking
  if (app.autoTrack && focusPosition) {
    ctx.save();
    const bx = focusPosition.x, by = focusPosition.y;
    const boxSize = 42;
    const bracketLen = 10;
    const pulseOffset = Math.sin(visualNow * 0.005) * 2;

    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.8;
    // Top-Left bracket
    ctx.beginPath();
    ctx.moveTo(bx - boxSize - pulseOffset + bracketLen, by - boxSize - pulseOffset);
    ctx.lineTo(bx - boxSize - pulseOffset, by - boxSize - pulseOffset);
    ctx.lineTo(bx - boxSize - pulseOffset, by - boxSize - pulseOffset + bracketLen);
    ctx.stroke();
    // Top-Right bracket
    ctx.beginPath();
    ctx.moveTo(bx + boxSize + pulseOffset - bracketLen, by - boxSize - pulseOffset);
    ctx.lineTo(bx + boxSize + pulseOffset, by - boxSize - pulseOffset);
    ctx.lineTo(bx + boxSize + pulseOffset, by - boxSize - pulseOffset + bracketLen);
    ctx.stroke();
    // Bottom-Left bracket
    ctx.beginPath();
    ctx.moveTo(bx - boxSize - pulseOffset + bracketLen, by + boxSize + pulseOffset);
    ctx.lineTo(bx - boxSize - pulseOffset, by + boxSize + pulseOffset);
    ctx.lineTo(bx - boxSize - pulseOffset, by + boxSize + pulseOffset - bracketLen);
    ctx.stroke();
    // Bottom-Right bracket
    ctx.beginPath();
    ctx.moveTo(bx + boxSize + pulseOffset - bracketLen, by + boxSize + pulseOffset);
    ctx.lineTo(bx + boxSize + pulseOffset, by + boxSize + pulseOffset);
    ctx.lineTo(bx + boxSize + pulseOffset, by + boxSize + pulseOffset - bracketLen);
    ctx.stroke();

    ctx.font = "800 8px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace";
    ctx.fillStyle = "#38bdf8";
    ctx.textAlign = "center";
    ctx.fillText("TARGET ACQUIRED · AUTO-FOLLOW", bx, by - boxSize - pulseOffset - 4);
    ctx.restore();
  }
}

function renderTower() {
  if (!app.snapshot) return;
  const cellId = app.selectedCell;
  setText("towerHeading", `Tower C${cellId}`);

  for (const name of ["normal", "proposed"]) {
    const cell = app.snapshot[name].cells[cellId];
    const host = $(name + "Tower");
    if (!host) continue;
    host.replaceChildren();

    const root = document.createElement("div");
    root.className = "ledger";

    const head = document.createElement("div");
    head.className = "ledger-head";

    const title = document.createElement("span");
    title.className = name === "proposed" ? "proposed" : "";
    title.textContent = name === "normal" ? "No reservation" : "PPO reservation";

    const sum = document.createElement("span");
    sum.className = "ledger-sum";
    sum.textContent = `${cell.allocated} used · ${cell.reserved} held · ${cell.free} free / ${cell.capacity}`;
    head.append(title, sum);
    root.append(head);

    const bar = document.createElement("div");
    bar.className = "capacity-bar";
    let hasSliceAlloc = false;
    for (const s of cell.slices) {
      if (s.allocated > 0) {
        hasSliceAlloc = true;
        const part = document.createElement("i");
        part.className = `allocated slice-${s.name.toLowerCase()}`;
        part.style.width = `${(100 * s.allocated) / cell.capacity}%`;
        part.title = `${s.name}: ${s.allocated} RB allocated`;
        bar.append(part);
      }
    }
    if (!hasSliceAlloc && cell.allocated > 0) {
      const part = document.createElement("i");
      part.className = "allocated";
      part.style.width = `${(100 * cell.allocated) / cell.capacity}%`;
      bar.append(part);
    }
    if (cell.reserved > 0) {
      const part = document.createElement("i");
      part.className = "held";
      part.style.width = `${(100 * cell.reserved) / cell.capacity}%`;
      part.title = `PPO Held: ${cell.reserved} RB`;
      bar.append(part);
    }
    root.append(bar);

    const sliceGrid = document.createElement("div");
    sliceGrid.className = "ledger-slice-grid";
    for (const s of cell.slices) {
      const item = document.createElement("div");
      item.className = "ledger-slice-item";
      const sKey = s.name.toLowerCase();

      const hdr = document.createElement("div");
      hdr.className = "slice-item-header";
      const nameSpan = document.createElement("span");
      nameSpan.className = `slice-item-name ${sKey}`;
      const dot = document.createElement("i");
      dot.className = "slice-item-dot";
      nameSpan.append(dot, document.createTextNode(" " + s.name));
      const uesSpan = document.createElement("span");
      uesSpan.className = "slice-item-sla";
      uesSpan.textContent = `${s.ues} UE${s.ues === 1 ? "" : "s"}`;
      hdr.append(nameSpan, uesSpan);

      const rbsDiv = document.createElement("div");
      rbsDiv.className = "slice-item-rbs";
      rbsDiv.textContent = `${s.allocated}/${s.demand} RB`;
      if (s.reserved > 0) {
        const heldTag = document.createElement("span");
        heldTag.className = "held-tag";
        heldTag.textContent = `+${s.reserved} held`;
        rbsDiv.append(heldTag);
      }

      const slaDiv = document.createElement("div");
      slaDiv.className = "slice-item-sla";
      slaDiv.textContent = `SLA: ${s.sla_rbs_per_ue} RB/UE · <${s.latency_max_ms}ms`;

      item.append(hdr, rbsDiv, slaDiv);
      sliceGrid.append(item);
    }
    root.append(sliceGrid);

    const slices = document.createElement("div");
    slices.className = "ledger-slices";
    slices.textContent = cell.slices
      .map(s => `${s.name} ${s.allocated}${s.reserved ? ` + ${s.reserved} held` : ""}`)
      .join("   ·   ");
    root.append(slices);
    host.append(root);
  }

  for (const button of $("towerNav").querySelectorAll("button")) {
    const selected = Number(button.dataset.cell) === cellId;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function selectUE(id) {
  app.selectedUEId = id;
  app.trail = [];
  renderDeviceChips();
  if (app.snapshot) {
    updateTeaching(app.snapshot);
  }
}

function renderDeviceChips() {
  const host = $("deviceChips");
  if (!host || !app.snapshot) return;
  const ues = app.snapshot.proposed.ues;
  const currentSelected = app.selectedUEId ?? app.snapshot.focus_id;
  host.replaceChildren();

  for (const ue of ues) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `device-chip ${ue.id === currentSelected ? "active" : ""} slice-${ue.slice.toLowerCase()}`;
    const iconSym = ue.entity_type === "car" ? "🚗" : ue.entity_type === "drone" ? "🛸" : "🤖";
    const isFocus = ue.id === app.snapshot.focus_id;
    chip.textContent = `${iconSym} UE${ue.id} · ${ue.slice} (${ue.allocated}/${ue.demand} RB)${isFocus ? " ★" : ""}`;
    chip.title = `Click to inspect UE${ue.id} (${ue.entity_type}, ${ue.slice} slice, ${ue.demand} RB SLA)`;
    chip.addEventListener("click", () => selectUE(ue.id));
    host.append(chip);
  }
}

function renderTowerNav() {
  const nav = $("towerNav");
  if (!nav || !app.snapshot) return;
  const cells = app.snapshot.proposed.cells;
  if (nav.children.length === cells.length) {
    for (const button of nav.querySelectorAll("button")) {
      const selected = Number(button.dataset.cell) === app.selectedCell;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
    return;
  }
  nav.replaceChildren();
  for (const cell of cells) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.cell = String(cell.id);
    button.textContent = `C${cell.id}`;
    button.setAttribute("aria-label", `Inspect tower C${cell.id}`);
    button.addEventListener("click", () => selectTower(cell.id));
    if (cell.id === app.selectedCell) {
      button.classList.add("active");
      button.setAttribute("aria-pressed", "true");
    }
    nav.append(button);
  }
}

function buildSparklineData(historyList, key, width = 200, height = 44, padY = 4) {
  const bottomY = height - padY;
  const topY = padY;
  const usableH = bottomY - topY;
  if (!historyList || historyList.length === 0) {
    return {
      line: `M 0,${bottomY} L ${width},${bottomY}`,
      area: `M 0,${bottomY} L ${width},${bottomY} L ${width},${bottomY} Z`,
      last: {x: width, y: bottomY, val: 0},
      points: [],
    };
  }
  const n = historyList.length;
  const points = historyList.map((item, idx) => {
    const x = n > 1 ? (idx / (n - 1)) * width : width;
    const rawVal = item[key] ?? 0;
    const val = clamp(rawVal, 0, 1);
    const y = bottomY - val * usableH;
    return {x, y, val: rawVal, step: item.t ?? idx};
  });
  const linePoints = points.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" L ");
  const line = `M ${linePoints}`;
  const firstX = points[0].x.toFixed(1);
  const lastX = points[points.length - 1].x.toFixed(1);
  const area = `M ${firstX},${bottomY} L ${linePoints} L ${lastX},${bottomY} Z`;
  const last = points[points.length - 1];
  return {line, area, last, points};
}

function updateSparklines(history) {
  if (!history) return;
  const w = 200, h = 44, padY = 4;
  const nHsrData = buildSparklineData(history.normal, "hsr", w, h, padY);
  const pHsrData = buildSparklineData(history.proposed, "hsr", w, h, padY);
  const nQosData = buildSparklineData(history.normal, "qos", w, h, padY);
  const pQosData = buildSparklineData(history.proposed, "qos", w, h, padY);

  app.sparkData = {
    hsr: {normal: nHsrData, proposed: pHsrData},
    qos: {normal: nQosData, proposed: pQosData},
  };

  const nHsr = $("hsrSparkNormal"), pHsr = $("hsrSparkProposed");
  const nHsrArea = $("hsrAreaNormal"), pHsrArea = $("hsrAreaProposed");
  const nHsrDot = $("hsrDotNormal"), pHsrDot = $("hsrDotProposed");

  if (nHsr && nHsrArea) {
    nHsr.setAttribute("d", nHsrData.line);
    nHsrArea.setAttribute("d", nHsrData.area);
    if (nHsrDot && nHsrData.last) {
      nHsrDot.setAttribute("cx", nHsrData.last.x.toFixed(1));
      nHsrDot.setAttribute("cy", nHsrData.last.y.toFixed(1));
    }
  }
  if (pHsr && pHsrArea) {
    pHsr.setAttribute("d", pHsrData.line);
    pHsrArea.setAttribute("d", pHsrData.area);
    if (pHsrDot && pHsrData.last) {
      pHsrDot.setAttribute("cx", pHsrData.last.x.toFixed(1));
      pHsrDot.setAttribute("cy", pHsrData.last.y.toFixed(1));
    }
  }

  const nQos = $("qosSparkNormal"), pQos = $("qosSparkProposed");
  const nQosArea = $("qosAreaNormal"), pQosArea = $("qosAreaProposed");
  const nQosDot = $("qosDotNormal"), pQosDot = $("qosDotProposed");

  if (nQos && nQosArea) {
    nQos.setAttribute("d", nQosData.line);
    nQosArea.setAttribute("d", nQosData.area);
    if (nQosDot && nQosData.last) {
      nQosDot.setAttribute("cx", nQosData.last.x.toFixed(1));
      nQosDot.setAttribute("cy", nQosData.last.y.toFixed(1));
    }
  }
  if (pQos && pQosArea) {
    pQos.setAttribute("d", pQosData.line);
    pQosArea.setAttribute("d", pQosData.area);
    if (pQosDot && pQosData.last) {
      pQosDot.setAttribute("cx", pQosData.last.x.toFixed(1));
      pQosDot.setAttribute("cy", pQosData.last.y.toFixed(1));
    }
  }
}

function setupSparklineScrubbers() {
  for (const metric of ["hsr", "qos"]) {
    const svg = $(`${metric}Sparkline`);
    const tooltip = $(`${metric}ScrubTooltip`);
    const scrubLine = $(`${metric}ScrubLine`);
    if (!svg || !tooltip) continue;

    const onMove = e => {
      const data = app.sparkData?.[metric];
      if (!data || !data.proposed.points.length) return;
      const rect = svg.getBoundingClientRect();
      const clientX = e.clientX ?? e.touches?.[0]?.clientX;
      if (clientX == null) return;
      const xPct = clamp((clientX - rect.left) / rect.width, 0, 1);
      const svgX = xPct * 200;

      const points = data.proposed.points;
      let closest = points[0], minD = Infinity;
      for (const pt of points) {
        const d = Math.abs(pt.x - svgX);
        if (d < minD) { minD = d; closest = pt; }
      }
      const pIndex = points.indexOf(closest);
      const normalPt = data.normal.points[pIndex] || data.normal.points[data.normal.points.length - 1];

      if (scrubLine) {
        scrubLine.style.display = "block";
        scrubLine.setAttribute("x1", closest.x.toFixed(1));
        scrubLine.setAttribute("x2", closest.x.toFixed(1));
      }

      const pVal = ((closest.val ?? 0) * 100).toFixed(1);
      const nVal = ((normalPt?.val ?? 0) * 100).toFixed(1);
      tooltip.textContent = `t=${closest.step}s · PPO: ${pVal}% · Baseline: ${nVal}%`;
      tooltip.classList.add("active");
    };

    const onLeave = () => {
      if (scrubLine) scrubLine.style.display = "none";
      tooltip.textContent = metric === "hsr" ? "Target cell admission rate" : "SLA fulfilled on arrival";
      tooltip.classList.remove("active");
    };

    svg.addEventListener("pointermove", onMove);
    svg.addEventListener("pointerleave", onLeave);
    svg.addEventListener("touchstart", onMove, {passive: true});
    svg.addEventListener("touchend", onLeave, {passive: true});
  }
}

function focusHandover(events, focusId) {
  return events.find(event => event.startsWith(`HO UE${focusId} `));
}

function updateLineStatusHud(snapshot, trackedId) {
  if (!snapshot) return;
  const proposed = snapshot.proposed;
  const selectedUE = proposed.ues.find(ue => ue.id === trackedId) || proposed.ues[0];
  if (!selectedUE) return;

  const targetEl = $("hudSelectedUe");
  if (targetEl) {
    const isFocus = selectedUE.id === snapshot.focus_id;
    targetEl.textContent = `INSPECTING: UE${selectedUE.id} (${selectedUE.slice.toUpperCase()} ${selectedUE.entity_type.toUpperCase()})${isFocus ? " ★" : ""}`;
  }

  // 1. Serving Beam Status
  const connected = selectedUE.allocated > 0;
  const qualityPct = Math.round(selectedUE.signal_quality * 100);
  const beamCard = $("hudBeamCard");
  const beamBadge = $("hudBeamBadge");
  const beamText = $("hudBeamText");

  if (beamCard && beamBadge && beamText) {
    if (!connected) {
      beamBadge.className = "hud-badge danger";
      beamBadge.textContent = "CONGESTED (0 RB)";
      beamText.textContent = `Tower C${selectedUE.cell} is saturated (100% capacity). 0 RBs allocated; UE${selectedUE.id} is queued.`;
    } else if (selectedUE.signal_quality < 0.48) {
      beamBadge.className = "hud-badge warning";
      beamBadge.textContent = "FADING LINK";
      beamText.textContent = `Serving Tower C${selectedUE.cell} ➔ UE${selectedUE.id} (${selectedUE.allocated}/${selectedUE.demand} RBs). Signal dropping (${qualityPct}% RSRP); 3GPP A3 Handover Triggered.`;
    } else {
      beamBadge.className = "hud-badge active";
      beamBadge.textContent = "ACTIVE CARRIER";
      beamText.textContent = `Serving Tower C${selectedUE.cell} ➔ UE${selectedUE.id} · Active Data Link · ${selectedUE.allocated}/${selectedUE.demand} RBs (${qualityPct}% RSRP · SLA Met).`;
    }
  }

  // 2. PPO Pre-Allocation Arc Status
  const contract = proposed.contracts.find(item => item.ue === trackedId);
  const resCard = $("hudResCard");
  const resBadge = $("hudResBadge");
  const resText = $("hudResText");

  if (resCard && resBadge && resText) {
    if (contract?.active) {
      resBadge.className = "hud-badge held";
      resBadge.textContent = "CAPACITY HELD";
      resText.textContent = `Tower C${contract.target} has locked ${contract.amount} RBs exclusively for UE${selectedUE.id} arrival (Eq. 21 · Guaranteed Handover).`;
    } else if (contract) {
      resBadge.className = "hud-badge active";
      resBadge.textContent = "REQUEST FLIGHT";
      resText.textContent = `Requesting ${contract.amount} RBs in advance at Tower C${contract.target} via 3GPP Xn backhaul signaling.`;
    } else {
      resBadge.className = "hud-badge idle";
      resBadge.textContent = "IDLE";
      resText.textContent = `UE${selectedUE.id} is stable inside Tower C${selectedUE.cell} coverage. No reservation needed yet.`;
    }
  }

  // 3. 3GPP Xn Backhaul Mesh Status
  const backhaulCard = $("hudBackhaulCard");
  const backhaulBadge = $("hudBackhaulBadge");
  const backhaulText = $("hudBackhaulText");

  if (backhaulCard && backhaulBadge && backhaulText) {
    const isHoActive = app.pulse && app.pulse.ue === trackedId && (performance.now() - app.pulse.at < 3200);
    if (isHoActive) {
      backhaulBadge.className = "hud-badge success";
      backhaulBadge.textContent = "SESSION TRANSFER";
      backhaulText.textContent = `Transferring UE${selectedUE.id} data plane context from Tower C${app.pulse.from} to C${app.pulse.to} across optical fiber (0ms Delay).`;
    } else if (contract?.active) {
      backhaulBadge.className = "hud-badge active";
      backhaulBadge.textContent = "HOLD CONFIRMED";
      backhaulText.textContent = `Inter-gNB optical fiber link active: Tower C${selectedUE.cell} and C${contract.target} confirmed ${contract.amount} RB reservation (TS 38.423).`;
    } else if (contract) {
      backhaulBadge.className = "hud-badge active";
      backhaulBadge.textContent = "SIGNALING";
      backhaulText.textContent = `Transmitting 3GPP XnAP Handover Preparation Request between C${selectedUE.cell} and C${contract.target}.`;
    } else {
      backhaulBadge.className = "hud-badge standby";
      backhaulBadge.textContent = "MONITORING";
      backhaulText.textContent = `3GPP Xn-C optical fiber mesh online (TS 38.423). Standby for inter-gNB coordination.`;
    }
  }

  // 4. Handover Status
  const hoCard = $("hudHoCard");
  const hoBadge = $("hudHoBadge");
  const hoText = $("hudHoText");

  if (hoCard && hoBadge && hoText) {
    const isHoActive = app.pulse && app.pulse.ue === trackedId && (performance.now() - app.pulse.at < 3200);
    if (isHoActive) {
      hoBadge.className = "hud-badge success";
      hoBadge.textContent = "0ms ADMITTED";
      hoText.textContent = `Seamless Handover Executed! Session migrated C${app.pulse.from} ➔ C${app.pulse.to} with zero interruption.`;
    } else if (contract?.active) {
      hoBadge.className = "hud-badge held";
      hoBadge.textContent = "TARGET PREPARED";
      hoText.textContent = `Target Tower C${contract.target} is armed with ${contract.amount} reserved RBs. Ready for zero-packet-loss switch.`;
    } else if (selectedUE.signal_quality < 0.48) {
      hoBadge.className = "hud-badge warning";
      hoBadge.textContent = "A3 TRIGGERED";
      hoText.textContent = `Signal fading below threshold. Cell boundary crossed; evaluating candidate target towers.`;
    } else {
      hoBadge.className = "hud-badge nominal";
      hoBadge.textContent = "NOMINAL";
      hoText.textContent = `Stable connection with Tower C${selectedUE.cell} (${qualityPct}% RSRP · ${selectedUE.speed.toFixed(1)} m/s).`;
    }
  }
}

function updateTeaching(snapshot) {
  const focusId = snapshot.focus_id;
  const trackedId = app.selectedUEId ?? focusId;
  const proposed = snapshot.proposed;
  const normal = snapshot.normal;
  const proposedEvent = focusHandover(proposed.events, trackedId);
  const normalEvent = focusHandover(normal.events, trackedId);
  const contract = proposed.contracts.find(item => item.ue === trackedId);

  if (proposedEvent) {
    const parsed = proposedEvent.match(/^HO UE(\d+) C(\d+)->C(\d+) (SUCCESS|FAIL) \((\d+)\/(\d+)RB\)/);
    const baseline = normalEvent?.match(/(SUCCESS|FAIL) \((\d+)\/(\d+)RB\)/);
    if (parsed) {
      const hoUeId = Number(parsed[1]);
      app.outcome = {
        normal: baseline ? {
          success: baseline[1] === "SUCCESS",
          used: Number(baseline[2]),
          demand: Number(baseline[3]),
        } : null,
        proposed: {
          success: parsed[4] === "SUCCESS",
          used: Number(parsed[5]),
          demand: Number(parsed[6]),
        },
        from: Number(parsed[2]),
        to: Number(parsed[3]),
        ue: hoUeId,
      };
      app.outcomeStep = proposed.step;
      if (app.outcome.proposed.success) {
        app.pulse = {from: app.outcome.from, to: app.outcome.to, at: performance.now(), ue: hoUeId};
      }
    }
  }

  let number = "01", label = "APPROACH", title = "Follow the highlighted car.";
  let explanation = "A solid line is active RB service. Red around the car means it is near the edge of its serving cell in the illustrative signal proxy.";
  let stepIndex = 0;

  if (app.outcome && app.outcome.ue === trackedId && proposed.step - app.outcomeStep < 18) {
    number = "04";
    label = "HANDOVER RESULT";
    stepIndex = 3;
    const {normal: base, proposed: ppo, from, to} = app.outcome;
    title = `UE${trackedId} moves C${from} → C${to}: ${ppo.success ? "PPO admits it" : "PPO blocks it"}.`;
    explanation = `Same arrival: no reservation could offer only ${base?.used ?? "?"}/${base?.demand ?? "?"} RB (${base?.success ? "admitted" : "blocked"}); PPO granted ${ppo.used}/${ppo.demand} RB (${ppo.success ? "admitted" : "blocked"}). ${ppo.used === ppo.demand ? "Full demand was met." : "Full-demand QoS may still fail."}`;
  } else if (contract?.active) {
    number = "03";
    label = "CAPACITY HELD";
    stepIndex = 2;
    title = `Tower C${contract.target} now holds ${contract.amount} RBs.`;
    explanation = `The orange path connects UE${trackedId} to its forecast target. The target's RB ledger below shows what was held before arrival.`;
  } else if (contract) {
    number = "02";
    label = "FORECAST & REQUEST";
    stepIndex = 1;
    title = `PPO requests ${contract.amount} RBs at C${contract.target}.`;
    explanation = "The dashed orange path is a planned reservation. It consumes no RBs until activation succeeds.";
  }

  setText("phaseNumber", number);
  setText("phaseLabel", label);
  setText("phaseTitle", title);
  setText("phaseText", explanation);

  // Update step progress dots
  const dots = document.querySelectorAll(".step-dots .dot");
  dots.forEach((dot, idx) => {
    dot.classList.toggle("active", idx === stepIndex);
  });

  // Update 7-Stage text2.txt Pipeline Nodes
  const pipeNodes = document.querySelectorAll(".pipeline-flow .pipe-node");
  pipeNodes.forEach((node, idx) => {
    let isActive = false, isCompleted = false;
    if (stepIndex === 0) { // 01 APPROACH
      isActive = idx === 0 || idx === 1;
      isCompleted = false;
    } else if (stepIndex === 1) { // 02 FORECAST & REQUEST
      isActive = idx === 2 || idx === 3;
      isCompleted = idx < 2;
    } else if (stepIndex === 2) { // 03 CAPACITY HELD
      isActive = idx === 4;
      isCompleted = idx < 4;
    } else if (stepIndex === 3) { // 04 HANDOVER RESULT
      isActive = idx === 5 || idx === 6;
      isCompleted = idx < 5;
    }
    node.classList.toggle("active", isActive);
    node.classList.toggle("completed", isCompleted);
  });

  for (const name of ["normal", "proposed"]) {
    const outcome = app.outcome?.[name];
    const element = $(name + "Outcome");
    if (!element) continue;
    element.textContent = outcome
      ? `${outcome.success ? "ADMITTED" : "BLOCKED"} · ${outcome.used}/${outcome.demand} RB ${outcome.success ? "granted" : "available"}`
      : name === "normal" ? "Waiting for crossing" : "Watching for a hold";

    element.parentElement.classList.toggle("fail", Boolean(outcome && !outcome.success));
    element.parentElement.classList.toggle("success", Boolean(outcome && outcome.success));
    const metrics = snapshot[name].metrics;
    setText(name + "Score", `${metrics.successes} / ${metrics.handovers} admitted`);
  }

  const decision = proposed.decision;
  if (decision?.ue === focusId) {
    setText("decisionHeading", decision.action ? `PPO: ${decision.label}` : "PPO: no reservation");
    setText("decisionText", `Forecast target C${decision.target}; estimated arrival ${decision.eta_steps} step${decision.eta_steps === 1 ? "" : "s"} away. The score is heuristic, not calibrated probability.`);
  } else if (contract) {
    setText("decisionHeading", contract.active ? "Hold active" : "Hold requested");
    setText("decisionText", `Focus car UE${focusId}: ${contract.amount} RBs for tower C${contract.target}.`);
  } else {
    setText("decisionHeading", "Monitoring the six-device network");
    setText("decisionText", "PPO acts on one forecast candidate per interval; other devices continue generating background traffic.");
  }

  const facts = $("decisionFacts");
  if (facts) {
    facts.replaceChildren();
    const ue = proposed.ues.find(item => item.id === focusId);
    const factList = [
      "Saved linear PPO",
      `UE${focusId} · ${ue?.slice || "uRLLC"}`,
      `${Math.round((ue?.signal_quality || 0.5) * 100)}% signal proxy`,
    ];
    if (decision?.value_estimate != null) {
      factList.push(`Critic V(s) = ${decision.value_estimate.toFixed(2)}`);
    }
    for (const value of factList) {
      const chip = document.createElement("span");
      chip.textContent = value;
      facts.append(chip);
    }
  }

  // Update Real-Time Telemetry Stats Strip
  const nMetrics = normal.metrics, pMetrics = proposed.metrics;
  const nHsr = nMetrics.handovers > 0 ? (nMetrics.successes / nMetrics.handovers * 100) : 0;
  const pHsr = pMetrics.handovers > 0 ? (pMetrics.successes / pMetrics.handovers * 100) : 0;
  setText("statHsrNormal", `${nHsr.toFixed(1)}%`);
  setText("statHsrProposed", `${pHsr.toFixed(1)}%`);
  const hsrDelta = pHsr - nHsr;
  setText("statHsrDelta", `${hsrDelta >= 0 ? "+" : ""}${hsrDelta.toFixed(1)} pp`);

  const nQos = nMetrics.qos_rate != null ? nMetrics.qos_rate * 100 : (nMetrics.handovers > 0 ? (nMetrics.qos_successes / nMetrics.handovers * 100) : 0);
  const pQos = pMetrics.qos_rate != null ? pMetrics.qos_rate * 100 : (pMetrics.handovers > 0 ? (pMetrics.qos_successes / pMetrics.handovers * 100) : 0);
  setText("statQosNormal", `${nQos.toFixed(1)}%`);
  setText("statQosProposed", `${pQos.toFixed(1)}%`);
  const qosDelta = pQos - nQos;
  setText("statQosDelta", `${qosDelta >= 0 ? "+" : ""}${qosDelta.toFixed(1)} pp`);

  const totalAlloc = proposed.cells.reduce((sum, c) => sum + c.allocated, 0);
  const totalHeld = proposed.cells.reduce((sum, c) => sum + c.reserved, 0);
  const totalCap = proposed.cells.reduce((sum, c) => sum + c.capacity, 0);
  const avgLoad = totalCap > 0 ? Math.round(((totalAlloc + totalHeld) / totalCap) * 100) : 0;

  setText("statTotalAlloc", `${totalAlloc} RB`);
  setText("statTotalReserved", `${totalHeld} RB held`);
  setText("statCellLoad", `${avgLoad}%`);
  setText("statCapacityChip", `Pool: ${totalCap} RB`);
  setText("statCapacityDesc", `Capacity utilization across the ${proposed.cells.length} base stations`);

  // Update Live Flight Recorder Event Stream
  const stream = $("liveEventStream");
  if (stream) {
    const rawEvents = [];
    if (proposed.events) {
      for (const ev of proposed.events) rawEvents.push({text: ev, net: "PPO"});
    }
    if (normal.events) {
      for (const ev of normal.events) {
        if (ev.startsWith("HO ")) rawEvents.push({text: ev, net: "Baseline"});
      }
    }
    if (rawEvents.length > 0) {
      stream.replaceChildren();
      for (const item of rawEvents.slice(-6).reverse()) {
        const row = document.createElement("div");
        row.className = "event-row";
        if (item.text.includes("SUCCESS")) row.classList.add("success");
        else if (item.text.includes("FAIL")) row.classList.add("fail");
        else if (item.text.includes("reserve")) row.classList.add("reserve");
        row.textContent = `[${snapshot.proposed.time.toFixed(1)}s] ${item.net}: ${item.text}`;
        stream.append(row);
      }
    }
  }

  // Update Rolling Sparklines
  updateSparklines(snapshot.history);

  // Update Live Line Activity HUD for selected device
  updateLineStatusHud(snapshot, trackedId);
}

function acceptSnapshot(snapshot) {
  app.previous = app.snapshot;
  app.snapshot = snapshot;
  app.frameStart = performance.now();
  setText("simTime", `${snapshot.proposed.time.toFixed(1)} s`);

  const trackedId = app.selectedUEId ?? snapshot.focus_id;
  const focus = snapshot.proposed.ues.find(u => u.id === trackedId);
  if (focus) {
    app.trail.push({x: focus.x, y: focus.y});
    if (app.trail.length > 25) app.trail.shift();
  }

  updateTeaching(snapshot);
  renderDeviceChips();
  renderTowerNav();
  renderTower();
}

function frame(now) {
  drawMap(now);
  requestAnimationFrame(frame);
}

async function advance(generation) {
  if (!app.playing || app.busy || !app.token || generation !== app.generation) return;
  app.busy = true;
  try {
    const snapshot = await requestJSON("/api/step", {session: app.token, count: 1});
    if (generation === app.generation && app.playing) acceptSnapshot(snapshot);
  } catch (error) {
    if (generation === app.generation && app.playing) showError(error);
  } finally {
    app.busy = false;
    if (app.playing && generation === app.generation) {
      app.timer = setTimeout(() => advance(generation), STEP_MS);
    }
  }
}

async function startScenario() {
  const generation = ++app.generation;
  clearTimeout(app.timer);
  app.token = null;
  app.snapshot = null;
  app.previous = null;
  app.outcome = null;
  app.outcomeStep = null;
  app.pulse = null;
  app.trail = [];
  app.hover = null;
  app.selectedUEId = null;
  app.spotlightLine = null;
  app.view = {zoom: 1, cx: 50, cy: 50};
  app.selectedCell = 0;
  clampView();
  app.playing = true;
  app.busy = false;
  app.pausedProgress = 0;
  app.pauseWall = null;
  setPlaybackButton();

  const scenario = $("scenario").value;
  const preset = app.scenarios?.[scenario];
  setText("scenarioDescription", preset?.description || "");

  const stream = $("liveEventStream");
  if (stream) {
    stream.replaceChildren();
    const initRow = document.createElement("div");
    initRow.className = "event-row idle";
    initRow.textContent = `Scenario "${scenario}" initialized. Awaiting handover telemetry...`;
    stream.append(initRow);
  }

  try {
    const response = await requestJSON("/api/session", {scenario, seed: preset?.seed ?? 21});
    if (generation !== app.generation) return;
    app.token = response.session;
    acceptSnapshot(response.snapshot);
    app.timer = setTimeout(() => advance(generation), STEP_MS);
  } catch (error) {
    if (generation === app.generation) showError(error);
  }
}

function togglePlayback() {
  app.playing = !app.playing;
  if (app.playing) {
    const pausedFor = performance.now() - (app.pauseWall ?? performance.now());
    if (app.pulse) app.pulse.at += pausedFor;
    app.pauseWall = null;
    app.frameStart = performance.now() - app.pausedProgress * STEP_MS;
    app.timer = setTimeout(() => advance(app.generation), Math.max(0, STEP_MS * (1 - app.pausedProgress)));
  } else {
    clearTimeout(app.timer);
    app.pauseWall = performance.now();
    app.pausedProgress = clamp((app.pauseWall - app.frameStart) / STEP_MS, 0, 1);
  }
  setPlaybackButton();
}

function selectTower(cellId) {
  app.selectedCell = cellId;
  renderTower();
}

function setupMapInteractions() {
  const canvas = $("networkCanvas");
  canvasSize();
  window.addEventListener("resize", canvasSize);
  $("zoomIn").addEventListener("click", () => zoomTo(app.view.zoom + 0.25));
  $("zoomOut").addEventListener("click", () => zoomTo(app.view.zoom - 0.25));
  $("fitMap").addEventListener("click", () => {
    app.autoTrack = false;
    app.view = {zoom: 1, cx: 50, cy: 50};
    clampView();
  });

  const trackBtn = $("trackCar");
  if (trackBtn) {
    trackBtn.addEventListener("click", () => {
      app.autoTrack = !app.autoTrack;
      if (app.autoTrack) {
        if (app.view.zoom < 1.4) app.view.zoom = 1.5;
        const focus = app.snapshot?.proposed?.ues?.find(ue => ue.id === app.snapshot?.focus_id);
        if (focus) {
          app.view.cx = focus.x;
          app.view.cy = focus.y;
        }
      }
      clampView();
    });
  }

  const handoverBtn = $("focusHandover");
  if (handoverBtn) {
    handoverBtn.addEventListener("click", () => {
      app.autoTrack = false;
      app.view = {zoom: 1.5, cx: 28, cy: 50};
      clampView();
    });
  }

  window.addEventListener("keydown", event => {
    if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
    const rect = canvas.getBoundingClientRect();
    const p = projection(rect.width, rect.height);
    const panStep = (35 / p.side) * 100;
    if (event.key === "ArrowLeft" || event.key === "a" || event.key === "A") {
      app.autoTrack = false;
      app.view.cx -= panStep;
      clampView();
    } else if (event.key === "ArrowRight" || event.key === "d" || event.key === "D") {
      app.autoTrack = false;
      app.view.cx += panStep;
      clampView();
    } else if (event.key === "ArrowUp" || event.key === "w" || event.key === "W") {
      app.autoTrack = false;
      app.view.cy -= panStep;
      clampView();
    } else if (event.key === "ArrowDown" || event.key === "s" || event.key === "S") {
      app.autoTrack = false;
      app.view.cy += panStep;
      clampView();
    } else if (event.key === "+" || event.key === "=") {
      zoomTo(app.view.zoom + 0.25);
    } else if (event.key === "-" || event.key === "_") {
      zoomTo(app.view.zoom - 0.25);
    } else if (event.key === "f" || event.key === "F" || event.key === "0") {
      app.autoTrack = false;
      app.view = {zoom: 1, cx: 50, cy: 50};
      clampView();
    } else if (event.key === "h" || event.key === "H") {
      app.autoTrack = false;
      app.view = {zoom: 1.5, cx: 36, cy: 50};
      clampView();
    } else if (event.key === " ") {
      event.preventDefault();
      app.autoTrack = !app.autoTrack;
      if (app.autoTrack && app.view.zoom < 1.4) app.view.zoom = 1.5;
      clampView();
    }
  });

  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomTo(app.view.zoom + (event.deltaY < 0 ? 0.2 : -0.2), {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  }, {passive: false});

  const onStart = (clientX, clientY, pointerId) => {
    app.autoTrack = false;
    canvas.classList.add("dragging");
    app.drag = {x: clientX, y: clientY, moved: false};
    if (pointerId != null) {
      try { canvas.setPointerCapture(pointerId); } catch {}
    }
  };

  const onMove = (clientX, clientY) => {
    const rect = canvas.getBoundingClientRect();
    const p = projection(rect.width, rect.height);
    const mx = clientX - rect.left;
    const my = clientY - rect.top;

    if (app.drag) {
      const dx = clientX - app.drag.x;
      const dy = clientY - app.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) app.drag.moved = true;
      app.view.cx -= (dx * 100) / p.side;
      app.view.cy -= (dy * 100) / p.side;
      clampView();
      app.drag.x = clientX;
      app.drag.y = clientY;
      app.hover = null;
      return;
    }

    if (!app.snapshot) return;
    const network = app.snapshot.proposed;
    let found = null;
    for (const ue of network.ues) {
      const px = p.x(ue.x), py = p.y(ue.y);
      if (Math.hypot(px - mx, py - my) < 28) {
        found = {type: "ue", data: ue, px: mx, py: my};
        break;
      }
    }
    if (!found) {
      for (const cell of network.cells) {
        const cx = p.x(cell.x), cy = p.y(cell.y);
        if (Math.hypot(cx - mx, cy - my) < 55) {
          found = {type: "cell", data: cell, px: mx, py: my};
          break;
        }
      }
    }
    app.hover = found;
  };

  const onEnd = (clientX, clientY) => {
    canvas.classList.remove("dragging");
    const wasClick = app.drag && !app.drag.moved;
    app.drag = null;
    if (!wasClick || !app.snapshot) return;
    const rect = canvas.getBoundingClientRect();
    const p = projection(rect.width, rect.height);
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    const closestCell = app.snapshot.proposed.cells
      .map(cell => ({
        id: cell.id,
        distance: Math.hypot(p.x(cell.x) - x, p.y(cell.y) - y),
      }))
      .sort((a, b) => a.distance - b.distance)[0];

    let clickedUE = null, minUeDist = 999;
    for (const ue of app.snapshot.proposed.ues) {
      const dist = Math.hypot(p.x(ue.x) - x, p.y(ue.y) - y);
      if (dist < 28 && dist < minUeDist) {
        minUeDist = dist;
        clickedUE = ue;
      }
    }

    if (clickedUE && (!closestCell || minUeDist < closestCell.distance || closestCell.distance > 30)) {
      selectUE(clickedUE.id);
      return;
    }

    if (closestCell && closestCell.distance < 72) {
      selectTower(closestCell.id);
    }
  };

  canvas.addEventListener("pointerdown", e => onStart(e.clientX, e.clientY, e.pointerId));
  canvas.addEventListener("mousedown", e => onStart(e.clientX, e.clientY));
  canvas.addEventListener("pointermove", e => onMove(e.clientX, e.clientY));
  canvas.addEventListener("mousemove", e => onMove(e.clientX, e.clientY));
  canvas.addEventListener("pointerleave", () => { app.hover = null; });
  canvas.addEventListener("pointerup", e => onEnd(e.clientX, e.clientY));
  canvas.addEventListener("mouseup", e => onEnd(e.clientX, e.clientY));
  canvas.addEventListener("pointercancel", () => { app.drag = null; canvas.classList.remove("dragging"); });

  const nav = $("towerNav");
  for (let id = 0; id < 7; id++) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.cell = String(id);
    button.textContent = `C${id}`;
    button.setAttribute("aria-label", `Inspect tower C${id}`);
    button.addEventListener("click", () => selectTower(id));
    if (id === 0) {
      button.classList.add("active");
      button.setAttribute("aria-pressed", "true");
    }
    nav.append(button);
  }
  clampView();
}

async function init() {
  setupMapInteractions();
  setupSparklineScrubbers();
  $("playBtn").addEventListener("click", togglePlayback);
  $("scenario").addEventListener("change", startScenario);

  // Interactive Slice Spotlight
  const sliceTags = document.querySelectorAll(".slice-tags .slice-tag");
  sliceTags.forEach(tag => {
    let sliceKey = null;
    if (tag.classList.contains("urllc")) sliceKey = "urllc";
    else if (tag.classList.contains("embb")) sliceKey = "embb";
    else if (tag.classList.contains("mmtc")) sliceKey = "mmtc";

    tag.style.cursor = "pointer";
    tag.addEventListener("pointerenter", () => { app.sliceHighlight = sliceKey; });
    tag.addEventListener("pointerleave", () => { app.sliceHighlight = null; });
  });

  // Interactive Line Guide Spotlight
  const legendPills = document.querySelectorAll(".legend-pill");
  legendPills.forEach(pill => {
    const key = pill.dataset.line;
    if (!key) return;
    pill.addEventListener("pointerenter", () => { app.spotlightLine = key; });
    pill.addEventListener("pointerleave", () => {
      if (app.spotlightLine === key) app.spotlightLine = null;
    });
    pill.addEventListener("click", () => {
      app.spotlightLine = (app.spotlightLine === key) ? null : key;
      legendPills.forEach(p => p.classList.toggle("active", p.dataset.line === app.spotlightLine));
    });
  });

  requestAnimationFrame(frame);

  try {
    const meta = await requestJSON("/api/meta");
    app.scenarios = meta.scenarios;
    const baseline = meta.published?.metrics?.find(item => item.method === "No reservation");
    const ppo = meta.published?.metrics?.find(item => item.method?.includes("PPO"));
    setText(
      "benchmarkStat",
      baseline && ppo
        ? `${(ppo.handover_success_rate * 100).toFixed(1)}% vs ${(baseline.handover_success_rate * 100).toFixed(1)}%`
        : "Unavailable"
    );
    await startScenario();
  } catch (error) {
    showError(error);
  }
}

init();
