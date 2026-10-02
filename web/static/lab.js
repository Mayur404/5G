"use strict";

const $ = id => document.getElementById(id);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const setText = (id, value) => {
  const el = $(id);
  if (el) el.textContent = value;
};
const STEP_MS = 620;

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
  hover: null, trail: [], sliceHighlight: null,
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
  const side = Math.min(width - 54, height - 54) * app.view.zoom;
  const x = value => width / 2 + (value - app.view.cx) * side / 100;
  const y = value => height / 2 + (value - app.view.cy) * side / 100;
  return {
    side, x, y,
    worldX: px => app.view.cx + (px - width / 2) * 100 / side,
    worldY: py => app.view.cy + (py - height / 2) * 100 / side,
  };
}

function clampView() {
  const halfVisible = 50 / app.view.zoom;
  app.view.cx = clamp(app.view.cx, halfVisible, 100 - halfVisible);
  app.view.cy = clamp(app.view.cy, halfVisible, 100 - halfVisible);
  setText("zoomLevel", `${Math.round(app.view.zoom * 100)}%`);
  $("zoomOut").disabled = app.view.zoom <= 1;
  $("zoomIn").disabled = app.view.zoom >= 2.8;
}

function zoomTo(next, anchor) {
  const rect = $("networkCanvas").getBoundingClientRect();
  const before = projection(rect.width, rect.height);
  const px = anchor?.x ?? rect.width / 2, py = anchor?.y ?? rect.height / 2;
  const worldX = before.worldX(px), worldY = before.worldY(py);
  app.view.zoom = clamp(Math.round(next * 20) / 20, 1, 2.8);
  const afterSide = Math.min(rect.width - 54, rect.height - 54) * app.view.zoom;
  app.view.cx = worldX - (px - rect.width / 2) * 100 / afterSide;
  app.view.cy = worldY - (py - rect.height / 2) * 100 / afterSide;
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
  const width = metrics.width + 22;
  const height = 26;
  const rx = x - width / 2;
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
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawMap(now) {
  if (!app.snapshot) return;
  const {ctx, width, height} = canvasSize();
  const p = projection(width, height);
  const network = app.snapshot.proposed;
  const previous = app.previous?.proposed;
  const focusId = app.snapshot.focus_id;
  const focus = network.ues.find(ue => ue.id === focusId);
  const visualNow = app.playing ? now : (app.pauseWall ?? now);
  const alpha = app.playing ? clamp((now - app.frameStart) / STEP_MS, 0, 1) : app.pausedProgress;

  ctx.clearRect(0, 0, width, height);

  // 1. Engineering Background
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);

  // Quadrant Zone backgrounds
  const sectors = [
    {cell: 0, x: 0, y: 0, shade: "#f1f5f9"},
    {cell: 1, x: 50, y: 0, shade: "#f8fafc"},
    {cell: 2, x: 0, y: 50, shade: "#f8fafc"},
    {cell: 3, x: 50, y: 50, shade: "#f1f5f9"},
  ];
  for (const s of sectors) {
    ctx.fillStyle = s.shade;
    ctx.fillRect(p.x(s.x), p.y(s.y), p.side / 2, p.side / 2);
  }

  // Engineering Dot Grid Pattern
  ctx.save();
  ctx.fillStyle = "#cbd5e1";
  const stepWorld = 10;
  for (let wx = 5; wx <= 95; wx += stepWorld) {
    for (let wy = 5; wy <= 95; wy += stepWorld) {
      const px = p.x(wx), py = p.y(wy);
      if (px >= 0 && px <= width && py >= 0 && py <= height) {
        ctx.fillRect(px - 1, py - 1, 2, 2);
      }
    }
  }
  ctx.restore();

  // Transit Corridors (Road Network where vehicles navigate)
  ctx.save();
  // North-South Main Corridor (through C2 and C0 at x = 25)
  const roadWidth = p.side * 0.16;
  ctx.fillStyle = "rgba(226, 232, 240, 0.75)";
  ctx.fillRect(p.x(25) - roadWidth / 2, p.y(0), roadWidth, p.side);
  // East-West Corridor (through C2/C3 at y = 75 and C0/C1 at y = 25)
  ctx.fillRect(p.x(0), p.y(25) - roadWidth / 2, p.side, roadWidth);
  ctx.fillRect(p.x(0), p.y(75) - roadWidth / 2, p.side, roadWidth);

  // Corridor Curbs / Borders
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1.2;
  // N-S Corridor curbs
  ctx.beginPath();
  ctx.moveTo(p.x(25) - roadWidth / 2, p.y(0));
  ctx.lineTo(p.x(25) - roadWidth / 2, p.y(100));
  ctx.moveTo(p.x(25) + roadWidth / 2, p.y(0));
  ctx.lineTo(p.x(25) + roadWidth / 2, p.y(100));
  // E-W Corridor curbs
  ctx.moveTo(p.x(0), p.y(25) - roadWidth / 2);
  ctx.lineTo(p.x(100), p.y(25) - roadWidth / 2);
  ctx.moveTo(p.x(0), p.y(25) + roadWidth / 2);
  ctx.lineTo(p.x(100), p.y(25) + roadWidth / 2);
  ctx.moveTo(p.x(0), p.y(75) - roadWidth / 2);
  ctx.lineTo(p.x(100), p.y(75) - roadWidth / 2);
  ctx.moveTo(p.x(0), p.y(75) + roadWidth / 2);
  ctx.lineTo(p.x(100), p.y(75) + roadWidth / 2);
  ctx.stroke();

  // Road Dash Centerline
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(p.x(25), p.y(0));
  ctx.lineTo(p.x(25), p.y(100));
  ctx.moveTo(p.x(0), p.y(25));
  ctx.lineTo(p.x(100), p.y(25));
  ctx.moveTo(p.x(0), p.y(75));
  ctx.lineTo(p.x(100), p.y(75));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // Cell Boundaries (Orthogonal Grid at x = 50, y = 50)
  ctx.save();
  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(p.x(50), p.y(0));
  ctx.lineTo(p.x(50), p.y(100));
  ctx.moveTo(p.x(0), p.y(50));
  ctx.lineTo(p.x(100), p.y(50));
  ctx.stroke();
  ctx.setLineDash([]);

  // Boundary Labels
  ctx.font = "800 9.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  ctx.fillStyle = "#64748b";
  ctx.letterSpacing = "0.08em";
  ctx.textAlign = "center";
  ctx.fillText("CELL BOUNDARY · HANDOVER FRONTIER", p.x(50), p.y(50) - 7);
  ctx.restore();

  // 2. Translucent RF Coverage Contour Rings from each cell tower
  for (const cell of network.cells) {
    const cx = p.x(cell.x), cy = p.y(cell.y);
    const r1 = p.side * 0.18, r2 = p.side * 0.32, r3 = p.side * 0.46;
    ctx.save();
    ctx.strokeStyle = "rgba(100, 116, 139, 0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, r1, 0, Math.PI * 2);
    ctx.arc(cx, cy, r2, 0, Math.PI * 2);
    ctx.stroke();

    // Handover decision threshold perimeter (r3)
    ctx.strokeStyle = "rgba(228, 43, 12, 0.18)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.arc(cx, cy, r3, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // 3. Background Radio Links for other devices
  for (const ue of network.ues) {
    if (ue.id === focusId) continue;
    const pos = pointFor(ue, previous, alpha, p);
    const cell = network.cells[ue.cell];
    ctx.save();
    ctx.strokeStyle = ue.allocated > 0 ? "rgba(148, 163, 184, 0.55)" : "rgba(239, 68, 68, 0.35)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash(ue.allocated > 0 ? [] : [4, 5]);
    ctx.beginPath();
    ctx.moveTo(p.x(cell.x), p.y(cell.y));
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.restore();
  }

  // 4. Focus UE Active Serving Carrier Beam
  if (focus) {
    const pos = pointFor(focus, previous, alpha, p);
    const cell = network.cells[focus.cell];
    const sx = p.x(cell.x), sy = p.y(cell.y);
    const connected = focus.allocated > 0;

    ctx.save();
    // Carrier beam outer glow
    ctx.strokeStyle = connected ? "rgba(2, 132, 199, 0.2)" : "rgba(239, 68, 68, 0.2)";
    ctx.lineWidth = 9;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    // Carrier beam core
    ctx.strokeStyle = connected ? "#0284c7" : "#ef4444";
    ctx.lineWidth = 3.5;
    ctx.setLineDash(connected ? [] : [7, 6]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.restore();

    // Flowing Data Packets
    if (connected) {
      for (let packet = 0; packet < 3; packet++) {
        const fraction = (visualNow * 0.00038 + packet * 0.33) % 1;
        const px = sx + (pos.x - sx) * fraction;
        const py = sy + (pos.y - sy) * fraction;
        ctx.save();
        ctx.fillStyle = "#0284c7";
        ctx.shadowColor = "rgba(2, 132, 199, 0.6)";
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
  }

  // 5. Anticipatory Reservation Arc (PPO Target Curve)
  const contract = network.contracts.find(item => item.ue === focusId);
  if (focus && contract) {
    const pos = pointFor(focus, previous, alpha, p);
    const target = network.cells[contract.target];
    const tx = p.x(target.x), ty = p.y(target.y);
    const active = contract.active;
    const midX = (pos.x + tx) / 2;
    const midY = (pos.y + ty) / 2 - (p.side * 0.12);

    ctx.save();
    // Glow effect
    ctx.strokeStyle = active ? "rgba(228, 43, 12, 0.22)" : "rgba(245, 158, 11, 0.18)";
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.quadraticCurveTo(midX, midY, tx, ty);
    ctx.stroke();

    // Main line
    ctx.strokeStyle = active ? "#e42b0c" : "#f59e0b";
    ctx.lineWidth = 3.5;
    ctx.setLineDash(active ? [10, 7] : [5, 7]);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.quadraticCurveTo(midX, midY, tx, ty);
    ctx.stroke();
    ctx.restore();

    // Animated transmission particle along curve
    const tFrac = (visualNow * 0.00045) % 1;
    const p1x = (1 - tFrac) * pos.x + tFrac * midX;
    const p1y = (1 - tFrac) * pos.y + tFrac * midY;
    const p2x = (1 - tFrac) * midX + tFrac * tx;
    const p2y = (1 - tFrac) * midY + tFrac * ty;
    const qx = (1 - tFrac) * p1x + tFrac * p2x;
    const qy = (1 - tFrac) * p1y + tFrac * p2y;

    ctx.save();
    ctx.fillStyle = active ? "#e42b0c" : "#f59e0b";
    ctx.shadowColor = active ? "rgba(228, 43, 12, 0.8)" : "rgba(245, 158, 11, 0.8)";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(qx, qy, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Floating Badge for Reservation
    const badgeX = tx < width / 2 ? tx + 115 : tx - 115;
    const badgeText = `${contract.amount} RB ${active ? "HELD" : "REQUESTED"}`;
    roundedLabel(
      ctx, badgeText,
      clamp(badgeX, 90, width - 90),
      clamp(ty - 88, 30, height - 30),
      active ? "#0f172a" : "#1e293b",
      "#ffffff",
      active ? "#e42b0c" : "#f59e0b"
    );
  }

  // 6. Cell Base Stations (Towers & Segmented Capacity Meters)
  for (const cell of network.cells) {
    const x = p.x(cell.x), y = p.y(cell.y);
    const selected = cell.id === app.selectedCell;

    // Ground platform shadow & pad
    ctx.save();
    ctx.fillStyle = "rgba(15, 23, 42, 0.06)";
    ctx.beginPath();
    ctx.ellipse(x, y + 40, 36, 12, 0, 0, Math.PI * 2);
    ctx.fill();

    // Selection Glow Ring
    if (selected) {
      ctx.fillStyle = "rgba(228, 43, 12, 0.08)";
      ctx.beginPath();
      ctx.arc(x, y, 74, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = "#e42b0c";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(x, y, 74, 0, Math.PI * 2);
      ctx.stroke();

      // Delicate radar pulse ripple
      const ripplePhase = (visualNow * 0.001) % 1;
      ctx.strokeStyle = `rgba(228, 43, 12, ${0.4 * (1 - ripplePhase)})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 74 + ripplePhase * 26, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();

    // Tower graphic
    if (towerImage.complete && towerImage.naturalWidth) {
      ctx.drawImage(towerImage, x - 55, y - 68, 110, 121);
    } else {
      ctx.fillStyle = "#334155";
      ctx.fillRect(x - 12, y - 38, 24, 76);
    }

    // Tower Identifier Badge: "C0", "C1", etc.
    const badgeW = 66, badgeH = 26;
    ctx.save();
    ctx.shadowColor = "rgba(15, 23, 42, 0.12)";
    ctx.shadowBlur = 6;
    ctx.shadowOffsetY = 2;
    ctx.fillStyle = selected ? "#0f172a" : "#ffffff";
    ctx.beginPath();
    ctx.roundRect(x - badgeW / 2, y + 36, badgeW, badgeH, 6);
    ctx.fill();

    ctx.shadowColor = "transparent";
    ctx.strokeStyle = selected ? "#e42b0c" : "#cbd5e1";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = selected ? "#ffffff" : "#0f172a";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "800 12.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillText(`C${cell.id}`, x, y + 49);
    ctx.restore();

    // Segmented Capacity Gauge HUD
    const barW = 66, barH = 8, barX = x - barW / 2, barY = y + 68;
    ctx.save();
    ctx.fillStyle = "#e2e8f0";
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, 4);
    ctx.fill();

    // Sliced allocation segments
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
          ctx.roundRect(currentX, barY, segW, barH, 2);
          ctx.fill();
          currentX += segW;
        }
      }
    } else if (cell.allocated > 0) {
      const allocW = (barW * cell.allocated) / cell.capacity;
      ctx.fillStyle = "#334155";
      ctx.beginPath();
      ctx.roundRect(barX, barY, allocW, barH, 4);
      ctx.fill();
      currentX += allocW;
    }

    // Held / Reserved RBs (Glowing Vermillion)
    if (cell.reserved > 0) {
      const heldW = (barW * cell.reserved) / cell.capacity;
      ctx.fillStyle = "#ea580c";
      ctx.shadowColor = "rgba(234, 88, 12, 0.6)";
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.roundRect(currentX, barY, heldW, barH, 2);
      ctx.fill();
    }
    ctx.restore();

    // Capacity text label below gauge
    ctx.save();
    ctx.font = "800 9px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    ctx.fillStyle = selected ? "#e42b0c" : "#64748b";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`${cell.allocated + cell.reserved}/${cell.capacity} RB`, x, barY + 11);
    ctx.restore();
  }

  // 7. Handover Admitted Transition Animation (Shockwave & Banner)
  if (app.pulse && visualNow - app.pulse.at < 3200) {
    const elapsed = visualNow - app.pulse.at;
    const progress = clamp(elapsed / 3200, 0, 1);
    const source = network.cells[app.pulse.from], target = network.cells[app.pulse.to];
    const sx = p.x(source.x), sy = p.y(source.y) - 10;
    const tx = p.x(target.x), ty = p.y(target.y) - 10;
    const midX = (sx + tx) / 2, midY = (sy + ty) / 2 - 56;

    ctx.save();
    ctx.globalAlpha = 1 - progress * 0.6;
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 5;
    ctx.shadowColor = "rgba(16, 185, 129, 0.7)";
    ctx.shadowBlur = 12;

    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.quadraticCurveTo(midX, midY - 25, tx, ty);
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(ty - (midY - 25), tx - midX);
    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - 18 * Math.cos(angle - 0.45), ty - 18 * Math.sin(angle - 0.45));
    ctx.lineTo(tx - 18 * Math.cos(angle + 0.45), ty - 18 * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Top Celebration Banner
    roundedLabel(ctx, "PPO HANDOVER ADMITTED", width / 2, 42, "#047857", "#ffffff", "#10b981");
  }

  // 8. Focus Vehicle Breadcrumb Motion Trail
  if (app.trail && app.trail.length > 1) {
    ctx.save();
    ctx.beginPath();
    for (let i = 0; i < app.trail.length; i++) {
      const pt = app.trail[i];
      const sx = p.x(pt.x), sy = p.y(pt.y);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.strokeStyle = "rgba(228, 43, 12, 0.35)";
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Small trajectory beads along trail
    for (let i = 0; i < app.trail.length; i++) {
      const pt = app.trail[i];
      const opacity = ((i + 1) / app.trail.length) * 0.45;
      ctx.fillStyle = `rgba(228, 43, 12, ${opacity})`;
      ctx.beginPath();
      ctx.arc(p.x(pt.x), p.y(pt.y), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // 9. Devices Rendering (Vehicles, Drones, Robots)
  const focusPosition = focus ? pointFor(focus, previous, alpha, p) : null;
  const deviceOrder = [...network.ues].sort((a, b) => Number(a.id === focusId) - Number(b.id === focusId));

  for (const ue of deviceOrder) {
    const pos = pointFor(ue, previous, alpha, p);
    const prior = previous?.ues.find(item => item.id === ue.id) || ue;
    const quality = prior.signal_quality + (ue.signal_quality - prior.signal_quality) * alpha;

    // Quality color gradient
    const ringColor = quality < 0.48 ? "#ef4444" : quality < 0.68 ? "#f59e0b" : "#10b981";
    const focused = ue.id === focusId;
    const iconSize = focused ? 48 : 25;
    const radius = focused ? 34 : 18;

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

    // Near cell edge warning halo for focus vehicle
    if (focused && quality < 0.48) {
      const pulsePhase = (visualNow * 0.003) % 1;
      ctx.strokeStyle = `rgba(239, 68, 68, ${0.7 * (1 - pulsePhase)})`;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius + 10 + pulsePhase * 18, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Directional velocity heading vector
    if (focused && (ue.vx !== 0 || ue.vy !== 0)) {
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
    ctx.shadowColor = focused ? "rgba(15, 23, 42, 0.2)" : "rgba(15, 23, 42, 0.08)";
    ctx.shadowBlur = focused ? 12 : 5;
    ctx.shadowOffsetY = focused ? 3 : 1;

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowColor = "transparent";

    // Signal Quality Stroke
    ctx.strokeStyle = ringColor;
    ctx.lineWidth = focused ? 4 : 2.5;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
    ctx.stroke();

    if (focused) {
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius + 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Icon render
    const icon = icons[ue.entity_type];
    if (icon && icon.complete && icon.naturalWidth) {
      ctx.drawImage(icon, pos.x - iconSize / 2, pos.y - iconSize / 2, iconSize, iconSize);
    } else {
      ctx.fillStyle = "#334155";
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 9, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // Floating Target HUD Tag
    if (focused) {
      roundedLabel(
        ctx,
        `UE${ue.id} · ${ue.slice} CAR`,
        clamp(pos.x, 90, width - 90),
        clamp(pos.y + 56, 24, height - 24),
        "#0f172a",
        "#ffffff"
      );
    } else if (!focusPosition || Math.hypot(pos.x - focusPosition.x, pos.y - focusPosition.y) > 66) {
      ctx.save();
      ctx.fillStyle = "#475569";
      ctx.textAlign = "center";
      ctx.font = "800 10.5px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
      ctx.fillText(`UE${ue.id}`, pos.x, pos.y + 28);
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

function buildSparklinePath(historyList, key, width = 180, height = 32, padY = 2) {
  if (!historyList || historyList.length < 2) {
    return `M 0,${height - padY} L ${width},${height - padY}`;
  }
  const n = historyList.length;
  const usableH = height - padY * 2;
  const points = historyList.map((item, idx) => {
    const x = (idx / (n - 1)) * width;
    const val = clamp(item[key] ?? 0, 0, 1);
    const y = (height - padY) - val * usableH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `M ${points.join(" L ")}`;
}

function updateSparklines(history) {
  if (!history) return;
  const w = 180, h = 32;
  const nHsr = $("hsrSparkNormal"), pHsr = $("hsrSparkProposed");
  const nQos = $("qosSparkNormal"), pQos = $("qosSparkProposed");

  if (nHsr && pHsr && history.normal && history.proposed) {
    nHsr.setAttribute("d", buildSparklinePath(history.normal, "hsr", w, h));
    pHsr.setAttribute("d", buildSparklinePath(history.proposed, "hsr", w, h));
  }
  if (nQos && pQos && history.normal && history.proposed) {
    nQos.setAttribute("d", buildSparklinePath(history.normal, "qos", w, h));
    pQos.setAttribute("d", buildSparklinePath(history.proposed, "qos", w, h));
  }
}

function focusHandover(events, focusId) {
  return events.find(event => event.startsWith(`HO UE${focusId} `));
}

function updateTeaching(snapshot) {
  const focusId = snapshot.focus_id;
  const proposed = snapshot.proposed;
  const normal = snapshot.normal;
  const proposedEvent = focusHandover(proposed.events, focusId);
  const normalEvent = focusHandover(normal.events, focusId);
  const contract = proposed.contracts.find(item => item.ue === focusId);

  if (proposedEvent) {
    const parsed = proposedEvent.match(/^HO UE\d+ C(\d+)->C(\d+) (SUCCESS|FAIL) \((\d+)\/(\d+)RB\)/);
    const baseline = normalEvent?.match(/(SUCCESS|FAIL) \((\d+)\/(\d+)RB\)/);
    if (parsed) {
      app.outcome = {
        normal: baseline ? {
          success: baseline[1] === "SUCCESS",
          used: Number(baseline[2]),
          demand: Number(baseline[3]),
        } : null,
        proposed: {
          success: parsed[3] === "SUCCESS",
          used: Number(parsed[4]),
          demand: Number(parsed[5]),
        },
        from: Number(parsed[1]),
        to: Number(parsed[2]),
      };
      app.outcomeStep = proposed.step;
      app.pulse = app.outcome.proposed.success
        ? {from: app.outcome.from, to: app.outcome.to, at: performance.now()}
        : null;
    }
  }

  let number = "01", label = "APPROACH", title = "Follow the highlighted car.";
  let explanation = "A solid line is active RB service. Red around the car means it is near the edge of its serving cell in the illustrative signal proxy.";
  let stepIndex = 0;

  if (app.outcome && proposed.step - app.outcomeStep < 18) {
    number = "04";
    label = "HANDOVER RESULT";
    stepIndex = 3;
    const {normal: base, proposed: ppo, from, to} = app.outcome;
    title = `Car moves C${from} → C${to}: ${ppo.success ? "PPO admits it" : "PPO blocks it"}.`;
    explanation = `Same arrival: no reservation could offer only ${base?.used ?? "?"}/${base?.demand ?? "?"} RB (${base?.success ? "admitted" : "blocked"}); PPO granted ${ppo.used}/${ppo.demand} RB (${ppo.success ? "admitted" : "blocked"}). ${ppo.used === ppo.demand ? "Full demand was met." : "Full-demand QoS may still fail."}`;
  } else if (contract?.active) {
    number = "03";
    label = "CAPACITY HELD";
    stepIndex = 2;
    title = `Tower C${contract.target} now holds ${contract.amount} RBs.`;
    explanation = "The orange path connects the car to its forecast target. The target's RB ledger below shows what was held before arrival.";
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
}

function acceptSnapshot(snapshot) {
  app.previous = app.snapshot;
  app.snapshot = snapshot;
  app.frameStart = performance.now();
  setText("simTime", `${snapshot.proposed.time.toFixed(1)} s`);

  const focus = snapshot.proposed.ues.find(u => u.id === snapshot.focus_id);
  if (focus) {
    app.trail.push({x: focus.x, y: focus.y});
    if (app.trail.length > 25) app.trail.shift();
  }

  updateTeaching(snapshot);
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
    if (generation === app.generation) acceptSnapshot(snapshot);
  } catch (error) {
    if (generation === app.generation) showError(error);
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
  $("zoomIn").addEventListener("click", () => zoomTo(app.view.zoom + 0.25));
  $("zoomOut").addEventListener("click", () => zoomTo(app.view.zoom - 0.25));
  $("fitMap").addEventListener("click", () => {
    app.view = {zoom: 1, cx: 50, cy: 50};
    clampView();
  });

  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomTo(app.view.zoom + (event.deltaY < 0 ? 0.2 : -0.2), {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
  }, {passive: false});

  canvas.addEventListener("pointerdown", event => {
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("dragging");
    app.drag = {x: event.clientX, y: event.clientY, moved: false};
  });

  canvas.addEventListener("pointermove", event => {
    const rect = canvas.getBoundingClientRect();
    const p = projection(rect.width, rect.height);
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;

    if (app.drag) {
      const dx = event.clientX - app.drag.x;
      const dy = event.clientY - app.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) app.drag.moved = true;
      app.view.cx -= (dx * 100) / p.side;
      app.view.cy -= (dy * 100) / p.side;
      clampView();
      app.drag.x = event.clientX;
      app.drag.y = event.clientY;
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
  });

  canvas.addEventListener("pointerleave", () => {
    app.hover = null;
  });

  canvas.addEventListener("pointerup", event => {
    canvas.classList.remove("dragging");
    const wasClick = app.drag && !app.drag.moved;
    app.drag = null;
    if (!wasClick || !app.snapshot) return;
    const rect = canvas.getBoundingClientRect();
    const p = projection(rect.width, rect.height);
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const closest = app.snapshot.proposed.cells
      .map(cell => ({
        id: cell.id,
        distance: Math.hypot(p.x(cell.x) - x, p.y(cell.y) - y),
      }))
      .sort((a, b) => a.distance - b.distance)[0];
    if (closest && closest.distance < 72) selectTower(closest.id);
  });

  canvas.addEventListener("pointercancel", () => {
    app.drag = null;
    canvas.classList.remove("dragging");
  });

  const nav = $("towerNav");
  for (let id = 0; id < 4; id++) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.cell = String(id);
    button.textContent = `C${id}`;
    button.setAttribute("aria-label", `Inspect tower C${id}`);
    button.addEventListener("click", () => selectTower(id));
    nav.append(button);
  }
  clampView();
}

async function init() {
  setupMapInteractions();
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
