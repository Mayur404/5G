"use strict";
const $ = id => document.getElementById(id);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const setText = (id, value) => {
  const el = $(id);
  if (el) el.textContent = value;
};
const STEP_MS = 750;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const FONT_FAMILY = getComputedStyle(document.body).fontFamily;
const MAP_FONT = "12px " + FONT_FAMILY;
const MIN_ZOOM = .5, MAX_ZOOM = 8;
const COLORS = {allocated:"#2563eb", held:"#b45309", free:"#dbeafe", strong:"#15803d", fair:"#b45309", weak:"#dc2626"};
const CELL_FILLS = ["#e0edff", "#e8f7ee", "#f3eaff", "#fff2dc", "#e0f5f5", "#ffe9ee", "#eef1ff"];
const SLICE_COLORS={eMBB:"#8b5cf6",uRLLC:"#2563eb",mMTC:"#0d9da5"};
const policyLabel=()=>app.snapshot?.controller_kind==="linear_ppo"?"PPO":"Neural";
function signalState(ue) {
  const quality = clamp(ue.signal_quality, 0, 1);
  const level = quality >= .7 ? "strong" : quality >= .4 ? "fair" : "weak";
  return {quality, level, color:COLORS[level], text:level[0].toUpperCase()+level.slice(1)+" · "+Math.floor(quality*100)+"%"};
}
function towerLabel(ctx, cell, x, y) {
  const compact=app.mapCompact,w=compact?96:148, h=compact?28:48;
  ctx.fillStyle="#ffffff";
  ctx.strokeStyle=cell.free===0?"#f59e0b":"#b8cbea";
  ctx.lineWidth=1;
  ctx.beginPath();ctx.roundRect(x-w/2,y-h/2,w,h,8);ctx.fill();ctx.stroke();
  ctx.textAlign="center";ctx.textBaseline="middle";
  ctx.font="500 "+(compact?10:11)+"px "+FONT_FAMILY;ctx.fillStyle="#315e9b";
  ctx.fillText("C"+cell.id+" · "+cell.allocated+"/"+cell.capacity+" RB"+(compact?"":" allocated"),x,y-(compact?3:10));
  ctx.font=(compact?9:10)+"px "+FONT_FAMILY;ctx.fillStyle="#7a8aa1";
  if(!compact)ctx.fillText(cell.reserved+" held · "+cell.free+" free",x,y+4);
  const left=x-w/2+12, barW=w-24;
  const barY=y+(compact?7:15);
  ctx.fillStyle=COLORS.free;ctx.fillRect(left,barY,barW,3);
  let offset=0;
  for(const slice of cell.slices) {
    const part=barW*slice.allocated/cell.capacity;
    ctx.fillStyle=SLICE_COLORS[slice.name];ctx.fillRect(left+offset,barY,part,3);offset+=part;
  }
  ctx.fillStyle="#f59e0b";ctx.fillRect(left+barW*cell.allocated/cell.capacity,barY,barW*cell.reserved/cell.capacity,3);
}
const icons = Object.fromEntries(["car", "drone", "robot", "tower"].map(type => {
  const icon = new Image();
  icon.addEventListener("load", () => { app.dirty = true; });
  icon.src = "/static/icons/" + type + ".svg";
  return [type, icon];
}));
const app = {
  token: null, snapshot: null, previous: null, pendingSnapshot: null, scenarios: null,
  playing: true, busy: false, generation: 0, timer: null,
  frameStart: performance.now(), pausedProgress: 0, pauseWall: null,
  selectedCell: 0, selectedUEId: null, view: {
    zoom: 1, cx: 50, cy: 50
  },
  autoTrack: false, drag: null, hover: null, events: [],
  outcomes: new Map(), layers: {
    hex: true, labels: true, backhaul: false
  },
  cellsKey: null, polygons: [], renderPoints: new Map(), labelHits: [], dirty: true,
  connectionLinks: [], handoverKeys: new Set(), visibleHandovers: [], handoverRoutes: [],
  inspector: "Device", pointers: new Map(), pinch: null, expanded: false,
};
async function requestJSON(url, payload) {
  const response = await fetch(url, payload ? {
    method: "POST", headers: {
      "Content-Type": "application/json"
    }, body: JSON.stringify(payload),
  } : {
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "HTTP " + response.status);
  return body;
}
function setPlaybackButton() {
  setText("playIcon", app.playing ? "Ⅱ" : "▶");
  setText("playText", app.playing ? "Pause" : "Resume");
  $("playBtn").setAttribute("aria-pressed", String(!app.playing));
  $("playBtn").disabled = !app.token;
  $("restartBtn").disabled = !app.scenarios;
}
function showError(error) {
  app.playing = false;
  clearTimeout(app.timer);
  setPlaybackButton();
  setText("phaseLabel", "SIMULATION STOPPED");
  setText("phaseTitle", "The simulation could not continue.");
  setText("phaseText", error.message || String(error));
  const notice = $("errorNotice");
  notice.textContent = (error.message || String(error)) + ". Choose a scenario to start a fresh session.";
  notice.hidden = false;
}
function canvasSize() {
  const canvas = $("networkCanvas"), rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    app.dirty = true;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {
    ctx, width: rect.width, height: rect.height
  };
}
function projection(width, height) {
  const world = app.snapshot?.world_size || 100;
  const horizontal=window.matchMedia("(max-width:650px)").matches;
  const centerX=width/2,centerY=height/2+(horizontal?12:0);
  const side = Math.max(1, Math.min(width-44, height-(horizontal?88:50))) * app.view.zoom;
  const x = value => centerX + (value - app.view.cx) * side / world;
  const y = value => centerY + (value - app.view.cy) * side / world;
  return {
    side, x, y, centerX, centerY, worldX: px => app.view.cx + (px - centerX) * world / side,
    worldY: py => app.view.cy + (py - centerY) * world / side
  };
}
function clampView() {
  const world = app.snapshot?.world_size || 100;
  app.view.cx = clamp(app.view.cx, -world / 4, world * 1.25);
  app.view.cy = clamp(app.view.cy, -world / 4, world * 1.25);
  setText("zoomLevel", Math.round(app.view.zoom * 100) + "%");
  $("zoomOut").disabled = app.view.zoom <= MIN_ZOOM;
  $("zoomIn").disabled = app.view.zoom >= MAX_ZOOM;
  $("trackCar").classList.toggle("active", app.autoTrack);
  $("trackCar").setAttribute("aria-pressed", String(app.autoTrack));
  app.dirty = true;
}
function zoomTo(next, anchor) {
  const rect = $("networkCanvas").getBoundingClientRect(), before = projection(rect.width, rect.height);
  const px = anchor?.x ?? before.centerX, py = anchor?.y ?? before.centerY;
  const wx = before.worldX(px), wy = before.worldY(py), world = app.snapshot?.world_size || 100;
  app.view.zoom = clamp(next, MIN_ZOOM, MAX_ZOOM);
  const after = projection(rect.width, rect.height);
  app.view.cx = wx - (px - after.centerX) * world / after.side;
  app.view.cy = wy - (py - after.centerY) * world / after.side;
  app.autoTrack = false;
  clampView();
}
function fitMap() {
  const center = (app.snapshot?.world_size || 100) / 2;
  app.autoTrack = false;
  app.view = {
    zoom: 1, cx: center, cy: center
  };
  clampView();
}
// Half-plane clipping produces exactly the nearest-site Voronoi regions used
// by the simulator, including non-hexagonal boundary cells and world edges.
function cellPolygons(cells, world) {
  return cells.map(cell => {
    let vertices = [[0,0], [world,0], [world,world], [0,world]];
    for (const other of cells) {
      if (other.id === cell.id) continue;
      const nx = other.x - cell.x, ny = other.y - cell.y;
      const limit = (other.x ** 2 + other.y ** 2 - cell.x ** 2 - cell.y ** 2) / 2, clipped = [];
      for (let i = 0; i < vertices.length; i++) {
        const a = vertices[i], b = vertices[(i + 1) % vertices.length];
        const da = nx * a[0] + ny * a[1] - limit, db = nx * b[0] + ny * b[1] - limit;
        if (da <= 1e-9) clipped.push(a);
        if ((da <= 0) !== (db <= 0)) {
          const f = da / (da - db);
          clipped.push([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]);
        }
      }
      vertices = clipped;
    }
    return vertices;
  });
}
function mapLabel(ctx, text, x, y, dark = false, compact = false) {
  ctx.font = compact?"10px "+FONT_FAMILY:MAP_FONT;
  const w = ctx.measureText(text).width + 16, h = compact?20:28;
  ctx.fillStyle = dark ? "#173454" : "#ffffff";
  ctx.strokeStyle = dark ? "#173454" : "#cbd9eb";
  ctx.beginPath();
  ctx.roundRect(x - w / 2, y - h / 2, w, h, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = dark ? "#fff" : "#101828";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
}
function drawTower(ctx, cell, x, y, selected) {
  ctx.save();
  if(icons.tower.complete && icons.tower.naturalWidth) {
    ctx.fillStyle=selected?"#e9f2ff":"#ffffffb8";
    ctx.strokeStyle=selected?"#93b8ee":"#ffffff";
    ctx.lineWidth=1.5;
    ctx.beginPath();ctx.ellipse(x,y,25,24,0,0,Math.PI*2);ctx.fill();ctx.stroke();
    const towerW=app.mapCompact?42:50,towerH=app.mapCompact?48:57;
    ctx.drawImage(icons.tower,x-towerW/2,y-towerH/2,towerW,towerH);
    ctx.font="600 9px "+FONT_FAMILY;ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillStyle="#fff";ctx.strokeStyle="#cbd9eb";
    ctx.beginPath();ctx.roundRect(x+10,y+13,23,16,4);ctx.fill();ctx.stroke();
    ctx.fillStyle="#315e9b";ctx.fillText("C"+cell.id,x+21.5,y+21);
    ctx.restore();
    return;
  }
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = selected ? "#2563eb" : "#7c9bc4";
  ctx.lineWidth = selected ? 2 : 1;
  ctx.beginPath();
  ctx.roundRect(x - 16, y - 18, 32, 36, 8);
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x - 7, y + 11);
  ctx.lineTo(x, y - 10);
  ctx.lineTo(x + 7, y + 11);
  ctx.moveTo(x - 4, y + 3);
  ctx.lineTo(x + 4, y + 3);
  ctx.moveTo(x, y - 10);
  ctx.lineTo(x, y + 11);
  ctx.stroke();
  ctx.fillStyle = cell.free === 0 ? "#f59e0b" : "#2563eb";
  ctx.beginPath();
  ctx.arc(x, y - 11, 2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}
function drawMapLabels(ctx, labels, obstacles, width, height, keepHits=false) {
  ctx.font = MAP_FONT;
  // Reserve the screen controls as obstacles as well as map objects.
  const controls=app.controlsHorizontal?{left:0,right:width,top:0,bottom:52}:{left:width-70,right:width,top:0,bottom:230};
  const occupied = [...obstacles,controls,{left:0,right:100,top:height-34,bottom:height}];
  if(!app.controlsHorizontal)occupied.push({left:0,right:220,top:0,bottom:42});
  if(!keepHits)app.labelHits=[];
  for (const label of labels) {
    if (label.x < -24 || label.x > width+24 || label.y < -24 || label.y > height+24) continue;
    ctx.font=label.compact?"10px "+FONT_FAMILY:MAP_FONT;
    const w = label.cell ? (app.mapCompact?96:148) : ctx.measureText(label.text).width + 16, h = label.cell ? (app.mapCompact?28:48) : label.compact?20:28;
    let best = null;
    const candidates=[[0,0]];
    const radii=label.device?[h+8,60,88]:[h+8,88,128,176];
    for(const radius of radii)for(const [dx,dy] of [[0,-1],[0,1],[-1,0],[1,0],[-.8,-.7],[.8,-.7],[-.8,.7],[.8,.7]])candidates.push([dx*radius,dy*radius]);
    for (const [dx, dy] of candidates) {
      const x = clamp(label.x + dx, w/2+8, width-w/2-8);
      const y = clamp(label.y + dy, h/2+8, height-h/2-8);
      const box = {
        left:x-w/2-3,right:x+w/2+3,top:y-h/2-3,bottom:y+h/2+3
      };
      const overlap = occupied.reduce((total,b)=>total+(label.cell&&b.kind==="ue"&&!b.selected?0:1)*Math.max(0,Math.min(box.right,b.right)-Math.max(box.left,b.left))*Math.max(0,Math.min(box.bottom,b.bottom)-Math.max(box.top,b.top)),0);
      const score = overlap*10+Math.hypot(x-label.x,y-label.y);
      if (!best || score < best.score) best = {
        x,y,box,score
      };
    }
    // Labels never introduce extra lines that could be mistaken for connections.
    if(label.cell) towerLabel(ctx,label.cell,best.x,best.y);
    else mapLabel(ctx,label.text,best.x,best.y,label.dark,label.compact);
    if(label.cell) app.labelHits.push({...best.box,id:label.cell.id});
    occupied.push(best.box);
  }
}
function drawMap(now) {
  const {
    ctx, width, height
  } = canvasSize();
  ctx.clearRect(0, 0, width, height);
  app.mapCompact=width<600;
  app.controlsHorizontal=window.matchMedia("(max-width:650px)").matches;
  ctx.fillStyle = "#f1f6fc";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle="#cbd9eb";
  for(let gx=12;gx<width;gx+=24)for(let gy=12;gy<height;gy+=24) {
    ctx.beginPath();ctx.arc(gx,gy,.65,0,Math.PI*2);ctx.fill();
  }
  if (!app.snapshot) {
    ctx.fillStyle="#4a5565";
    ctx.font="16px system-ui";
    ctx.textAlign="center";
    ctx.fillText("Loading simulation…",width/2,height/2);
    return;
  }
  const network = app.snapshot.proposed, world = app.snapshot.world_size;
  const selectedId = app.selectedUEId ?? app.snapshot.focus_id, selected = network.ues.find(ue => ue.id === selectedId);
  const elapsed = app.playing ? clamp((now - app.frameStart) / STEP_MS, 0, 1) : app.pausedProgress;
  // Paused inspection shows the exact completed sample, keeping positions,
  // nearest-site assignment, and the displayed resource ledger in agreement.
  const alpha = reducedMotion.matches || !app.playing ? 1 : elapsed;
  const displayedUEs=network.ues.map(ue=>{
    const previous=app.previous?.proposed.ues.find(u=>u.id===ue.id);
    const x=previous?previous.x+(ue.x-previous.x)*alpha:ue.x;
    const y=previous?previous.y+(ue.y-previous.y)*alpha:ue.y;
    let cell=ue.cell;
    if(previous&&previous.cell!==ue.cell&&alpha<1) {
      // Nearest-site changes happen at the bisector. QoS changes are discrete
      // decisions at the completed sample, after the time-to-trigger check.
      const nearest=network.cells.reduce((best,c)=>Math.hypot(x-c.x,y-c.y)<Math.hypot(x-best.x,y-best.y)?c:best).id;
      if(app.snapshot.handover_mode==="qos_aware"||nearest!==ue.cell)cell=previous.cell;
    }
    const sample=cell===ue.cell?ue:previous,tower=network.cells.find(c=>c.id===cell);
    return {...sample,id:ue.id,x,y,cell,signal_quality:clamp(1-Math.hypot(x-tower.x,y-tower.y)/(app.snapshot.signal_radius||48),0,1)};
  });
  app.displayedUEs=displayedUEs;
  const oldUE = app.previous?.proposed.ues.find(ue => ue.id === selectedId);
  if (app.autoTrack && selected) {
    app.view.cx = oldUE ? oldUE.x + (selected.x - oldUE.x) * alpha : selected.x;
    app.view.cy = oldUE ? oldUE.y + (selected.y - oldUE.y) * alpha : selected.y;
  }
  const p = projection(width, height), key = JSON.stringify([network.cells.map(c => [c.x,c.y]),app.snapshot.cell_polygons,world]);
  const scaleTarget=80*world/p.side,scalePower=10**Math.floor(Math.log10(scaleTarget));
  const scaleUnits=[1,2,5,10].map(n=>n*scalePower).reduce((best,n)=>Math.abs(n-scaleTarget)<Math.abs(best-scaleTarget)?n:best,scalePower);
  $("mapScale").style.width=(scaleUnits*p.side/world)+"px";
  setText("mapScale",scaleUnits+" world units");
  const labels = [], obstacles = [];
  if (key !== app.cellsKey) {
    app.cellsKey = key;
    app.polygons = app.snapshot.cell_polygons || cellPolygons(network.cells, world);
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(p.x(0), p.y(0), p.side, p.side);
  ctx.clip();
  if (app.layers.hex) for (let i = 0; i < network.cells.length; i++) {
    const cell = network.cells[i], polygon = app.polygons[i];
    if (!polygon.length) continue;
    ctx.beginPath();
    polygon.forEach(([x,y],j)=>j?ctx.lineTo(p.x(x),p.y(y)):ctx.moveTo(p.x(x),p.y(y)));
    ctx.closePath();
    ctx.fillStyle = CELL_FILLS[i % CELL_FILLS.length];
    ctx.strokeStyle = cell.id === selected?.cell ? "#7095cb" : "#c1cddd";
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  }
  if (app.layers.backhaul && app.view.zoom>=1) {
    const distances=network.cells.flatMap((a,i)=>network.cells.slice(i+1).map(b=>Math.hypot(a.x-b.x,a.y-b.y))),min=Math.min(...distances);
    ctx.strokeStyle="#99a1af";
    ctx.lineWidth=1;
    ctx.setLineDash([3,5]);
    for(let i=0;i<network.cells.length;i++)for(const b of network.cells.slice(i+1)) {
      const a=network.cells[i];
      if(Math.hypot(a.x-b.x,a.y-b.y)>min*1.1)continue;
      ctx.beginPath();
      ctx.moveTo(p.x(a.x),p.y(a.y));
      ctx.lineTo(p.x(b.x),p.y(b.y));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.font="14px system-ui";
    ctx.fillStyle="#676767";
    ctx.textAlign="left";
    ctx.fillText("Illustrative site connections",Math.max(16,p.x(0)+16),Math.max(20,p.y(0)+20));
  }
  app.renderPoints.clear();
  for(const ue of displayedUEs) {
    app.renderPoints.set(ue.id, {
      x:p.x(ue.x),y:p.y(ue.y)
    });
  }
  app.connectionLinks=[];
  for(const ue of displayedUEs) {
    const tower=network.cells.find(c=>c.id===ue.cell),point=app.renderPoints.get(ue.id),isSelected=ue.id===selectedId;
    if(!tower)continue;
    const start={x:p.x(tower.x),y:p.y(tower.y)};
    app.connectionLinks.push({ue:ue.id,cell:ue.cell,allocated:ue.allocated,start,end:point});
    ctx.strokeStyle=ue.allocated>0?(isSelected?"#2366dc":"#729bd4"):"#a86a76";
    ctx.globalAlpha=isSelected?1:.7;
    ctx.lineWidth=isSelected?2.5:1.25;
    ctx.setLineDash(ue.allocated>0?[]:[3,4]);
    ctx.beginPath();ctx.moveTo(start.x,start.y);ctx.lineTo(point.x,point.y);ctx.stroke();
  }
  ctx.globalAlpha=1;ctx.setLineDash([]);
  if(selected) {
    const point=app.renderPoints.get(selectedId),tower=network.cells.find(c=>c.id===selected.cell),contract=network.contracts.find(c=>c.ue===selectedId);
    if(contract) {
      const target=network.cells.find(c=>c.id===contract.target);
      ctx.strokeStyle="#f59e0b";
      ctx.lineWidth=contract.active?2.5:1.5;
      ctx.setLineDash([7,5]);
      ctx.beginPath();
      ctx.moveTo(point.x,point.y);
      ctx.lineTo(p.x(target.x),p.y(target.y));
      ctx.stroke();
      ctx.setLineDash([]);
      labels.push( {
        text:contract.amount+" RB "+(contract.active?"held":"pending"),x:(point.x+p.x(target.x))/2,y:(point.y+p.y(target.y))/2-16
      });
    }
  }
  // One current transition for the inspected device; history stays in the inspector.
  const displayedSelected=displayedUEs.find(u=>u.id===selectedId);
  app.visibleHandovers=(network.recent_handovers||network.handover_details).filter(event=>event.ue===selectedId&&network.step-(event.step??network.step)<8&&(!(event.step===network.step)||displayedSelected.cell===event.target)).slice(-1);
  app.handoverRoutes=[];
  for(const result of app.visibleHandovers) {
    const source=network.cells.find(c=>c.id===result.source),target=network.cells.find(c=>c.id===result.target);
    const ax=p.x(source.x),ay=p.y(source.y),bx=p.x(target.x),by=p.y(target.y),angle=Math.atan2(by-ay,bx-ax);
    const clearance=Math.min(34,Math.hypot(bx-ax,by-ay)/4);
    const sx=ax+clearance*Math.cos(angle),sy=ay+clearance*Math.sin(angle),ex=bx-clearance*Math.cos(angle),ey=by-clearance*Math.sin(angle);
    app.handoverRoutes.push({ue:result.ue,source:result.source,target:result.target,start:{x:sx,y:sy},end:{x:ex,y:ey},targetCenter:{x:bx,y:by},clearance});
    const bend=result.ue===selectedId?22:14,cx=(sx+ex)/2-Math.sin(angle)*bend,cy=(sy+ey)/2+Math.cos(angle)*bend;
    const tipAngle=Math.atan2(ey-cy,ex-cx);
    ctx.strokeStyle=result.success?COLORS.strong:COLORS.weak;ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=result.ue===selectedId?3:2;ctx.setLineDash([9,5]);
    ctx.globalAlpha=result.ue===selectedId?1:.6;
    ctx.beginPath();ctx.moveTo(sx,sy);ctx.quadraticCurveTo(cx,cy,ex,ey);ctx.stroke();ctx.setLineDash([]);
    ctx.beginPath();ctx.moveTo(ex,ey);ctx.lineTo(ex-11*Math.cos(tipAngle-.5),ey-11*Math.sin(tipAngle-.5));ctx.lineTo(ex-11*Math.cos(tipAngle+.5),ey-11*Math.sin(tipAngle+.5));ctx.closePath();ctx.fill();
    ctx.globalAlpha=1;
    if(result.ue===selectedId)labels.push({text:"Handover C"+result.source+" → C"+result.target+" · "+(result.success?"admitted":"blocked"),x:(ax+bx)/2,y:(ay+by)/2-20});
  }
  ctx.restore();
  ctx.strokeStyle="#d4d4d4";
  ctx.lineWidth=1;
  if(!app.snapshot.cell_polygons)ctx.strokeRect(p.x(0),p.y(0),p.side,p.side);
  for(const cell of network.cells) {
    const x=p.x(cell.x),y=p.y(cell.y);
    drawTower(ctx,cell,x,y,app.inspector==="Towers"&&cell.id===app.selectedCell);
    if(x>=-20&&x<=width+20&&y>=-20&&y<=height+20) {
      obstacles.push( {
        left:x-26,right:x+34,top:y-(app.mapCompact?25:31),bottom:y+(app.mapCompact?24:28)
      });
      if(app.layers.labels&&app.view.zoom>=1) labels.push({cell,x,y:y+(app.mapCompact?43:60),anchorX:x,anchorY:y});
    }
  }
  for(const ue of network.ues) {
    const point=app.renderPoints.get(ue.id);
    obstacles.push({left:point.x-23,right:point.x+30,top:point.y-32,bottom:point.y+24,kind:"ue",selected:ue.id===selectedId});
  }
  // Keep tower ledgers stable and behind moving devices. Background traffic
  // does not displace them; the selected device retains visual priority.
  drawMapLabels(ctx,labels.filter(label=>label.cell),obstacles,width,height);
  const devices=[...displayedUEs].sort((a,b)=>(a.id===selectedId)-(b.id===selectedId));
  for(const ue of devices) {
    const point=app.renderPoints.get(ue.id),isSelected=ue.id===selectedId;
    ctx.save();
    ctx.globalAlpha=isSelected?1:.85;
    ctx.fillStyle="#ffffffe8";
    ctx.strokeStyle=isSelected?signalState(ue).color:"#b8cce3";
    ctx.lineWidth=isSelected?2:1;
    const iconScale=isSelected?1:clamp(app.view.zoom,.65,1);
    const radius=isSelected?25:21*iconScale;
    const crossing=[...app.visibleHandovers].reverse().find(event=>event.ue===ue.id);
    if(crossing) {
      ctx.strokeStyle=crossing.success?COLORS.strong:COLORS.weak;
      ctx.lineWidth=2.5;
      ctx.globalAlpha=.65;
      ctx.beginPath();ctx.arc(point.x,point.y,radius+6,0,Math.PI*2);ctx.stroke();
      ctx.globalAlpha=isSelected?1:.85;ctx.strokeStyle=signalState(ue).color;ctx.lineWidth=isSelected?2:1;
    }
    ctx.shadowColor="#234a7420";ctx.shadowBlur=6;ctx.shadowOffsetY=2;
    ctx.beginPath();
    ctx.arc(point.x,point.y,radius,0,Math.PI*2);
    ctx.fill();
    ctx.stroke();
    ctx.shadowColor="transparent";
    const icon=icons[ue.entity_type];
    if(icon?.complete&&icon.naturalWidth) {
      const size=isSelected?43:36*iconScale;
      ctx.drawImage(icon,point.x-size/2,point.y-size/2,size,size);
      ctx.filter="none";
    }
    const signal=signalState(ue);
    ctx.fillStyle=signal.color;
    for(let bar=0;isSelected&&bar<3;bar++) {
      const active=bar===0 || bar===1&&signal.quality>=.4 || bar===2&&signal.quality>=.7;
      ctx.globalAlpha=active?1:.18;ctx.fillRect(point.x+17+bar*4,point.y-25-(bar*3),3,5+bar*3);
    }
    ctx.restore();
    if(isSelected)labels.unshift( {
      text:"UE"+ue.id+" · C"+ue.cell+" · "+ue.allocated+"/"+ue.demand+" RB",x:point.x,y:point.y-43,dark:true,device:true
    });
    if(crossing) {
      ctx.font="500 10px "+FONT_FAMILY;ctx.textAlign="center";
      ctx.fillStyle=crossing.success?COLORS.strong:COLORS.weak;
      ctx.beginPath();ctx.arc(point.x-21,point.y-21,8,0,Math.PI*2);ctx.fill();
      ctx.fillStyle="#fff";ctx.textBaseline="middle";
      ctx.fillText(crossing.success?"✓":"×",point.x-21,point.y-21);
    }
  }
  drawMapLabels(ctx,labels.filter(label=>!label.cell).sort((a,b)=>Number(Boolean(b.device))-Number(Boolean(a.device))),[...obstacles,...app.labelHits],width,height,true);
  if(app.hover) {
    const item=app.hover,text=item.type==="ue"?(item.data.id===selectedId?"UE"+item.data.id+" · "+item.data.slice+" · C"+item.data.cell:"UE"+item.data.id+" · click to inspect"):"C"+item.data.id+" · "+item.data.allocated+" used · "+item.data.reserved+" held · "+item.data.free+" free";
    ctx.font=MAP_FONT;
    const w=ctx.measureText(text).width+16;
    mapLabel(ctx,text,clamp(item.x,w/2+8,width-w/2-8),clamp(item.y-42,20,height-20));
  }
}
function capacityBar(host,allocated,held,capacity,slices) {
  host.replaceChildren();
  for(const item of slices||[ {
    name:"total",allocated
  }]) {
    if(!item.allocated)continue;
    const part=document.createElement("i");
    part.className="allocated slice-"+item.name.toLowerCase();
    part.style.width=(100*item.allocated/capacity)+"%";
    part.title=item.name+": "+item.allocated+" allocated RB";
    host.append(part);
  }
  if(held) {
    const part=document.createElement("i");
    part.className="held";
    part.style.width=(100*held/capacity)+"%";
    part.title=held+" held RB";
    host.append(part);
  }
  host.setAttribute("role","img");
  host.setAttribute("aria-label",allocated+" allocated, "+held+" held, "+(capacity-allocated-held)+" free of "+capacity+" RB");
}
function renderTower() {
  if(!app.snapshot)return;
  setText("towerHeading","Tower C"+app.selectedCell);
  for(const name of ["normal","proposed"]) {
    const cell=app.snapshot[name].cells.find(c=>c.id===app.selectedCell),host=$(name+"Tower");
    host.replaceChildren();
    if(!cell)continue;
    const root=document.createElement("div");
    root.className="ledger";
    const head=document.createElement("div");
    head.className="ledger-head";
    const title=document.createElement("span");
    title.textContent=name==="normal"?"Nearest / no reservation":policyLabel()+" reservation";
    const sum=document.createElement("span");
    sum.className="ledger-sum";
    sum.textContent=cell.allocated+" used · "+cell.reserved+" held · "+cell.free+" free / "+cell.capacity;
    head.append(title,sum);
    root.append(head);
    const bar=document.createElement("div");
    bar.className="capacity-bar";
    capacityBar(bar,cell.allocated,cell.reserved,cell.capacity,cell.slices);
    root.append(bar);
    const grid=document.createElement("div");
    grid.className="ledger-slice-grid";
    for(const s of cell.slices) {
      const item=document.createElement("div");
      item.className="ledger-slice-item";
      item.style.borderLeft="3px solid "+SLICE_COLORS[s.name];
      const hdr=document.createElement("div");
      hdr.className="slice-item-header";
      const label=document.createElement("span");
      label.className="slice-item-name";
      label.style.color=SLICE_COLORS[s.name];
      label.textContent=s.name;
      const count=document.createElement("span");
      count.textContent=s.ues+" devices";
      hdr.append(label,count);
      const rbs=document.createElement("div");
      rbs.className="slice-item-rbs";
      rbs.textContent=s.allocated+"/"+s.demand+" RB";
      if(s.reserved) {
        const tag=document.createElement("span");
        tag.className="held-tag";
        tag.textContent="+"+s.reserved+" held";
        rbs.append(tag);
      }
      const sla=document.createElement("div");
      sla.className="slice-item-sla";
      sla.textContent="Minimum "+s.sla_rbs_per_ue+" RB/device · ≤"+s.latency_max_ms+" ms";
      item.append(hdr,rbs,sla);
      grid.append(item);
    }
    root.append(grid);
    host.append(root);
  }
  for(const button of $("towerNav").querySelectorAll("button")) {
    const selected=Number(button.dataset.cell)===app.selectedCell;
    button.classList.toggle("active",selected);
    button.setAttribute("aria-pressed",String(selected));
  }
}
function renderTowerNav() {
  const nav=$("towerNav"),cells=app.snapshot.proposed.cells;
  if(nav.children.length===cells.length)return;
  nav.replaceChildren();
  for(const cell of cells) {
    const button=document.createElement("button");
    button.type="button";
    button.dataset.cell=String(cell.id);
    button.textContent="C"+cell.id;
    button.setAttribute("aria-label","Inspect tower C"+cell.id);
    button.addEventListener("click",()=>selectTower(cell.id));
    nav.append(button);
  }
}
function selectTower(id) {
  app.selectedCell=id;
  renderTower();
  app.dirty=true;
  document.querySelector(".tower-card").open=true;
  setInspector("Towers",true);
}
function selectUE(id) {
  app.selectedUEId=id;
  app.hover=null;
  app.dirty=true;
  $("deviceSelect").value=String(id);
  renderTeaching();
  renderEvents();
  renderConnectionRoster();
  renderHandoverFeed();
  setInspector("Device",true);
}
function renderConnectionRoster() {
  if(!app.snapshot)return;
  const host=$("connectionRoster"),ues=app.snapshot.proposed.ues;
  if(host.children.length!==ues.length) {
    host.replaceChildren();
    for(const ue of ues) {
      const button=document.createElement("button"),dot=document.createElement("i"),text=document.createElement("span");
      button.type="button";button.dataset.ue=String(ue.id);dot.setAttribute("aria-hidden","true");
      button.append(dot,text);button.addEventListener("click",()=>selectUE(Number(button.dataset.ue)));host.append(button);
    }
  }
  for(const [index,ue] of ues.entries()) {
    const button=host.children[index],signal=signalState(ue);
    button.lastElementChild.textContent="UE"+ue.id+" → C"+ue.cell;
    button.style.setProperty("--signal-color",signal.color);
    button.dataset.allocated=String(ue.allocated);
    button.setAttribute("aria-pressed",String(ue.id===(app.selectedUEId??app.snapshot.focus_id)));
    button.title="UE"+ue.id+" assigned to C"+ue.cell+" · "+ue.allocated+"/"+ue.demand+" RB allocated · "+signal.text+" distance-proxy signal";
  }
}
function setInspector(panel,reveal=false) {
  app.inspector=panel;
  for(const button of document.querySelectorAll("[data-panel]")) {
    const active=button.dataset.panel===panel;
    button.setAttribute("aria-selected",String(active));
    button.tabIndex=active?0:-1;
    $("panel"+button.dataset.panel).hidden=!active;
  }
  if(reveal) {
    $("simulation").classList.remove("inspector-hidden");
    $("showInspector").setAttribute("aria-pressed","true");
  }
  app.dirty=true;
}
function setExpanded(expanded) {
  const workspace=$("simulation");
  if(expanded) {
    app.inspectorBeforeExpand=!workspace.classList.contains("inspector-hidden");
    workspace.classList.add("inspector-hidden");
    $("showInspector").setAttribute("aria-pressed","false");
    app.scrollBeforeExpand={x:scrollX,y:scrollY};
    app.expandedInert=[...document.querySelectorAll(".topbar,main > :not(#simulation),footer,.skip-link")].map(el=>({el,inert:el.inert}));
    app.expandedInert.forEach(({el})=>{el.inert=true;});
    workspace.setAttribute("role","dialog");workspace.setAttribute("aria-modal","true");
  }else {
    workspace.classList.toggle("inspector-hidden",!app.inspectorBeforeExpand);
    $("showInspector").setAttribute("aria-pressed",String(app.inspectorBeforeExpand));
    app.expandedInert?.forEach(({el,inert})=>{el.inert=inert;});
    workspace.removeAttribute("role");workspace.removeAttribute("aria-modal");
  }
  app.expanded=expanded;
  workspace.classList.toggle("map-expanded",expanded);
  document.body.classList.toggle("map-expanded",expanded);
  $("expandMap").setAttribute("aria-pressed",String(expanded));
  $("expandMap").setAttribute("aria-label",expanded?"Exit expanded map":"Expand map");
  $("expandMap").title=expanded?"Exit expanded map (Escape)":"Expand map";
  app.hover=null;app.dirty=true;
  if(!expanded && app.scrollBeforeExpand)scrollTo({...app.scrollBeforeExpand,behavior:"instant"});
  if(!expanded)app.sizeWorkspace?.();
}
function setupWorkspace() {
  // Fit the available viewport even when scenario copy or controls wrap.
  app.sizeWorkspace=()=> {
    if(app.expanded)return;
    const stage=$("simulation"),top=stage.getBoundingClientRect().top+scrollY;
    stage.style.setProperty("--stage-height",Math.max(440,innerHeight-top-16)+"px");
  };
  new ResizeObserver(()=>requestAnimationFrame(app.sizeWorkspace)).observe(document.querySelector(".intro"));
  window.addEventListener("resize",app.sizeWorkspace);
  app.sizeWorkspace();
  const tabs=[...document.querySelectorAll("[data-panel]")];
  for(const button of tabs) {
    button.addEventListener("click",()=>setInspector(button.dataset.panel));
    button.addEventListener("keydown",event=> {
      const index=tabs.indexOf(button);
      const next=event.key==="ArrowRight"?(index+1)%tabs.length:event.key==="ArrowLeft"?(index+tabs.length-1)%tabs.length:event.key==="Home"?0:event.key==="End"?tabs.length-1:null;
      if(next===null)return;
      event.preventDefault();setInspector(tabs[next].dataset.panel);tabs[next].focus();
    });
  }
  $("showInspector").addEventListener("click",()=> {
    const hidden=$("simulation").classList.toggle("inspector-hidden");
    $("showInspector").setAttribute("aria-pressed",String(!hidden));
    app.dirty=true;
  });
  $("expandMap").addEventListener("click",()=>setExpanded(!app.expanded));
  document.addEventListener("keydown",event=> {
    if(event.key==="Escape" && app.expanded) { setExpanded(false);$("expandMap").focus(); }
    if(event.key==="Tab" && app.expanded) {
      const controls=[...$("simulation").querySelectorAll("button:not(:disabled),select:not(:disabled),summary,a[href],[tabindex]:not([tabindex='-1'])")].filter(el=>el.tabIndex>=0&&el.getClientRects().length>0);
      const index=controls.indexOf(document.activeElement);
      if(event.shiftKey && index<=0) {event.preventDefault();controls.at(-1)?.focus();}
      else if(!event.shiftKey && (index===controls.length-1||index<0)) {event.preventDefault();controls[0]?.focus();}
    }
  });
}
function renderDeviceSelect() {
  const host=$("deviceSelect"),selected=app.selectedUEId??app.snapshot.focus_id;
  if(host.options.length!==app.snapshot.proposed.ues.length) {
    host.replaceChildren();
    for(const ue of app.snapshot.proposed.ues) {
      const option=document.createElement("option");
      option.value=String(ue.id);
      option.textContent="UE"+ue.id+" · "+ue.slice;
      host.append(option);
    }
  }
  host.value=String(selected);
}
function pct(value) {
  return value==null||!Number.isFinite(value)?"—":(value*100).toFixed(1)+"%";
}
function delta(a,b) {
  return a==null||b==null?"—":((b-a)>=0?"+":"")+((b-a)*100).toFixed(1)+" pp";
}
function renderTeaching() {
  const snapshot=app.snapshot;
  if(!snapshot)return;
  const network=snapshot.proposed,trackedId=app.selectedUEId??snapshot.focus_id;
  const ue=network.ues.find(u=>u.id===trackedId),contract=network.contracts.find(c=>c.ue===trackedId),outcome=app.outcomes.get(trackedId);
  let number="01",label="APPROACH",title="Following UE"+trackedId+".";
  let explanation=snapshot.handover_mode==="qos_aware"?"Target selection balances signal, capacity and modeled latency. Hysteresis and a two-sample trigger reduce rapid switching.":"Nearest-site geometry selects its cell. The controller can reserve capacity ahead of a predicted crossing.";
  if(contract?.active) {
    number="03";
    label="CAPACITY HELD";
    title=contract.amount+" RBs held at C"+contract.target+".";
    explanation="The dashed path marks protected capacity. A hold can still be wasted if the forecast is wrong or admission fails.";
  }else if(contract) {
    number="02";
    label="FORECAST & REQUEST";
    title="A request for "+contract.amount+" RBs.";
    explanation="Activation at "+(contract.activate_at*snapshot.dt_s).toFixed(1)+" s. Pending capacity is not held; the target will recheck availability.";
  }else if(outcome&&network.step-outcome.step<5) {
    number="04";
    label="HANDOVER RESULT";
    title="UE"+trackedId+" crossed C"+outcome.proposed.source+" → C"+outcome.proposed.target+".";
    explanation=outcome.proposed.success?(outcome.proposed.qos?"The controller admitted the device and met its handover QoS check.":"The controller admitted the device; full-demand QoS was not met."):"Admission failed. The crossing below shows available capacity and modeled latency.";
  }
  setText("phaseNumber",number);
  setText("phaseLabel",label);
  setText("phaseTitle",title);
  setText("phaseText",explanation);
  $("phaseLabel").closest(".step-card").dataset.phase=label==="HANDOVER RESULT"?(outcome.proposed.success?"admitted":"blocked"):contract?"reservation":"approach";
  setText("deviceSummary",ue.slice+" · C"+ue.cell+" · "+ue.allocated+"/"+ue.demand+" RB allocated");
  const signal=signalState(ue);
  setText("signalStatus",signal.text+" signal");
  $("signalStatus").dataset.level=signal.level;
  $("signalStatus").title="Distance proxy: 100 × clamp(1 − distance to serving site / "+(snapshot.signal_radius||48)+", 0, 1). Strong ≥ 70%; fair ≥ 40%; weak < 40%. "+(snapshot.handover_mode==="qos_aware"?"QoS selection also considers capacity and latency, with a "+Math.round(100*(snapshot.minimum_signal??.35))+"% eligibility floor.":"Nearest-site selection determines the cell.")+" Not measured RSRP or SINR.";
  const facts=$("selectedFacts");
  facts.replaceChildren();
  for(const [key,value] of [["Device","UE"+ue.id+" · "+ue.slice],["Serving cell","C"+ue.cell],["Current allocation",ue.allocated+" / "+ue.demand+" RB"],["Reservation",contract?contract.amount+" RB "+(contract.active?"held":"pending"):"None"]]) {
    const row=document.createElement("div"),name=document.createElement("span"),val=document.createElement("b");
    name.textContent=key;
    val.textContent=value;
    row.append(name,val);
    facts.append(row);
  }
  setText("outcomeHeading",outcome?"UE"+trackedId+" · latest crossing":"UE"+trackedId+" · no crossing yet");
  document.querySelector(".outcome-card>.eyebrow").textContent=snapshot.handover_mode==="qos_aware"?"SAME MOVEMENT · DIFFERENT CELL DECISIONS":"SAME MOVEMENT · DIFFERENT RESERVATIONS";
  for(const name of ["normal","proposed"]) {
    const result=outcome?.[name],el=$(name+"Outcome");
    const unpaired=name==="normal"&&snapshot.handover_mode==="qos_aware"&&outcome&&!result;
    el.textContent=result?(result.success?"Admitted":"Blocked"):unpaired?"No same-time crossing":"No crossing yet";
    el.parentElement.classList.toggle("success",Boolean(result?.success));
    el.parentElement.classList.toggle("fail",Boolean(result&&!result.success));
    setText(name+"OutcomeDetail",result?(result.success?result.allocated_rbs+" / "+result.demand_rbs+" RB granted":result.available_rbs+" / "+result.demand_rbs+" RB available")+" · "+result.latency_ms.toFixed(2)+" ms · QoS "+(result.qos?"met":"unmet"):unpaired?"Baseline currently uses C"+snapshot.normal.ues.find(u=>u.id===trackedId).cell+". The two target policies can switch at different times.":"Waiting for this device to change cells.");
  }
  setText("outcomeNote",outcome?"Crossing at "+outcome.time.toFixed(1)+" s · C"+outcome.proposed.source+" → C"+outcome.proposed.target+". Current allocation above may change in later intervals.":"Admission and full-demand QoS are separate checks.");
  const decision=network.decision,decisionFacts=$("decisionFacts");
  decisionFacts.replaceChildren();
  for(const text of [snapshot.controller_kind==="linear_ppo"?"Saved linear PPO":"Trained MLP · 35–64–32–24",snapshot.controller_kind==="linear_ppo"?"Masked action probabilities":"Fixed reservation rule · not neural action probabilities","One candidate per interval",decision?.ue!=null?"Decision for UE"+decision.ue:"No forecast candidate"]) {
    const span=document.createElement("span");
    span.textContent=text;
    decisionFacts.append(span);
  }
  if(decision?.ue!=null) {
    setText("decisionHeading",decision.action?"Reserve "+decision.amount_rbs+" RB at C"+decision.target:"No reservation for UE"+decision.ue);
    setText("decisionText","Last interval: UE"+decision.ue+" → C"+decision.target+"; ETA "+decision.eta_s.toFixed(1)+" s; heuristic forecast score "+pct(decision.probability)+". "+(decision.action?"Activation delay "+decision.delay_s.toFixed(1)+" s.":"The policy selected no hold."));
  }else {
    setText("decisionHeading","No eligible forecast this interval");
    setText("decisionText","The policy takes no action. Devices continue moving and generating traffic.");
  }
  const distribution=$("policyDistribution");
  distribution.replaceChildren();
  for(const action of decision?.distribution||[]) {
    const row=document.createElement("div");
    row.className="policy-row"+(action.action===decision.action?" chosen":"");
    const label=document.createElement("span");
    label.textContent=action.label+(action.action===decision.action?" · chosen":"");
    const val=document.createElement("b");
    val.textContent=pct(action.probability);
    const bar=document.createElement("i"),fill=document.createElement("em");
    fill.style.width=(action.probability*100)+"%";
    bar.append(fill);
    row.append(label,val,bar);
    distribution.append(row);
  }
}
function renderMetrics() {
  const {
    normal,proposed
  }=app.snapshot,n=normal.metrics,p=proposed.metrics;
  setText("statHsrNormal",pct(n.handover_success_rate));
  setText("statHsrProposed",pct(p.handover_success_rate));
  setText("statHsrDelta",delta(n.handover_success_rate,p.handover_success_rate));
  setText("statQosNormal",pct(n.qos_rate));
  setText("statQosProposed",pct(p.qos_rate));
  setText("statQosDelta",delta(n.qos_rate,p.qos_rate));
  setText("normalScore",n.successes+" / "+n.handovers+" baseline");
  setText("proposedScore",p.successes+" / "+p.handovers+" "+policyLabel());
  const totals=proposed.cells.reduce((a,c)=>( {
    allocated:a.allocated+c.allocated,held:a.held+c.reserved,free:a.free+c.free,capacity:a.capacity+c.capacity
  }), {
    allocated:0,held:0,free:0,capacity:0
  });
  setText("statTotalAlloc",totals.allocated);
  setText("statTotalReserved",totals.held);
  setText("statTotalFree",totals.free);
  setText("statCapacityChip",totals.capacity+" RB");
  setText("statCapacityDesc",totals.allocated+" allocated + "+totals.held+" held + "+totals.free+" free = "+totals.capacity+" RB");
  setText("reservationCost","Held capacity over time: "+pct(p.reservation_occupancy)+" · settled unused holds: "+pct(p.settled_unused_reservation_fraction));
  const slices=Object.keys(SLICE_COLORS).map(name=>({name,allocated:proposed.cells.reduce((sum,c)=>sum+c.slices.find(s=>s.name===name).allocated,0)}));
  capacityBar($("networkCapacityBar"),totals.allocated,totals.held,totals.capacity,slices);
  updateSparklines(app.snapshot.history);
}
function buildSparklineData(history,key,width=200,height=44,pad=4) {
  const points=history.map((item,i)=>( {
    x:history.length>1?i/(history.length-1)*width:width,y:item[key]==null?null:height-pad-clamp(item[key],0,1)*(height-2*pad),val:item[key],time:item.time,step:item.t
  }));
  let line="",connected=false;
  for(const point of points) {
    if(point.y==null) {
      connected=false;
      continue;
    }line+=(connected?" L ":" M ")+point.x.toFixed(1)+","+point.y.toFixed(1);
    connected=true;
  }return {
    line,points
  };
}
function updateSparklines(history) {
  app.sparkData= {
  };
  for(const metric of ["hsr","qos"]) {
    app.sparkData[metric]= {
    };
    for(const name of ["normal","proposed"]) {
      const data=buildSparklineData(history[name],metric);
      app.sparkData[metric][name]=data;
      $(metric+"Spark"+(name==="normal"?"Normal":"Proposed")).setAttribute("d",data.line);
    }
  }
}
function setupSparklineScrubbers() {
  for(const metric of ["hsr","qos"]) {
    const svg=$(metric+"Sparkline"),tooltip=$(metric+"ScrubTooltip"),line=$(metric+"ScrubLine");
    svg.addEventListener("pointermove",event=> {
      const data=app.sparkData?.[metric];
      if(!data?.proposed.points.length)return;
      const rect=svg.getBoundingClientRect(),fraction=clamp((event.clientX-rect.left)/rect.width,0,1),index=Math.round(fraction*(data.proposed.points.length-1));
      const point=data.proposed.points[index],baseline=data.normal.points[index];
      line.removeAttribute("hidden");
      line.setAttribute("x1",point.x);
      line.setAttribute("x2",point.x);
      tooltip.textContent="t="+point.time.toFixed(1)+" s · PPO "+pct(point.val)+" · baseline "+pct(baseline?.val);
    });
    svg.addEventListener("pointerleave",()=> {
      line.setAttribute("hidden","");
      tooltip.textContent="Solid PPO · dashed no reservation · cumulative rate";
    });
  }
}
function renderEvents() {
  if(!app.snapshot)return;
  const selected=app.selectedUEId??app.snapshot.focus_id,events=app.events.filter(e=>$("eventFilter").value==="all"||e.ue===selected);
  setText("eventCount",events.length+" events");
  const stream=$("liveEventStream");
  stream.replaceChildren();
  if(!events.length) {
    const row=document.createElement("p");
    row.className="event-row idle";
    row.textContent="No events for this selection yet.";
    stream.append(row);
    return;
  }
  for(const item of [...events].reverse()) {
    const row=document.createElement("div");
    row.className="event-row";
    const label=document.createElement("strong");
    label.textContent=item.time.toFixed(1)+" s · "+item.net;
    const text=document.createElement("small");
    text.textContent=item.text;
    row.append(label,text);
    stream.append(row);
  }
}
function acceptSnapshot(snapshot) {
  app.previous=app.snapshot;
  app.snapshot=snapshot;
  app.frameStart=performance.now();
  app.dirty=true;
  setText("simTime",snapshot.proposed.time.toFixed(1)+" s");
  setText("mapEyebrow","LIVE "+policyLabel().toUpperCase()+" NETWORK · "+snapshot.proposed.cells.length+" CELLS / "+snapshot.proposed.ues.length+" DEVICES");
  setText("activeControllerLabel",snapshot.controller_kind==="linear_ppo"?"Linear PPO":"Deep forecast");
  document.querySelector('.inspector-badge').textContent=policyLabel();
  for(const el of document.querySelectorAll('.outcome-tag,.stat-compare small,.stat-metric-row small'))if(/PPO|NEURAL/.test(el.textContent))el.textContent=policyLabel().toUpperCase()+" RESERVATION";
  const freshHandovers={};
  for(const name of ["normal","proposed"])freshHandovers[name]=(snapshot[name].recent_handovers||snapshot[name].handover_details||[]).filter(event=>{
    const key=name+":"+(event.step??snapshot[name].step)+":"+event.ue;
    if(app.handoverKeys.has(key))return false;
    app.handoverKeys.add(key);return true;
  });
  app.handoverKeys=new Set(["normal","proposed"].flatMap(name=>(snapshot[name].recent_handovers||snapshot[name].handover_details||[]).map(event=>name+":"+(event.step??snapshot[name].step)+":"+event.ue)));
  for(const result of freshHandovers.proposed) {
    const baseline=(snapshot.normal.recent_handovers||snapshot.normal.handover_details).find(r=>r.ue===result.ue&&(r.step??snapshot.normal.step)===(result.step??snapshot.proposed.step));
    app.outcomes.set(result.ue, {
      normal:baseline,proposed:result,step:result.step??snapshot.proposed.step,time:result.time??snapshot.proposed.time
    });
  }
  for(const name of ["normal","proposed"]) {
    const net=name==="normal"?"No reservation":policyLabel(),time=snapshot[name].time;
    for(const event of snapshot[name].contract_events||[]) {
      const details=event.kind==="settled"?" · "+event.used_rbs+" RB used":"";
      const text="UE"+event.ue+" · R"+event.reservation+" "+event.kind+" · "+event.amount_rbs+" RB at C"+event.target+details;
      app.events.push( {
        text,net,time,ue:event.ue
      });
    }
    for(const event of freshHandovers[name]) {
      const text="UE"+event.ue+" · C"+event.source+" → C"+event.target+" · "+(event.success?"admitted":"blocked")+" · "+event.available_rbs+"/"+event.demand_rbs+" RB available · QoS "+(event.qos?"met":"unmet");
      app.events.push( {
        text,net,time:event.time??time,ue:event.ue
      });
    }
  }
  app.events=app.events.slice(-100);
  renderDeviceSelect();
  renderConnectionRoster();
  renderTowerNav();
  renderTower();
  renderTeaching();
  renderMetrics();
  renderEvents();
  renderHandoverFeed();
}
function renderHandoverFeed() {
  const network=app.snapshot?.proposed;
  if(!network)return;
  setText("handoverCounter",network.metrics.handovers+" handovers");
  const selectedId=app.selectedUEId??app.snapshot.focus_id;
  const events=(network.recent_handovers||[]).filter(event=>event.ue===selectedId&&network.step-event.step<8).slice(-1).reverse(),host=$("recentHandoverList");
  const key=selectedId+"|"+events.map(event=>event.step+":"+event.ue+":"+event.success).join("|");
  if(host.dataset.key===key)return;
  host.dataset.key=key;host.replaceChildren();
  if(!events.length) {
    const note=document.createElement("span");note.className="handover-idle";
    note.textContent="UE"+selectedId+": no recent handover · Other devices show their tower link";host.append(note);return;
  }
  for(const event of events) {
    const button=document.createElement("button");button.type="button";
    button.className="handover-chip "+(event.success?"admitted":"blocked");
    button.textContent="UE"+event.ue+" · C"+event.source+" → C"+event.target+" · "+(event.success?"admitted":"blocked");
    button.title="Crossing at "+event.time.toFixed(1)+" s · select this device";
    button.addEventListener("click",()=>selectUE(event.ue));host.append(button);
  }
}
async function advance(generation) {
  if(!app.playing||app.busy||!app.token||generation!==app.generation)return;
  app.busy=true;
  try {
    const snapshot=await requestJSON("/api/step", {
      session:app.token,count:1
    });
    if(generation===app.generation) {
      if(app.playing)acceptSnapshot(snapshot);
      else app.pendingSnapshot=snapshot;
    }
  }catch(error) {
    if(generation===app.generation)showError(error);
  }
  finally {
    if(generation===app.generation) {
      app.busy=false;
      if(app.playing)app.timer=setTimeout(()=>advance(generation),STEP_MS);
    }
  }
}
async function startScenario() {
  const generation=++app.generation;
  clearTimeout(app.timer);
  Object.assign(app, {
    token:null,snapshot:null,previous:null,pendingSnapshot:null,selectedUEId:null,selectedCell:0,playing:true,busy:false,pausedProgress:0,pauseWall:null,events:[],autoTrack:false,hover:null
  });
  app.outcomes.clear();
  app.handoverKeys.clear();
  $("recentHandoverList").removeAttribute("data-key");
  fitMap();
  setPlaybackButton();
  $("errorNotice").hidden=true;
  const scenario=$("scenario").value,preset=app.scenarios?.[scenario];
  setText("scenarioDescription",preset?.description||"");
  try {
    const response=await requestJSON("/api/session", {
      scenario,seed:preset?.seed??21,controller:$("controller").value
    });
    if(generation!==app.generation)return;
    app.token=response.session;
    acceptSnapshot(response.snapshot);
    setPlaybackButton();
    app.timer=setTimeout(()=>advance(generation),STEP_MS);
  }catch(error) {
    if(generation===app.generation)showError(error);
  }
}
function togglePlayback() {
  if(!app.token)return;
  app.playing=!app.playing;
  if(app.playing) {
    if(app.pendingSnapshot) {
      acceptSnapshot(app.pendingSnapshot);
      app.pendingSnapshot=null;
      app.pausedProgress=0;
    }
    app.frameStart=performance.now()-app.pausedProgress*STEP_MS;
    app.pauseWall=null;
    app.timer=setTimeout(()=>advance(app.generation),Math.max(0,STEP_MS*(1-app.pausedProgress)));
  }
  else {
    clearTimeout(app.timer);
    app.pauseWall=performance.now();
    app.pausedProgress=clamp((app.pauseWall-app.frameStart)/STEP_MS,0,1);
  }
  app.dirty=true;
  setPlaybackButton();
}
function setupMapInteractions() {
  const canvas=$("networkCanvas");
  new ResizeObserver(()=> {
    canvasSize();
    app.dirty=true;
  }).observe(canvas);
  $("zoomIn").addEventListener("click",()=>zoomTo(app.view.zoom+.25));
  $("zoomOut").addEventListener("click",()=>zoomTo(app.view.zoom-.25));
  $("fitMap").addEventListener("click",fitMap);
  $("trackCar").addEventListener("click",()=> {
    app.autoTrack=!app.autoTrack;
    clampView();
  });
  canvas.addEventListener("keydown",event=> {
    if(event.altKey||event.ctrlKey||event.metaKey)return;
    const rect=canvas.getBoundingClientRect(),p=projection(rect.width,rect.height),distance=24*(app.snapshot?.world_size||100)/p.side;
    if(["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(event.key)) {
      event.preventDefault();
      app.autoTrack=false;
      app.view.cx+=(event.key==="ArrowRight"?distance:event.key==="ArrowLeft"?-distance:0);
      app.view.cy+=(event.key==="ArrowDown"?distance:event.key==="ArrowUp"?-distance:0);
      clampView();
    }else if(event.key==="+"||event.key==="=") {
      event.preventDefault();
      zoomTo(app.view.zoom+.25);
    }
    else if(event.key==="-") {
      event.preventDefault();
      zoomTo(app.view.zoom-.25);
    }
    else if(event.key.toLowerCase()==="f") {
      event.preventDefault();
      fitMap();
    }
    else if(event.key===" ") {
      event.preventDefault();
      togglePlayback();
    }
  });
  canvas.addEventListener("wheel",event=> {
    event.preventDefault();
    const r=canvas.getBoundingClientRect();
    zoomTo(app.view.zoom*Math.exp(-clamp(event.deltaY,-240,240)*.002), {
      x:event.clientX-r.left,y:event.clientY-r.top
    });
  }, {
    passive:false
  });
  canvas.addEventListener("dblclick",event=> {
    event.preventDefault();const rect=canvas.getBoundingClientRect();
    zoomTo(app.view.zoom*(event.shiftKey?1/1.5:1.5),{x:event.clientX-rect.left,y:event.clientY-rect.top});
  });
  canvas.addEventListener("pointerdown",event=> {
    if(event.pointerType==="mouse"&&event.button!==0)return;
    app.autoTrack=false;
    app.pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
    if(app.pointers.size>=2) {
      if(app.pointers.size>2) {canvas.setPointerCapture(event.pointerId);return;}
      const [a,b]=[...app.pointers.values()],rect=canvas.getBoundingClientRect(),p=projection(rect.width,rect.height);
      const mx=(a.x+b.x)/2-rect.left,my=(a.y+b.y)/2-rect.top;
      app.pinch={distance:Math.max(1,Math.hypot(a.x-b.x,a.y-b.y)),zoom:app.view.zoom,worldX:p.worldX(mx),worldY:p.worldY(my)};
      app.drag=null;app.hover=null;
      canvas.setPointerCapture(event.pointerId);canvas.classList.add("dragging");
      return;
    }
    app.drag= {
      x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false,id:event.pointerId
    };
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add("dragging");
    clampView();
  });
  canvas.addEventListener("pointermove",event=> {
    const rect=canvas.getBoundingClientRect(),p=projection(rect.width,rect.height),x=event.clientX-rect.left,y=event.clientY-rect.top;
    if(app.pointers.has(event.pointerId))app.pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
    if(app.pinch && app.pointers.size>=2) {
      const [a,b]=[...app.pointers.values()],world=app.snapshot?.world_size||100;
      app.view.zoom=clamp(app.pinch.zoom*Math.hypot(a.x-b.x,a.y-b.y)/app.pinch.distance,MIN_ZOOM,MAX_ZOOM);
      const next=projection(rect.width,rect.height),mx=(a.x+b.x)/2-rect.left,my=(a.y+b.y)/2-rect.top;
      app.view.cx=app.pinch.worldX-(mx-next.centerX)*world/next.side;
      app.view.cy=app.pinch.worldY-(my-next.centerY)*world/next.side;
      clampView();return;
    }
    if(app.drag) {
      const dx=event.clientX-app.drag.x,dy=event.clientY-app.drag.y;
      if(Math.hypot(event.clientX-app.drag.startX,event.clientY-app.drag.startY)>5)app.drag.moved=true;
      if(app.drag.moved) {
        app.view.cx-=dx*(app.snapshot?.world_size||100)/p.side;
        app.view.cy-=dy*(app.snapshot?.world_size||100)/p.side;
        clampView();
      }
      app.drag.x=event.clientX;
      app.drag.y=event.clientY;
      app.hover=null;
      return;
    }
    app.hover=null;
    if(app.snapshot) {
      for(const ue of app.snapshot.proposed.ues) {
        const pos=app.renderPoints.get(ue.id);
        if(pos&&Math.hypot(x-pos.x,y-pos.y)<26) {
          app.hover= {
            type:"ue",data:ue,x,y
          };
          break;
        }
      }
      if(!app.hover)for(const cell of app.snapshot.proposed.cells) {
        const label=app.labelHits.find(l=>l.id===cell.id&&x>=l.left&&x<=l.right&&y>=l.top&&y<=l.bottom);
        if(label||Math.hypot(x-p.x(cell.x),y-p.y(cell.y))<28) {
          app.hover= {
            type:"cell",data:cell,x,y
          };
          break;
        }
      }
    }
    app.dirty=true;
  });
  canvas.addEventListener("pointerup",event=> {
    const wasPinching=Boolean(app.pinch);
    app.pointers.delete(event.pointerId);
    if(wasPinching) {
      app.pinch=null;
      if(app.pointers.size===1) {
        const [id,point]=[...app.pointers.entries()][0];
        app.drag={x:point.x,y:point.y,startX:point.x,startY:point.y,moved:true,id};
      }else { app.drag=null;canvas.classList.remove("dragging"); }
      return;
    }
    const drag=app.drag;
    app.drag=null;
    canvas.classList.remove("dragging");
    if(!drag||drag.moved||!app.snapshot)return;
    const rect=canvas.getBoundingClientRect(),p=projection(rect.width,rect.height),x=event.clientX-rect.left,y=event.clientY-rect.top;
    const ue=app.snapshot.proposed.ues.map(u=>( {
      ue:u,point:app.renderPoints.get(u.id)
    })).filter(v=>v.point&&Math.hypot(x-v.point.x,y-v.point.y)<26)
    .sort((a,b)=>Math.hypot(x-a.point.x,y-a.point.y)-Math.hypot(x-b.point.x,y-b.point.y))[0];
    if(ue) {
      selectUE(ue.ue.id);
      return;
    }
    const label=app.labelHits.find(l=>x>=l.left&&x<=l.right&&y>=l.top&&y<=l.bottom);
    const cell=app.snapshot.proposed.cells.find(c=>c.id===label?.id||Math.hypot(x-p.x(c.x),y-p.y(c.y))<28);
    if(cell)selectTower(cell.id);
  });
  canvas.addEventListener("pointerleave",()=> {
    app.hover=null;
    app.dirty=true;
  });
  canvas.addEventListener("pointercancel",event=> {
    app.pointers.delete(event.pointerId);app.pinch=null;
    app.drag=null;
    canvas.classList.remove("dragging");
  });
  for(const button of document.querySelectorAll("[data-line]"))button.addEventListener("click",()=> {
    const key=button.dataset.line;
    app.layers[key]=!app.layers[key];
    app.dirty=true;
    button.classList.toggle("active",app.layers[key]);
    button.setAttribute("aria-pressed",String(app.layers[key]));
  });
}
let lastFrame=0;
function frame(now) {
  if(now-lastFrame>=1000/30&&!document.hidden&&(app.dirty||app.playing)) {
    drawMap(now);
    app.dirty=false;
    lastFrame=now;
  }requestAnimationFrame(frame);
}
async function init() {
  setupWorkspace();
  setupMapInteractions();
  setupSparklineScrubbers();
  $("playBtn").addEventListener("click",togglePlayback);
  $("scenario").addEventListener("change",startScenario);
  $("restartBtn").addEventListener("click",startScenario);
  $("deviceSelect").addEventListener("change",()=>selectUE(Number($("deviceSelect").value)));
  $("controller").addEventListener("change",startScenario);
  $("eventFilter").addEventListener("change",renderEvents);
  requestAnimationFrame(frame);
  try {
    const meta=await requestJSON("/api/meta");
    app.scenarios=meta.scenarios;
    const baseline=meta.published?.metrics?.find(m=>m.method==="No reservation"),ppo=meta.published?.metrics?.find(m=>m.method?.includes("PPO"));
    setText("benchmarkStat",baseline&&ppo?pct(ppo.handover_success_rate)+" vs "+pct(baseline.handover_success_rate):"Unavailable");
    for(const profile of meta.published?.simulation?.slices||[]) {
      const card=document.createElement("article"),title=document.createElement("h3"),text=document.createElement("p");
      title.textContent=profile.name;
      text.textContent="Minimum "+profile.sla_rbs+" RB per device · latency ≤ "+profile.latency_max_ms+" ms · priority "+profile.priority;
      card.append(title,text);
      $("sliceProfiles").append(card);
    }
    await startScenario();
  }catch(error) {
    showError(error);
  }
}
init();
