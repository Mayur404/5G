"use strict";

const $ = id => document.getElementById(id);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const setText = (id, value) => { $(id).textContent = value; };
const STEP_MS = 620;
const towerImage = new Image();
towerImage.src = "/static/icons/tower.svg";
const icons = Object.fromEntries(["car", "drone", "robot"].map(type => {
  const icon = new Image(); icon.src = `/static/icons/${type}.svg`; return [type, icon];
}));
const app = {
  token: null, snapshot: null, previous: null, scenarios: null,
  playing: true, busy: false, generation: 0, timer: null,
  frameStart: performance.now(), pausedProgress: 0, pauseWall: null,
  selectedCell: 0, view: {zoom: 1, cx: 50, cy: 50}, drag: null,
  outcome: null, outcomeStep: null, pulse: null,
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
  $("playBtn").setAttribute("aria-pressed", String(!app.playing));
}

function showError(error) {
  app.playing = false; clearTimeout(app.timer); setPlaybackButton();
  setText("phaseLabel", "SIMULATION ERROR");
  setText("phaseTitle", "The local simulator stopped.");
  setText("phaseText", String(error));
}

function canvasSize() {
  const canvas = $("networkCanvas"), rect = canvas.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * pixelRatio));
  const height = Math.max(1, Math.round(rect.height * pixelRatio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  return {ctx, width: rect.width, height: rect.height};
}

function projection(width, height) {
  const side = Math.min(width - 54, height - 54) * app.view.zoom;
  const x = value => width / 2 + (value - app.view.cx) * side / 100;
  const y = value => height / 2 + (value - app.view.cy) * side / 100;
  return {side, x, y, worldX: px => app.view.cx + (px - width / 2) * 100 / side,
    worldY: py => app.view.cy + (py - height / 2) * 100 / side};
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
  return {x: p.x(prior.x + (ue.x - prior.x) * alpha),
    y: p.y(prior.y + (ue.y - prior.y) * alpha)};
}

function roundedLabel(ctx, text, x, y, background, foreground = "#fff") {
  ctx.save(); ctx.font = "700 12px Segoe UI,Arial";
  const width = ctx.measureText(text).width + 20;
  ctx.fillStyle = background; ctx.beginPath(); ctx.roundRect(x - width / 2, y - 12, width, 24, 6); ctx.fill();
  ctx.fillStyle = foreground; ctx.textAlign = "center"; ctx.fillText(text, x, y + 4); ctx.restore();
}

function drawMap(now) {
  if (!app.snapshot) return;
  const {ctx, width, height} = canvasSize();
  const p = projection(width, height), network = app.snapshot.proposed;
  const previous = app.previous?.proposed;
  const focusId = app.snapshot.focus_id;
  const focus = network.ues.find(ue => ue.id === focusId);
  const visualNow = app.playing ? now : (app.pauseWall ?? now);
  const alpha = app.playing ? clamp((now - app.frameStart) / STEP_MS, 0, 1) : app.pausedProgress;
  ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#fafbfc"; ctx.fillRect(0, 0, width, height);

  // Four Voronoi service areas for the fixed sites. Only the map viewport is zoomed.
  for (const [cell, shade] of [[0,"#f5f7f8"],[1,"#fafbfc"],[2,"#fafbfc"],[3,"#f5f7f8"]]) {
    const x = cell % 2 ? 50 : 0, y = cell < 2 ? 0 : 50;
    ctx.fillStyle = shade; ctx.fillRect(p.x(x), p.y(y), p.side / 2, p.side / 2);
  }
  ctx.strokeStyle = "#c9d0d8"; ctx.lineWidth = 1.5; ctx.setLineDash([8, 7]);
  ctx.beginPath(); ctx.moveTo(p.x(50), p.y(0)); ctx.lineTo(p.x(50), p.y(100));
  ctx.moveTo(p.x(0), p.y(50)); ctx.lineTo(p.x(100), p.y(50)); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = "#e7ebee"; ctx.lineWidth = 8;
  ctx.beginPath(); ctx.moveTo(p.x(0), p.y(25)); ctx.lineTo(p.x(100), p.y(25));
  ctx.moveTo(p.x(0), p.y(75)); ctx.lineTo(p.x(100), p.y(75)); ctx.stroke();

  // Other users stay visible, but only the teaching UE carries an animated link.
  for (const ue of network.ues) {
    const pos = pointFor(ue, previous, alpha, p), cell = network.cells[ue.cell];
    ctx.save(); ctx.strokeStyle = ue.allocated > 0 ? "#aab3bc" : "#d5b5af";
    ctx.lineWidth = 1.3; ctx.globalAlpha = ue.id === focusId ? 0 : 0.46;
    ctx.setLineDash(ue.allocated > 0 ? [] : [4, 6]);
    ctx.beginPath(); ctx.moveTo(p.x(cell.x), p.y(cell.y)); ctx.lineTo(pos.x, pos.y); ctx.stroke(); ctx.restore();
  }
  if (focus) {
    const pos = pointFor(focus, previous, alpha, p), cell = network.cells[focus.cell];
    const sx = p.x(cell.x), sy = p.y(cell.y), connected = focus.allocated > 0;
    ctx.save(); ctx.strokeStyle = connected ? "#435464" : "#d73b2f"; ctx.lineWidth = 4;
    ctx.setLineDash(connected ? [] : [7, 6]); ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(pos.x, pos.y);
    ctx.stroke(); ctx.restore();
    if (connected) for (let packet = 0; packet < 2; packet++) {
      const fraction = (visualNow * 0.00031 + packet * 0.5) % 1;
      ctx.fillStyle = "#1f2b36"; ctx.beginPath();
      ctx.arc(sx + (pos.x - sx) * fraction, sy + (pos.y - sy) * fraction, 4, 0, Math.PI * 2); ctx.fill();
    }
  }

  const contract = network.contracts.find(item => item.ue === focusId);
  if (focus && contract) {
    const pos = pointFor(focus, previous, alpha, p), target = network.cells[contract.target];
    const tx = p.x(target.x), ty = p.y(target.y), active = contract.active;
    ctx.save(); ctx.strokeStyle = active ? "#e42b0c" : "#a76a23"; ctx.lineWidth = 4;
    ctx.setLineDash(active ? [10, 7] : [5, 7]);
    ctx.beginPath(); ctx.moveTo(pos.x, pos.y); ctx.quadraticCurveTo((pos.x + tx) / 2,
      (pos.y + ty) / 2 - 48, tx, ty); ctx.stroke(); ctx.restore();
    const badgeX=tx < width/2 ? tx+105 : tx-105;
    roundedLabel(ctx, `${contract.amount} RB ${active ? "HELD" : "REQUESTED"}`,
      clamp(badgeX, 82, width - 82), clamp(ty - 84, 28, height - 28), active ? "#e42b0c" : "#94611d");
  }

  for (const cell of network.cells) {
    const x = p.x(cell.x), y = p.y(cell.y), selected = cell.id === app.selectedCell;
    if (selected) {ctx.fillStyle = "rgba(228,43,12,.06)"; ctx.beginPath(); ctx.arc(x,y,70,0,Math.PI*2);ctx.fill();
      ctx.strokeStyle = "#e42b0c";ctx.lineWidth = 2;ctx.beginPath();ctx.arc(x,y,70,0,Math.PI*2);ctx.stroke();}
    if (towerImage.complete && towerImage.naturalWidth) ctx.drawImage(towerImage,x-55,y-68,110,121);
    else {ctx.fillStyle="#343d48";ctx.fillRect(x-12,y-38,24,76);}
    ctx.fillStyle="#fff";ctx.strokeStyle=selected?"#e42b0c":"#d4dce2";ctx.lineWidth=1.5;
    ctx.beginPath();ctx.roundRect(x-31,y+36,62,26,6);ctx.fill();ctx.stroke();
    ctx.fillStyle="#17212c";ctx.textAlign="center";ctx.font="700 13px Segoe UI,Arial";ctx.fillText(`C${cell.id}`,x,y+54);
    const barX=x-31,barY=y+67;ctx.fillStyle="#e0e5e9";ctx.fillRect(barX,barY,62,7);
    ctx.fillStyle="#404b57";ctx.fillRect(barX,barY,62*cell.allocated/cell.capacity,7);
    if(cell.reserved){ctx.fillStyle="#e42b0c";ctx.fillRect(barX+62*cell.allocated/cell.capacity,barY,62*cell.reserved/cell.capacity,7);}
  }

  if (app.pulse && visualNow - app.pulse.at < 3000) {
    const source=network.cells[app.pulse.from],target=network.cells[app.pulse.to];
    const sx=p.x(source.x),sy=p.y(source.y)-10,tx=p.x(target.x),ty=p.y(target.y)-10;
    const midx=(sx+tx)/2,midy=(sy+ty)/2-52;
    ctx.save();ctx.globalAlpha=1-clamp((visualNow-app.pulse.at)/3600,0,.55);
    ctx.strokeStyle="#20845a";ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(sx,sy);
    ctx.quadraticCurveTo(midx,midy-25,tx,ty);ctx.stroke();
    const angle=Math.atan2(ty-(midy-25),tx-midx);ctx.fillStyle="#20845a";ctx.beginPath();ctx.moveTo(tx,ty);
    ctx.lineTo(tx-17*Math.cos(angle-.45),ty-17*Math.sin(angle-.45));
    ctx.lineTo(tx-17*Math.cos(angle+.45),ty-17*Math.sin(angle+.45));ctx.closePath();ctx.fill();ctx.restore();
    roundedLabel(ctx,"PPO HANDOVER ADMITTED",width/2,37,"#20845a");
  }

  const focusPosition=focus?pointFor(focus,previous,alpha,p):null;
  const deviceOrder=[...network.ues].sort((a,b)=>Number(a.id===focusId)-Number(b.id===focusId));
  for (const ue of deviceOrder) {
    const pos=pointFor(ue,previous,alpha,p),prior=previous?.ues.find(item=>item.id===ue.id)||ue;
    const quality=prior.signal_quality+(ue.signal_quality-prior.signal_quality)*alpha;
    const color=quality<.48?"#d73b2f":quality<.68?"#d29a25":"#21805a";
    const focused=ue.id===focusId,size=focused?47:24,radius=focused?34:18;
    ctx.save();ctx.globalAlpha=focused?1:.72;
    ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(pos.x,pos.y,radius,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle=color;ctx.lineWidth=focused?4:2.5;ctx.beginPath();ctx.arc(pos.x,pos.y,radius,0,Math.PI*2);ctx.stroke();
    if(focused){ctx.strokeStyle="#202934";ctx.lineWidth=1.6;ctx.beginPath();ctx.arc(pos.x,pos.y,radius+6,0,Math.PI*2);ctx.stroke();}
    const icon=icons[ue.entity_type];if(icon.complete&&icon.naturalWidth)ctx.drawImage(icon,pos.x-size/2,pos.y-size/2,size,size);
    else{ctx.fillStyle="#27313b";ctx.beginPath();ctx.arc(pos.x,pos.y,9,0,Math.PI*2);ctx.fill();}
    ctx.restore();
    if(focused) roundedLabel(ctx,`UE${ue.id}  ${ue.slice} CAR`,clamp(pos.x,90,width-90),
      clamp(pos.y+54,20,height-20),"#202934");
    else if(!focusPosition||Math.hypot(pos.x-focusPosition.x,pos.y-focusPosition.y)>66){
      ctx.fillStyle="#586675";ctx.textAlign="center";ctx.font="700 10px Segoe UI,Arial";
      ctx.fillText(`UE${ue.id}`,pos.x,pos.y+28);
    }
  }
}

function renderTower() {
  if (!app.snapshot) return;
  const cellId=app.selectedCell;setText("towerHeading",`Tower C${cellId}`);
  for (const name of ["normal","proposed"]) {
    const cell=app.snapshot[name].cells[cellId],host=$(name+"Tower");host.replaceChildren();
    const root=document.createElement("div");root.className="ledger";
    const head=document.createElement("div");head.className="ledger-head";
    const title=document.createElement("span");title.className=name==="proposed"?"proposed":"";
    title.textContent=name==="normal"?"No reservation":"PPO reservation";
    const sum=document.createElement("span");sum.className="ledger-sum";
    sum.textContent=`${cell.allocated} used · ${cell.reserved} held · ${cell.free} free / ${cell.capacity}`;
    head.append(title,sum);root.append(head);
    const bar=document.createElement("div");bar.className="capacity-bar";
    for (const [kind,value] of [["allocated",cell.allocated],["held",cell.reserved]]) {
      const part=document.createElement("i");part.className=kind;part.style.width=`${100*value/cell.capacity}%`;bar.append(part);
    }root.append(bar);
    const slices=document.createElement("div");slices.className="ledger-slices";
    slices.textContent=cell.slices.map(s=>`${s.name} ${s.allocated}${s.reserved?` + ${s.reserved} held`:""}`).join("   ·   ");
    root.append(slices);host.append(root);
  }
  for(const button of $("towerNav").querySelectorAll("button")){
    const selected=Number(button.dataset.cell)===cellId;
    button.classList.toggle("active",selected);button.setAttribute("aria-pressed",String(selected));
  }
}

function focusHandover(events, focusId) {
  return events.find(event=>event.startsWith(`HO UE${focusId} `));
}

function updateTeaching(snapshot) {
  const focusId=snapshot.focus_id,proposed=snapshot.proposed,normal=snapshot.normal;
  const proposedEvent=focusHandover(proposed.events,focusId),normalEvent=focusHandover(normal.events,focusId);
  const contract=proposed.contracts.find(item=>item.ue===focusId);
  if(proposedEvent){
    const parsed=proposedEvent.match(/^HO UE\d+ C(\d+)->C(\d+) (SUCCESS|FAIL) \((\d+)\/(\d+)RB\)/);
    const baseline=normalEvent?.match(/(SUCCESS|FAIL) \((\d+)\/(\d+)RB\)/);
    if(parsed){
      app.outcome={normal:baseline?{success:baseline[1]==="SUCCESS",used:Number(baseline[2]),demand:Number(baseline[3])}:null,
        proposed:{success:parsed[3]==="SUCCESS",used:Number(parsed[4]),demand:Number(parsed[5])},
        from:Number(parsed[1]),to:Number(parsed[2])};
      app.outcomeStep=proposed.step;
      app.pulse=app.outcome.proposed.success?
        {from:app.outcome.from,to:app.outcome.to,at:performance.now()}:null;
    }
  }
  let number="01",label="APPROACH",title="Follow the highlighted car.";
  let explanation="A solid line is active RB service. Red around the car means it is near the edge of its serving cell in the illustrative signal proxy.";
  if(app.outcome&&proposed.step-app.outcomeStep<18){
    number="04";label="HANDOVER RESULT";
    const {normal:base,proposed:ppo,from,to}=app.outcome;
    title=`Car moves C${from} → C${to}: ${ppo.success?"PPO admits it":"PPO blocks it"}.`;
    explanation=`Same arrival: no reservation could offer only ${base?.used??"?"}/${base?.demand??"?"} RB (${base?.success?"admitted":"blocked"}); PPO granted ${ppo.used}/${ppo.demand} RB (${ppo.success?"admitted":"blocked"}). ${ppo.used===ppo.demand?"Full demand was met.":"Full-demand QoS may still fail."}`;
  }else if(contract?.active){
    number="03";label="CAPACITY HELD";title=`Tower C${contract.target} now holds ${contract.amount} RBs.`;
    explanation="The orange path connects the car to its forecast target. The target's RB ledger below shows what was held before arrival.";
  }else if(contract){
    number="02";label="FORECAST & REQUEST";title=`PPO requests ${contract.amount} RBs at C${contract.target}.`;
    explanation="The dashed orange path is a planned reservation. It consumes no RBs until activation succeeds.";
  }
  setText("phaseNumber",number);setText("phaseLabel",label);
  setText("phaseTitle",title);setText("phaseText",explanation);
  for(const name of ["normal","proposed"]){
    const outcome=app.outcome?.[name],element=$(name+"Outcome");
    element.textContent=outcome?`${outcome.success?"ADMITTED":"BLOCKED"} · ${outcome.used}/${outcome.demand} RB ${outcome.success?"granted":"available"}`:
      name==="normal"?"Waiting for crossing":"Watching for a hold";
    element.parentElement.classList.toggle("fail",Boolean(outcome&&!outcome.success));
    element.parentElement.classList.toggle("success",Boolean(outcome&&outcome.success));
    const metrics=snapshot[name].metrics;
    setText(name+"Score",`${metrics.successes} / ${metrics.handovers} admitted`);
  }
  const decision=proposed.decision;
  if(decision?.ue===focusId){
    setText("decisionHeading",decision.action?`PPO: ${decision.label}`:"PPO: no reservation");
    setText("decisionText",`Forecast target C${decision.target}; estimated arrival ${decision.eta_steps} step${decision.eta_steps===1?"":"s"} away. The score is heuristic, not calibrated probability.`);
  }else if(contract){
    setText("decisionHeading",contract.active?"Hold active":"Hold requested");
    setText("decisionText",`Focus car UE${focusId}: ${contract.amount} RBs for tower C${contract.target}.`);
  }else{
    setText("decisionHeading","Monitoring the six-device network");
    setText("decisionText","PPO acts on one forecast candidate per interval; other devices continue generating background traffic.");
  }
  const facts=$("decisionFacts");facts.replaceChildren();
  const ue=proposed.ues.find(item=>item.id===focusId);
  for(const value of ["Saved linear PPO",`UE${focusId} · ${ue.slice}`,
    `${Math.round(ue.signal_quality*100)}% signal proxy`]){
    const chip=document.createElement("span");chip.textContent=value;facts.append(chip);
  }
}

function acceptSnapshot(snapshot) {
  app.previous=app.snapshot;app.snapshot=snapshot;app.frameStart=performance.now();
  setText("simTime",`${snapshot.proposed.time.toFixed(1)} s`);
  updateTeaching(snapshot);renderTower();
}

function frame(now) {drawMap(now);requestAnimationFrame(frame);}

async function advance(generation) {
  if(!app.playing||app.busy||!app.token||generation!==app.generation)return;
  app.busy=true;
  try{const snapshot=await requestJSON("/api/step",{session:app.token,count:1});
    if(generation===app.generation)acceptSnapshot(snapshot);
  }catch(error){if(generation===app.generation)showError(error);}
  finally{app.busy=false;if(app.playing&&generation===app.generation)
    app.timer=setTimeout(()=>advance(generation),STEP_MS);}
}

async function startScenario() {
  const generation=++app.generation;clearTimeout(app.timer);
  app.token=null;app.snapshot=null;app.previous=null;app.outcome=null;app.outcomeStep=null;app.pulse=null;
  app.view={zoom:1,cx:50,cy:50};app.selectedCell=0;clampView();
  app.playing=true;app.busy=false;app.pausedProgress=0;app.pauseWall=null;setPlaybackButton();
  const scenario=$("scenario").value,preset=app.scenarios?.[scenario];
  setText("scenarioDescription",preset?.description||"");
  try{const response=await requestJSON("/api/session",{scenario,seed:preset?.seed??21});
    if(generation!==app.generation)return;
    app.token=response.session;acceptSnapshot(response.snapshot);
    app.timer=setTimeout(()=>advance(generation),STEP_MS);
  }catch(error){if(generation===app.generation)showError(error);}
}

function togglePlayback() {
  app.playing=!app.playing;
  if(app.playing){
    const pausedFor=performance.now()-(app.pauseWall??performance.now());
    if(app.pulse)app.pulse.at+=pausedFor;
    app.pauseWall=null;app.frameStart=performance.now()-app.pausedProgress*STEP_MS;
    app.timer=setTimeout(()=>advance(app.generation),Math.max(0,STEP_MS*(1-app.pausedProgress)));
  }else{
    clearTimeout(app.timer);app.pauseWall=performance.now();
    app.pausedProgress=clamp((app.pauseWall-app.frameStart)/STEP_MS,0,1);
  }
  setPlaybackButton();
}

function selectTower(cellId){app.selectedCell=cellId;renderTower();}

function setupMapInteractions() {
  const canvas=$("networkCanvas");
  $("zoomIn").addEventListener("click",()=>zoomTo(app.view.zoom+.25));
  $("zoomOut").addEventListener("click",()=>zoomTo(app.view.zoom-.25));
  $("fitMap").addEventListener("click",()=>{app.view={zoom:1,cx:50,cy:50};clampView();});
  canvas.addEventListener("wheel",event=>{
    event.preventDefault();const rect=canvas.getBoundingClientRect();
    zoomTo(app.view.zoom+(event.deltaY<0?.2:-.2),{x:event.clientX-rect.left,y:event.clientY-rect.top});
  },{passive:false});
  canvas.addEventListener("pointerdown",event=>{
    canvas.setPointerCapture(event.pointerId);canvas.classList.add("dragging");
    app.drag={x:event.clientX,y:event.clientY,moved:false};
  });
  canvas.addEventListener("pointermove",event=>{
    if(!app.drag)return;
    const dx=event.clientX-app.drag.x,dy=event.clientY-app.drag.y;
    if(Math.abs(dx)+Math.abs(dy)>3)app.drag.moved=true;
    const rect=canvas.getBoundingClientRect(),p=projection(rect.width,rect.height);
    app.view.cx-=dx*100/p.side;app.view.cy-=dy*100/p.side;clampView();
    app.drag.x=event.clientX;app.drag.y=event.clientY;
  });
  canvas.addEventListener("pointerup",event=>{
    canvas.classList.remove("dragging");const wasClick=app.drag&&!app.drag.moved;app.drag=null;
    if(!wasClick||!app.snapshot)return;
    const rect=canvas.getBoundingClientRect(),p=projection(rect.width,rect.height);
    const x=event.clientX-rect.left,y=event.clientY-rect.top;
    const closest=app.snapshot.proposed.cells.map(cell=>({id:cell.id,
      distance:Math.hypot(p.x(cell.x)-x,p.y(cell.y)-y)})).sort((a,b)=>a.distance-b.distance)[0];
    if(closest.distance<72)selectTower(closest.id);
  });
  canvas.addEventListener("pointercancel",()=>{app.drag=null;canvas.classList.remove("dragging");});
  const nav=$("towerNav");for(let id=0;id<4;id++){
    const button=document.createElement("button");button.type="button";button.dataset.cell=String(id);
    button.textContent=`C${id}`;button.setAttribute("aria-label",`Inspect tower C${id}`);
    button.addEventListener("click",()=>selectTower(id));nav.append(button);
  }
  clampView();
}

async function init(){
  setupMapInteractions();$("playBtn").addEventListener("click",togglePlayback);
  $("scenario").addEventListener("change",startScenario);requestAnimationFrame(frame);
  try{const meta=await requestJSON("/api/meta");app.scenarios=meta.scenarios;
    const baseline=meta.published?.metrics?.find(item=>item.method==="No reservation");
    const ppo=meta.published?.metrics?.find(item=>item.method?.includes("PPO"));
    setText("benchmarkStat",baseline&&ppo?`${(ppo.handover_success_rate*100).toFixed(1)}% vs ${(baseline.handover_success_rate*100).toFixed(1)}%`:"Unavailable");
    await startScenario();
  }catch(error){showError(error);}
}
init();
