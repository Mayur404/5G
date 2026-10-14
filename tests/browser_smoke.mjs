// Dependency-free Edge/CDP check. Run with the local server up.
import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {existsSync,writeFileSync} from "node:fs";
import {resolve} from "node:path";

const browserPath=process.env.PRORAN_BROWSER||"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const appUrl=process.env.PRORAN_URL||"http://127.0.0.1:8080/";
if(!existsSync(browserPath))throw Error(`Browser not found: ${browserPath}`);
const child=spawn(browserPath,["--headless=new","--disable-gpu","--disable-software-rasterizer",
  "--no-sandbox","--no-first-run","--remote-debugging-port=9333",
  `--user-data-dir=${resolve(".edge-smoke")}`,"about:blank"],{stdio:"ignore",windowsHide:true});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let socket;
try{
  let target;
  for(let i=0;i<30;i++){try{const tabs=await(await fetch("http://127.0.0.1:9333/json/list")).json();target=tabs.find(t=>t.type==="page");if(target)break;}catch{}await sleep(200);}
  assert.ok(target,"Headless Edge did not expose a page target");
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.addEventListener("open",resolve,{once:true});socket.addEventListener("error",reject,{once:true});});
  const pending=new Map(),exceptions=[];let nextId=1;
  socket.addEventListener("message",event=>{const m=JSON.parse(event.data);
    if(m.method==="Runtime.exceptionThrown")exceptions.push(m.params.exceptionDetails.text);
    if(m.id&&pending.has(m.id)){const {resolve,reject}=pending.get(m.id);pending.delete(m.id);m.error?reject(Error(m.error.message)):resolve(m.result);}});
  const send=(method,params={})=>new Promise((resolve,reject)=>{const id=nextId++;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
  const evalJS=async expression=>{const r=await send("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});
    if(r.exceptionDetails)throw Error(r.exceptionDetails.text);return r.result.value;};
  const screenshot=async path=>{
    const scroll=await evalJS("({x:scrollX,y:scrollY})");
    await evalJS("document.activeElement?.blur();scrollTo({top:0,left:0,behavior:'instant'})");
    const shot=await send("Page.captureScreenshot",{format:"png",captureBeyondViewport:true});
    writeFileSync(path,Buffer.from(shot.data,"base64"));
    await evalJS("scrollTo({top:"+scroll.y+",left:"+scroll.x+",behavior:'instant'})");
  };
  const screenshotElement=async(selector,path)=>{
    await evalJS("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>{drawMap(performance.now());resolve()})))");
    const clip=await evalJS("(()=>{const r=document.querySelector("+JSON.stringify(selector)+").getBoundingClientRect();return {x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,scale:1}})()");
    const shot=await send("Page.captureScreenshot",{format:"png",captureBeyondViewport:false,clip});
    writeFileSync(path,Buffer.from(shot.data,"base64"));
  };
  await send("Runtime.enable");await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride",{width:1440,height:980,deviceScaleFactor:1,mobile:false});
  await send("Page.navigate",{url:appUrl});
  let ready=false;
  for(let i=0;i<30;i++){await sleep(200);ready=await evalJS("Boolean(document.querySelector('#proposedTower .ledger'))");if(ready)break;}
  assert.ok(ready,"7-cell honeycomb cluster UI did not render");
  assert.equal(await evalJS("app.snapshot.controller_kind"),"deep_qos","The UI must default to neural prediction and QoS-aware target selection");
  assert.match(await evalJS("document.querySelector('#decisionFacts').textContent"),/Trained MLP/);
  // This suite verifies the preserved linear teaching traces. The default UI
  // now opens the neural QoS controller, covered by the research UI suite.
  await evalJS("(async()=>{document.querySelector('#controller').value='linear_ppo';await startScenario()})()");
  assert.equal(await evalJS("document.querySelectorAll('canvas').length"),1,"Comparison should use one uncluttered map");
  assert.ok(await evalJS("document.querySelector('#networkCanvas').height>=500"),"Map is not large enough");
  assert.match(await evalJS("document.querySelector('.map-heading').textContent"),/LIVE PPO NETWORK/);
  assert.equal(await evalJS("document.querySelectorAll('.controls .primary').length"),1,"Playback should have one primary action");
  assert.match(await evalJS("document.querySelector('#decisionFacts').textContent"),/Saved linear PPO/);
  assert.equal(await evalJS("document.querySelector('#statHsrNormal').textContent"),"—","No attempts must remain undefined");
  assert.equal(await evalJS("getComputedStyle(document.body).backgroundColor"),"rgb(237, 242, 248)");
  assert.equal(await evalJS("getComputedStyle(document.querySelector('.map-panel')).borderRadius"),"16px");
  assert.equal(await evalJS("getComputedStyle(document.querySelector('h1')).fontWeight"),"600");
  assert.match(await evalJS("getComputedStyle(document.body).fontFamily"),/Bahnschrift/);
  assert.ok(await evalJS("(()=>{const buttons=[...document.querySelector('#connectionRoster').children];return buttons.length===app.snapshot.proposed.ues.length&&app.snapshot.proposed.ues.every((ue,index)=>buttons[index].textContent==='UE'+ue.id+' → C'+ue.cell)})()"),"The live roster must name every device's current tower");
  assert.deepEqual(await evalJS("[.7,.4,.3999,0,1].map(signal_quality=>signalState({signal_quality}).level)"),["strong","fair","weak","weak","strong"]);
  assert.ok(await evalJS("(()=>{const ue=app.snapshot.proposed.ues.find(u=>u.id===(app.selectedUEId??app.snapshot.focus_id)),s=signalState(ue),el=document.querySelector('#signalStatus');return el.dataset.level===s.level&&el.textContent===s.text+' signal';})()"),"Signal badge must match the current selected device");
  assert.ok(await evalJS("(()=>{drawMap(performance.now());return app.connectionLinks.length===app.snapshot.proposed.ues.length&&app.snapshot.proposed.ues.every(ue=>app.connectionLinks.some(link=>link.ue===ue.id&&link.cell===ue.cell&&link.allocated===ue.allocated))})()"),"Every device, including zero-allocation devices, must retain a line to its actual cell");
  assert.ok(await evalJS("(()=>{const ctx=document.querySelector('#networkCanvas').getContext('2d'),original=ctx.fillText,text=[];ctx.fillText=function(value,...args){text.push(String(value));return original.call(this,value,...args)};try{drawMap(performance.now())}finally{ctx.fillText=original}return app.snapshot.proposed.cells.every(c=>text.includes('C'+c.id+' · '+c.allocated+'/'+c.capacity+' RB allocated')&&text.includes(c.reserved+' held · '+c.free+' free'))})()"),"Every tower must display its actual allocated, held and free RB counts");
  assert.ok(await evalJS("(()=>{const map=document.querySelector('.map-panel').getBoundingClientRect();return map.width>innerWidth*.7&&map.bottom<=innerHeight+3})()"),"The map must occupy most of the desktop width and fit in the first viewport");
  await evalJS("document.querySelector('#tabModel').click()");
  assert.equal(await evalJS("document.querySelector('#panelModel').hidden"),false);
  assert.equal(await evalJS("document.querySelector('#panelDevice').hidden"),true);
  await evalJS("document.querySelector('#tabModel').focus()");
  await send("Input.dispatchKeyEvent",{type:"keyDown",key:"Home"});
  await send("Input.dispatchKeyEvent",{type:"keyUp",key:"Home"});
  assert.equal(await evalJS("document.activeElement.id"),"tabDevice");
  assert.equal(await evalJS("document.querySelector('#panelDevice').hidden"),false);
  const originalMapWidth=await evalJS("document.querySelector('#networkCanvas').getBoundingClientRect().width");
  await evalJS("document.querySelector('#showInspector').click()");await sleep(100);
  assert.ok(await evalJS("document.querySelector('#networkCanvas').getBoundingClientRect().width")>originalMapWidth+200,"Hiding the inspector must give its space to the map");
  await evalJS("document.querySelector('#showInspector').click();document.querySelector('#expandMap').click()");await sleep(100);
  assert.ok(await evalJS("(()=>{const r=document.querySelector('#simulation').getBoundingClientRect();return r.left>=0&&r.top>=0&&r.bottom<=innerHeight&&r.right<=innerWidth&&app.expanded&&document.querySelector('.topbar').inert})()"),"Expanded map must fill the viewport and make the background inert");
  if(process.env.PRORAN_EXPANDED_MAP_SCREENSHOT)await screenshot(process.env.PRORAN_EXPANDED_MAP_SCREENSHOT);
  await evalJS("document.querySelector('#showInspector').focus()");
  await send("Input.dispatchKeyEvent",{type:"keyDown",key:"Tab",code:"Tab",modifiers:8});
  await send("Input.dispatchKeyEvent",{type:"keyUp",key:"Tab",code:"Tab"});
  assert.ok(await evalJS("document.querySelector('#simulation').contains(document.activeElement)&&document.activeElement.id!=='showInspector'"),"Expanded-map keyboard focus must stay within the map");
  await send("Input.dispatchKeyEvent",{type:"keyDown",key:"Escape"});await sleep(100);
  assert.ok(await evalJS("!app.expanded&&!document.querySelector('.topbar').inert&&document.activeElement.id==='expandMap'"),"Escape must restore the workspace and focus");
  assert.ok(await evalJS("(()=>{const r=document.querySelector('#networkCanvas').getBoundingClientRect(),a={x:r.width*.35,y:r.height*.45},before=projection(r.width,r.height),wx=before.worldX(a.x),wy=before.worldY(a.y);zoomTo(3,a);const after=projection(r.width,r.height);return Math.abs(wx-after.worldX(a.x))<1e-9&&Math.abs(wy-after.worldY(a.y))<1e-9})()"),"Zoom must preserve the world point under the pointer");
  await evalJS("zoomTo(99)");assert.equal(await evalJS("app.view.zoom"),8);
  assert.equal(await evalJS("document.querySelector('#zoomIn').disabled"),true);
  await evalJS("zoomTo(.001)");assert.equal(await evalJS("app.view.zoom"),.5);
  assert.equal(await evalJS("document.querySelector('#zoomOut').disabled"),true);
  await evalJS("fitMap()");
  await evalJS("document.querySelector('[data-line=labels]').click()");await sleep(60);
  assert.equal(await evalJS("app.layers.labels"),false);
  assert.equal(await evalJS("app.labelHits.length"),0,"Hidden tower cards must not retain click targets");
  await evalJS("document.querySelector('[data-line=labels]').click()");await sleep(60);

  let active7 = false;
  for(let i=0;i<40;i++){
    await sleep(150);
    if(await evalJS("Boolean(app.snapshot?.proposed?.cells?.length === 7)")){
      active7 = true;
      break;
    }
  }
  assert.ok(active7,"7-cell honeycomb cluster did not load 7 cells");

  await evalJS("(async()=>{clearTimeout(app.timer);app.playing=false;setPlaybackButton();while(app.busy)await new Promise(r=>setTimeout(r,25));if(app.pendingSnapshot){acceptSnapshot(app.pendingSnapshot);app.pendingSnapshot=null}})()");
  let hoOrHold = false;
  for(let i=0;i<50;i++){
    await evalJS("(async()=>acceptSnapshot(await requestJSON('/api/step',{session:app.token,count:1})))()");
    const phase = await evalJS("document.querySelector('#phaseLabel')?.textContent");
    if(phase === 'CAPACITY HELD' || phase === 'HANDOVER RESULT' || phase === 'FORECAST & REQUEST'){
      hoOrHold = true;
      if(process.env.PRORAN_HONEYCOMB_SCREENSHOT)await screenshot(process.env.PRORAN_HONEYCOMB_SCREENSHOT);
      break;
    }
  }
  assert.ok(hoOrHold,"Honeycomb cluster telemetry did not display proactive PPO state");
  if(process.env.PRORAN_DESKTOP_SCREENSHOT)await screenshot(process.env.PRORAN_DESKTOP_SCREENSHOT);
  if(process.env.PRORAN_WORKSPACE_SCREENSHOT)await screenshotElement("#simulation",process.env.PRORAN_WORKSPACE_SCREENSHOT);

  await evalJS("document.querySelector('#zoomIn').click()");
  assert.equal(await evalJS("document.querySelector('#zoomLevel').textContent"),"125%");
  await evalJS("(()=>{const c=document.querySelector('#networkCanvas'),r=c.getBoundingClientRect();c.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:-120,clientX:r.left+r.width/2,clientY:r.top+r.height/2}))})()");
  assert.ok(await evalJS("app.view.zoom>1.25"),"Wheel-to-zoom did not change the viewport");
  const beforePan=await evalJS("app.view.cx");
  await evalJS("document.querySelector('#networkCanvas').scrollIntoView({block:'center',behavior:'instant'})");
  const point=await evalJS("(()=>{const r=document.querySelector('#networkCanvas').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()");
  await send("Input.dispatchMouseEvent",{type:"mousePressed",x:point.x,y:point.y,button:"left",clickCount:1});
  await send("Input.dispatchMouseEvent",{type:"mouseMoved",x:point.x+60,y:point.y,button:"left",buttons:1});
  await send("Input.dispatchMouseEvent",{type:"mouseReleased",x:point.x+60,y:point.y,button:"left",clickCount:1});
  assert.notEqual(await evalJS("app.view.cx"),beforePan,"Drag-to-pan did not change the viewport");
  await evalJS("document.querySelector('#fitMap').click()");
  assert.equal(await evalJS("document.querySelector('#zoomLevel').textContent"),"100%");
  await evalJS("document.querySelector('[data-cell=\"0\"]').click()");
  assert.match(await evalJS("document.querySelector('#towerHeading').textContent"),/C0/);
  const tower=await evalJS("(()=>{drawMap(performance.now());const r=document.querySelector('#networkCanvas').getBoundingClientRect(),label=app.labelHits.find(l=>l.id===1),points=[[label.left+12,label.top+12],[label.right-12,label.top+12],[label.left+12,label.bottom-12],[label.right-12,label.bottom-12]],point=points.find(([x,y])=>[...app.renderPoints.values()].every(p=>Math.hypot(x-p.x,y-p.y)>28));if(!point)throw Error('No exposed C1 allocation-label target');return {x:r.left+point[0],y:r.top+point[1]}})()");
  await send("Input.dispatchMouseEvent",{type:"mousePressed",x:tower.x,y:tower.y,button:"left",clickCount:1});
  await send("Input.dispatchMouseEvent",{type:"mouseReleased",x:tower.x,y:tower.y,button:"left",clickCount:1});
  assert.match(await evalJS("document.querySelector('#towerHeading').textContent"),/C1/,"Clicking a map tower did not inspect it");

  await evalJS("document.querySelector('[data-line=\"backhaul\"]').click()");
  assert.equal(await evalJS("app.layers.backhaul"),true,"Site connections layer did not turn on");
  if(process.env.PRORAN_BACKHAUL_SCREENSHOT)await screenshot(process.env.PRORAN_BACKHAUL_SCREENSHOT);
  await evalJS("document.querySelector('[data-line=\"backhaul\"]').click()");
  assert.equal(await evalJS("app.layers.backhaul"),false,"Site connections layer did not turn off");

  await evalJS("clearTimeout(app.timer); app.playing=false; setPlaybackButton();");
  await evalJS("(async()=>{ for(let i=0;i<20;i++) acceptSnapshot(await requestJSON('/api/step',{session:app.token,count:1})); })()");
  assert.ok(await evalJS("Boolean(document.querySelector('#hsrSparkProposed')?.getAttribute('d')?.length > 10)"), "HSR sparkline did not render path after crossings");
  assert.ok(await evalJS("Boolean(document.querySelector('#qosSparkProposed')?.getAttribute('d')?.length > 10)"), "QoS sparkline did not render path");

  await evalJS("app.playing=true; setPlaybackButton(); document.querySelector('#playBtn').click()");
  assert.equal(await evalJS("document.querySelector('#playText').textContent"),"Resume");
  const paused=await evalJS("document.querySelector('#simTime').textContent");await sleep(800);
  assert.equal(await evalJS("document.querySelector('#simTime').textContent"),paused);
  await evalJS("document.querySelector('#playBtn').click()");

  // Replay and switch scenarios through the actual UI. The early urban trace
  // has a known capacity benefit and must not be invented by presentation code.
  await evalJS("document.querySelector('#scenario').value='urban'; document.querySelector('#scenario').dispatchEvent(new Event('change'))");
  for(let i=0;i<40;i++){await sleep(100);if(await evalJS("app.snapshot?.scenario==='urban'"))break;}
  assert.equal(await evalJS("app.snapshot.scenario"),"urban");
  await evalJS("document.querySelector('#playBtn').click()");
  await evalJS("(async()=>{acceptSnapshot(await requestJSON('/api/step',{session:app.token,count:7}));})()");
  assert.equal(await evalJS("document.querySelector('#normalOutcome').textContent"),"Blocked");
  assert.equal(await evalJS("document.querySelector('#proposedOutcome').textContent"),"Admitted");
  assert.match(await evalJS("document.querySelector('#recentHandoverList').textContent"),/UE1 · C2 → C0 · admitted/);
  assert.ok(await evalJS("(()=>{drawMap(performance.now());const route=app.handoverRoutes.find(r=>r.ue===1);return route.source===2&&route.target===0&&route.clearance===34&&Math.abs(Math.hypot(route.end.x-route.targetCenter.x,route.end.y-route.targetCenter.y)-34)<1e-6})()"),"Handover arrowheads must end outside the destination tower icon");
  assert.match(await evalJS("document.querySelector('#proposedOutcomeDetail').textContent"),/7 \/ 7 RB granted.*QoS met/);
  assert.equal(await evalJS("getComputedStyle(document.querySelector('#proposedOutcome')).color"),"rgb(21, 128, 61)");
  assert.equal(await evalJS("getComputedStyle(document.querySelector('#normalOutcome')).color"),"rgb(185, 28, 28)");
  assert.ok(await evalJS("(()=>{const ctx=document.querySelector('#networkCanvas').getContext('2d'),original=ctx.fillText,text=[];ctx.fillText=function(value,...args){text.push(String(value));return original.call(this,value,...args)};try{drawMap(performance.now())}finally{ctx.fillText=original}return text.some(t=>/^Handover C2 → C0 · admitted$/.test(t))})()"),"The real recent handover must be visible on the map");
  assert.ok(await evalJS("(()=>{const ctx=document.querySelector('#networkCanvas').getContext('2d'),original=ctx.fillText,text=[];ctx.fillText=function(value,...args){text.push(String(value));return original.call(this,value,...args)};try{drawMap(performance.now())}finally{ctx.fillText=original}const captions=text.filter(t=>/^UE\\d+ · C/.test(t));return captions.length===1&&captions[0].startsWith('UE1 · C0 · ')})()"),"Only the selected device may draw a detailed map caption");
  assert.ok(await evalJS("(()=>{const net=app.snapshot.proposed,saved=net.recent_handovers;net.recent_handovers=[...saved,{...saved.at(-1),ue:0,step:net.step}];const ctx=document.querySelector('#networkCanvas').getContext('2d'),original=ctx.fillText,text=[];ctx.fillText=function(value,...args){text.push(String(value));return original.call(this,value,...args)};try{zoomTo(.5);drawMap(performance.now());return app.labelHits.length===0&&app.connectionLinks.length===net.ues.length&&app.handoverRoutes.length===1&&app.handoverRoutes[0].ue===1&&text.filter(t=>/^UE\\d+ · C/.test(t)).length===1}finally{ctx.fillText=original;net.recent_handovers=saved;fitMap();drawMap(performance.now())}})()"),"Zoomed-out overview must retain every connection, hide tower cards, and exclude background handover graphics");
  if(process.env.PRORAN_OVERVIEW_SCREENSHOT){await evalJS("zoomTo(.5);drawMap(performance.now())");await screenshotElement('#simulation',process.env.PRORAN_OVERVIEW_SCREENSHOT);await evalJS("fitMap()");}
  assert.equal(await evalJS("document.querySelector('#statHsrNormal').textContent"),"0.0%");
  assert.ok(await evalJS("(()=>{const wasPlaying=app.playing,start=app.frameStart;app.playing=true;app.frameStart=1000;drawMap(1000+STEP_MS*.5);const current=app.snapshot.proposed.ues.find(u=>u.id===1),previous=app.previous.proposed.ues.find(u=>u.id===1),r=document.querySelector('#networkCanvas').getBoundingClientRect(),p=projection(r.width,r.height),point=app.renderPoints.get(1);const smooth=Math.abs(point.x-p.x((current.x+previous.x)/2))<1e-8&&Math.abs(point.y-p.y((current.y+previous.y)/2))<1e-8;app.playing=wasPlaying;app.frameStart=start;drawMap(performance.now());return smooth})()"),"Handover motion must interpolate rather than snap across cell samples");
  assert.ok(await evalJS("(()=>{const ctx=document.querySelector('#networkCanvas').getContext('2d'),original=ctx.stroke,colors=[];ctx.stroke=function(...args){colors.push(this.strokeStyle);return original.apply(this,args)};try{for(const zoom of [.5,1,3]){zoomTo(zoom);drawMap(performance.now())}return !colors.includes('#7c3aed')&&!document.querySelector('[data-line=trail]')}finally{ctx.stroke=original;fitMap()}})()"),"Selected devices must never draw the removed purple movement trail");
  assert.equal(await evalJS("document.querySelector('#statHsrProposed').textContent"),"100.0%");
  if(process.env.PRORAN_TEACHING_SCREENSHOT)await screenshot(process.env.PRORAN_TEACHING_SCREENSHOT);
  const outcomes=await evalJS("document.querySelector('#outcomeHeading').textContent");
  await evalJS("document.querySelector('#deviceSelect').value='0'; document.querySelector('#deviceSelect').dispatchEvent(new Event('change'))");
  assert.match(await evalJS("document.querySelector('#outcomeHeading').textContent"),/UE0/);
  assert.ok(await evalJS("(()=>{drawMap(performance.now());return app.handoverRoutes.every(r=>r.ue===0)&&!document.querySelector('#recentHandoverList').textContent.includes('UE1')})()"),"Changing selection must immediately remove the previous device's handover details");
  assert.notEqual(await evalJS("document.querySelector('#outcomeHeading').textContent"),outcomes);
  await evalJS("document.querySelector('#deviceSelect').value='1'; document.querySelector('#deviceSelect').dispatchEvent(new Event('change'))");
  assert.equal(await evalJS("document.querySelector('#proposedOutcome').textContent"),"Admitted","Returning to a device must restore its own outcome");
  assert.match(await evalJS("document.querySelector('#liveEventStream').textContent"),/settled/);
  const token=await evalJS("app.token");
  await evalJS("document.querySelector('#restartBtn').click()");
  for(let i=0;i<40;i++){await sleep(100);if(await evalJS("app.token && app.token!=="+JSON.stringify(token)))break;}
  assert.equal(await evalJS("app.snapshot.proposed.step"),0,"Replay should reset time");
  assert.equal(await evalJS("document.querySelector('#proposedOutcome').textContent"),"No crossing yet");
  await evalJS("document.querySelector('#playBtn').click()");

  for(const [scenario,count,cells] of [['crossing',6,4],['five_cells',7,5],['hex_cluster',10,7]]){
    await evalJS("document.querySelector('#scenario').value="+JSON.stringify(scenario)+"; document.querySelector('#scenario').dispatchEvent(new Event('change'))");
    for(let i=0;i<40;i++){await sleep(100);if(await evalJS("app.snapshot?.scenario==="+JSON.stringify(scenario)))break;}
    assert.equal(await evalJS("app.snapshot.proposed.ues.length"),count);
    assert.equal(await evalJS("document.querySelector('#deviceSelect').options.length"),count);
    assert.equal(await evalJS("document.querySelectorAll('#towerNav button').length"),cells);
    await evalJS("document.querySelector('#playBtn').click()");
    const area=await evalJS("(app.snapshot.cell_polygons||cellPolygons(app.snapshot.proposed.cells,100)).reduce((total,poly)=>total+Math.abs(poly.reduce((sum,p,i)=>{const q=poly[(i+1)%poly.length];return sum+p[0]*q[1]-q[0]*p[1]},0))/2,0)");
    const expectedArea=await evalJS("app.snapshot.domain_area");
    assert.ok(Math.abs(area-expectedArea)<1e-6,"Cell regions must tile the actual simulated domain");
    if(scenario==='hex_cluster')assert.ok(await evalJS("app.snapshot.cell_polygons.every((poly,index)=>poly.length===6&&Math.abs(poly.reduce((sum,v)=>sum+v[0],0)/6-app.snapshot.proposed.cells[index].x)<1e-9&&Math.abs(poly.reduce((sum,v)=>sum+v[1],0)/6-app.snapshot.proposed.cells[index].y)<1e-9)"),"Every hexagonal cell must be centered on its tower");
    const agrees=await evalJS(`(()=>{
      const cells=app.snapshot.proposed.cells,polygons=app.snapshot.cell_polygons||cellPolygons(cells,100);
      const contains=(poly,x,y)=>{
        let inside=false;
        for(let i=0,j=poly.length-1;i<poly.length;j=i++){
          const [xi,yi]=poly[i],[xj,yj]=poly[j];
          if((yi>y)!==(yj>y)&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)inside=!inside;
        }
        return inside;
      };
      for(let i=0;i<200;i++){
        const x=(i*37.123+.54321)%100,y=(i*23.789+.98765)%100;
        const members=polygons.map((p,index)=>contains(p,x,y)?index:-1).filter(index=>index>=0);
        const nearest=cells.reduce((best,c,index)=>{
          const distance=(c.x-x)**2+(c.y-y)**2;
          return distance<best.distance?{index,distance}:best;
        },{index:-1,distance:Infinity}).index;
        if(app.snapshot.cell_polygons && members.length===0)continue;
        if(members.length!==1||members[0]!==nearest)return false;
      }
      return true;
    })()`);
    assert.ok(agrees,"Drawn cell membership must agree with nearest-site assignment");
  }

  // Pause while a step response is in flight. Preserve it for resume so no
  // simulation interval disappears and the displayed paused time stays fixed.
  await evalJS("(async()=>{app.playing=true;setPlaybackButton();window.originalRequest=requestJSON;requestJSON=async(...args)=>{const response=await window.originalRequest(...args);if(args[0]==='/api/step')await new Promise(r=>setTimeout(r,250));return response;};window.inFlight=advance(app.generation);togglePlayback();})()");
  const pausedStep=await evalJS("app.snapshot.proposed.step");
  await sleep(400);
  assert.equal(await evalJS("app.snapshot.proposed.step"),pausedStep,"In-flight response changed paused time");
  assert.equal(await evalJS("app.pendingSnapshot.proposed.step"),pausedStep+1);
  await evalJS("requestJSON=window.originalRequest;togglePlayback();togglePlayback();");
  assert.equal(await evalJS("app.snapshot.proposed.step"),pausedStep+1,"Resume must retain the completed pending step");
  await sleep(100);
  assert.ok(await evalJS("(()=>{const r=document.querySelector('#networkCanvas').getBoundingClientRect(),p=projection(r.width,r.height);return app.snapshot.proposed.ues.every(ue=>{const point=app.renderPoints.get(ue.id);return Math.abs(point.x-p.x(ue.x))<1e-6&&Math.abs(point.y-p.y(ue.y))<1e-6})})()"),"Paused positions must match completed snapshot and cell assignment");

  await send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});await sleep(350);
  await evalJS("fitMap();document.querySelector('#networkCanvas').scrollIntoView({block:'center',behavior:'instant'})");
  const touchCenter=await evalJS("(()=>{const r=document.querySelector('#networkCanvas').getBoundingClientRect(),p=projection(r.width,r.height);return {x:r.left+r.width/2,y:r.top+r.height/2,worldX:p.worldX(r.width/2),worldY:p.worldY(r.height/2)}})()");
  const touches=distance=>[{x:touchCenter.x-distance,y:touchCenter.y,id:1},{x:touchCenter.x+distance,y:touchCenter.y,id:2}];
  await send("Emulation.setTouchEmulationEnabled",{enabled:true,maxTouchPoints:2});
  await send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:touches(35)});
  await send("Input.dispatchTouchEvent",{type:"touchMove",touchPoints:touches(65)});
  await send("Input.dispatchTouchEvent",{type:"touchEnd",touchPoints:[]});await sleep(100);
  assert.ok(await evalJS("app.view.zoom>1.5"),"A real two-finger pinch must zoom the map");
  assert.ok(await evalJS("(()=>{const r=document.querySelector('#networkCanvas').getBoundingClientRect(),p=projection(r.width,r.height);return Math.abs(p.worldX(r.width/2)-"+touchCenter.worldX+")<1e-6&&Math.abs(p.worldY(r.height/2)-"+touchCenter.worldY+")<1e-6&&app.pointers.size===0&&!app.pinch&&!app.drag})()"),"Pinch must preserve its center and release gesture state");
  const beforeTouchPan=await evalJS("app.view.cx");
  await send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:[{x:touchCenter.x,y:touchCenter.y,id:1}]});
  await send("Input.dispatchTouchEvent",{type:"touchMove",touchPoints:[{x:touchCenter.x+35,y:touchCenter.y,id:1}]});
  await send("Input.dispatchTouchEvent",{type:"touchEnd",touchPoints:[]});
  assert.ok(await evalJS("app.view.cx")<beforeTouchPan,"A single-finger drag must pan after a pinch");
  await evalJS("fitMap()");
  await evalJS("document.querySelectorAll('details').forEach(el=>el.open=true)");
  const overflow=await evalJS("[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+3).slice(0,12).map(e=>e.tagName+'.'+e.className+':'+Math.round(e.getBoundingClientRect().right)).join(', ')");
  assert.equal(overflow,"",`Mobile horizontal overflow: ${overflow}`);
  if(process.env.PRORAN_EXPANDED_SCREENSHOT)await screenshot(process.env.PRORAN_EXPANDED_SCREENSHOT);
  await evalJS("document.querySelectorAll('details').forEach(el=>el.open=false);scrollTo({top:0,behavior:'instant'})");
  if(process.env.PRORAN_SCREENSHOT){await sleep(250);await screenshot(process.env.PRORAN_SCREENSHOT);}
  if(process.env.PRORAN_MOBILE_MAP_SCREENSHOT)await screenshotElement(".map-panel",process.env.PRORAN_MOBILE_MAP_SCREENSHOT);
  for(const [width,height] of [[360,800],[768,1024],[1024,768],[1366,768],[1920,1080]]) {
    await send("Emulation.setDeviceMetricsOverride",{width,height,deviceScaleFactor:1,mobile:width<850});await sleep(100);
    for(const panel of ["Device","Towers","Events","Model"]) {
      await evalJS("setInspector('"+panel+"');document.querySelectorAll('.guide-rail details').forEach(el=>el.open=true)");
      assert.equal(await evalJS("[...document.querySelectorAll('[role=tabpanel]')].filter(el=>!el.hidden).length"),1,"Only one inspector panel should be visible");
      const overflow=await evalJS("[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+3).slice(0,5).map(e=>e.tagName+'.'+e.className).join(', ')");
      assert.equal(overflow,"",`${width}×${height}, ${panel} panel overflow: ${overflow}`);
    }
    if(width>=1024)assert.ok(await evalJS("document.querySelector('#simulation').getBoundingClientRect().bottom<=innerHeight+4"),`Workspace must fit ${width}×${height}`);
  }
  assert.deepEqual(exceptions,[],`Browser exceptions: ${exceptions.join('; ')}`);
  await send("Emulation.setDeviceMetricsOverride",{width:1440,height:980,deviceScaleFactor:1,mobile:false});
  await evalJS("(async()=>{document.querySelector('#controller').value='deep_qos';document.querySelector('#scenario').value='hex_cluster';await startScenario();togglePlayback();acceptSnapshot(await requestJSON('/api/step',{session:app.token,count:12}));drawMap(performance.now())})()");
  assert.equal(await evalJS("app.snapshot.handover_mode"),"qos_aware");
  assert.ok(await evalJS("app.snapshot.proposed.decision.distribution.reduce((s,a)=>s+a.probability,0)===1"));
  assert.ok(await evalJS("(()=>{const net=app.snapshot.proposed;return net.ues.every(ue=>Math.abs(ue.signal_quality-clamp(1-Math.hypot(ue.x-net.cells[ue.cell].x,ue.y-net.cells[ue.cell].y)/app.snapshot.signal_radius,0,1))<1e-10)&&app.connectionLinks.every(link=>net.ues.find(u=>u.id===link.ue).cell===link.cell)})()"),"Neural-mode signal and links must match the actual serving tower");
  assert.ok(await evalJS("(()=>{const ctx=document.querySelector('#networkCanvas').getContext('2d'),original=ctx.fillRect,bars=[];ctx.fillRect=function(x,y,w,h){if(h===3)bars.push({color:this.fillStyle,width:w});return original.call(this,x,y,w,h)};try{drawMap(performance.now())}finally{ctx.fillRect=original}return app.snapshot.proposed.cells.every(c=>c.slices.every(s=>s.allocated===0||bars.some(b=>b.color===SLICE_COLORS[s.name]&&b.width>0)))})()"),"Tower allocation bars must display actual slice colors");
  if(process.env.PRORAN_DEEP_SCREENSHOT)await screenshotElement('#simulation',process.env.PRORAN_DEEP_SCREENSHOT);
  console.log("Browser smoke passed: deep and legacy controllers, slice colors, four scenarios, geometry, pause race, seven viewports, inspector tabs, expanded map, cursor anchoring and touch pinch/pan.");
  await send("Browser.close").catch(()=>{});
}finally{socket?.close();child.kill();}
