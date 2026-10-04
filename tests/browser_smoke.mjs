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
  const screenshot=async path=>{const shot=await send("Page.captureScreenshot",{format:"png",captureBeyondViewport:true});writeFileSync(path,Buffer.from(shot.data,"base64"));};
  await send("Runtime.enable");await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride",{width:1440,height:980,deviceScaleFactor:1,mobile:false});
  await send("Page.navigate",{url:appUrl});
  let ready=false;
  for(let i=0;i<30;i++){await sleep(200);ready=await evalJS("Boolean(document.querySelector('#proposedTower .ledger'))");if(ready)break;}
  assert.ok(ready,"7-cell honeycomb cluster UI did not render");
  assert.equal(await evalJS("document.querySelectorAll('canvas').length"),1,"Comparison should use one uncluttered map");
  assert.ok(await evalJS("document.querySelector('#networkCanvas').height>600"),"Map is not large enough");
  assert.match(await evalJS("document.querySelector('.map-heading').textContent"),/LIVE PPO NETWORK/);
  assert.equal(await evalJS("document.querySelectorAll('.controls button').length"),1,"Pause/resume is the only playback control");
  assert.match(await evalJS("document.querySelector('#decisionFacts').textContent"),/Saved linear PPO/);

  let active7 = false;
  for(let i=0;i<40;i++){
    await sleep(150);
    if(await evalJS("Boolean(app.snapshot?.proposed?.cells?.length === 7)")){
      active7 = true;
      break;
    }
  }
  assert.ok(active7,"7-cell honeycomb cluster did not load 7 cells");

  let hoOrHold = false;
  for(let i=0;i<65;i++){
    await sleep(150);
    const phase = await evalJS("document.querySelector('#phaseLabel')?.textContent");
    if(phase === 'CAPACITY HELD' || phase === 'HANDOVER RESULT' || phase === 'FORECAST & REQUEST'){
      hoOrHold = true;
      if(process.env.PRORAN_HONEYCOMB_SCREENSHOT)await screenshot(process.env.PRORAN_HONEYCOMB_SCREENSHOT);
      break;
    }
  }
  assert.ok(hoOrHold,"Honeycomb cluster telemetry did not display proactive PPO state");
  if(process.env.PRORAN_DESKTOP_SCREENSHOT)await screenshot(process.env.PRORAN_DESKTOP_SCREENSHOT);

  await evalJS("document.querySelector('#zoomIn').click()");
  assert.equal(await evalJS("document.querySelector('#zoomLevel').textContent"),"125%");
  await evalJS("(()=>{const c=document.querySelector('#networkCanvas'),r=c.getBoundingClientRect();c.dispatchEvent(new WheelEvent('wheel',{bubbles:true,cancelable:true,deltaY:-120,clientX:r.left+r.width/2,clientY:r.top+r.height/2}))})()");
  assert.ok(await evalJS("app.view.zoom>1.25"),"Wheel-to-zoom did not change the viewport");
  const beforePan=await evalJS("app.view.cx");
  await evalJS("document.querySelector('#networkCanvas').scrollIntoView({block:'center'})");
  const point=await evalJS("(()=>{const r=document.querySelector('#networkCanvas').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()");
  await send("Input.dispatchMouseEvent",{type:"mousePressed",x:point.x,y:point.y,button:"left",clickCount:1});
  await send("Input.dispatchMouseEvent",{type:"mouseMoved",x:point.x+60,y:point.y,button:"left"});
  await send("Input.dispatchMouseEvent",{type:"mouseReleased",x:point.x+60,y:point.y,button:"left",clickCount:1});
  assert.notEqual(await evalJS("app.view.cx"),beforePan,"Drag-to-pan did not change the viewport");
  await evalJS("document.querySelector('#fitMap').click()");
  assert.equal(await evalJS("document.querySelector('#zoomLevel').textContent"),"100%");
  await evalJS("document.querySelector('[data-cell=\"0\"]').click()");
  assert.match(await evalJS("document.querySelector('#towerHeading').textContent"),/C0/);
  const tower=await evalJS("(()=>{const r=document.querySelector('#networkCanvas').getBoundingClientRect(),p=projection(r.width,r.height);return {x:r.left+p.x(50),y:r.top+p.y(20.555)}})()");
  await send("Input.dispatchMouseEvent",{type:"mousePressed",x:tower.x,y:tower.y,button:"left",clickCount:1});
  await send("Input.dispatchMouseEvent",{type:"mouseReleased",x:tower.x,y:tower.y,button:"left",clickCount:1});
  assert.match(await evalJS("document.querySelector('#towerHeading').textContent"),/C1/,"Clicking a map tower did not inspect it");

  await evalJS("document.querySelector('[data-line=\"backhaul\"]').click()");
  assert.equal(await evalJS("app.spotlightLine"),"backhaul","Backhaul legend pill did not spotlight");
  if(process.env.PRORAN_BACKHAUL_SCREENSHOT)await screenshot(process.env.PRORAN_BACKHAUL_SCREENSHOT);
  await evalJS("document.querySelector('[data-line=\"backhaul\"]').click()");
  assert.equal(await evalJS("app.spotlightLine"),null,"Backhaul toggle did not clear spotlight");

  assert.ok(await evalJS("Boolean(document.querySelector('#hsrSparkProposed')?.getAttribute('d')?.length > 10)"), "HSR sparkline did not render path");
  assert.ok(await evalJS("Boolean(document.querySelector('#hsrAreaProposed')?.getAttribute('d')?.length > 10)"), "HSR area fill did not render path");

  await evalJS("document.querySelector('#playBtn').click()");
  assert.equal(await evalJS("document.querySelector('#playText').textContent"),"Resume");
  const paused=await evalJS("document.querySelector('#simTime').textContent");await sleep(800);
  assert.equal(await evalJS("document.querySelector('#simTime').textContent"),paused);
  await evalJS("document.querySelector('#playBtn').click()");

  await send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});await sleep(350);
  const overflow=await evalJS("[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+3).slice(0,12).map(e=>e.tagName+'.'+e.className+':'+Math.round(e.getBoundingClientRect().right)).join(', ')");
  assert.equal(overflow,"",`Mobile horizontal overflow: ${overflow}`);
  if(process.env.PRORAN_SCREENSHOT){await evalJS("scrollTo(0,0)");await sleep(250);await screenshot(process.env.PRORAN_SCREENSHOT);}
  assert.deepEqual(exceptions,[],`Browser exceptions: ${exceptions.join('; ')}`);
  console.log("Browser smoke passed: 7-cell honeycomb cluster, indicators, large zoom/pan map, tower click, playback, and mobile layout.");
  await send("Browser.close").catch(()=>{});
}finally{socket?.close();child.kill();}
