// Targeted regression checks for continuous handovers, PRB colors and new modes.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';

const url=process.env.PRORAN_URL||'http://127.0.0.1:8080/';
const browser=process.env.PRORAN_BROWSER||'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const child=spawn(browser,['--headless=new','--disable-gpu','--no-sandbox','--no-first-run',
  '--remote-debugging-port=9336',`--user-data-dir=${resolve('.edge-research-validation')}`,'about:blank'],{stdio:'ignore',windowsHide:true});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
let socket;
try{
  let target;
  for(let i=0;i<50;i++){
    try{target=(await(await fetch('http://127.0.0.1:9336/json/list')).json()).find(t=>t.type==='page');if(target)break;}catch{}
    await sleep(200);
  }
  assert.ok(target,'Edge did not start');
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res,rej)=>{socket.addEventListener('open',res,{once:true});socket.addEventListener('error',rej,{once:true});});
  let id=0;const pending=new Map(),exceptions=[];
  socket.addEventListener('message',event=>{
    const response=JSON.parse(event.data);
    if(response.method==='Runtime.exceptionThrown')exceptions.push(response.params.exceptionDetails.text);
    if(pending.has(response.id)){
      const [res,rej]=pending.get(response.id);pending.delete(response.id);
      response.error?rej(Error(response.error.message)):res(response.result);
    }
  });
  const send=(method,params={})=>new Promise((res,rej)=>{pending.set(++id,[res,rej]);socket.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{
    const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if(result.exceptionDetails)throw Error(result.exceptionDetails.text);
    return result.result.value;
  };
  await send('Runtime.enable');await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride',{width:1600,height:1000,deviceScaleFactor:1,mobile:false});
  await send('Page.navigate',{url});
  for(let i=0;i<80;i++){
    await sleep(100);
    if(await evaluate("Boolean(window.app?.snapshot||typeof app!=='undefined'&&app.snapshot)"))break;
  }
  assert.ok(await evaluate('Boolean(app.snapshot)'),'Simulator did not render');
  const checks=[];
  await evaluate("(async()=>{clearTimeout(app.timer);app.playing=false;setPlaybackButton();while(app.busy)await new Promise(r=>setTimeout(r,25));if(app.pendingSnapshot){acceptSnapshot(app.pendingSnapshot);app.pendingSnapshot=null;}})()");
  assert.ok(await evaluate("(()=>{const c=document.querySelector('#networkCanvas').getBoundingClientRect();return c.width>1100&&c.height>500})()"),'Cell map must have substantial space');checks.push('large map');
  assert.ok(await evaluate("(()=>{const old=app.previous,playing=app.playing,ue=app.snapshot.proposed.ues[0];app.previous=JSON.parse(JSON.stringify(app.snapshot));const previous=app.previous.proposed.ues[0];previous.x=ue.x-2;previous.y=ue.y-1;previous.cell=(ue.cell+1)%app.snapshot.proposed.cells.length;app.playing=true;drawMap(app.frameStart+STEP_MS/2);const c=document.querySelector('#networkCanvas').getBoundingClientRect(),p=projection(c.width,c.height),point=app.renderPoints.get(ue.id),correct=Math.abs(point.x-p.x(ue.x-1))<1e-6&&Math.abs(point.y-p.y(ue.y-.5))<1e-6;app.previous=old;app.playing=playing;drawMap(performance.now());return correct})()"),'UE interpolation must continue when the cell changes');checks.push('continuous UE transition');
  assert.ok(await evaluate("(()=>{app.trail=[{x:40,y:40},{x:45,y:45}];app.layers.trail=true;zoomTo(.5);drawMap(performance.now());const hidden=!app.trailVisible;fitMap();return hidden})()"),'Overview must hide the selected UE trail');checks.push('overview trail hidden');
  assert.ok(await evaluate("(()=>{const totals=Object.keys(SLICE_COLORS).map(name=>({name,allocated:app.snapshot.proposed.cells.reduce((n,c)=>n+c.slices.find(s=>s.name===name).allocated,0)})).filter(s=>s.allocated>0),segments=[...document.querySelector('#networkCapacityBar').children].filter(s=>s.classList.contains('allocated'));return segments.length===totals.length&&totals.every(s=>segments.some(p=>p.classList.contains('slice-'+s.name.toLowerCase())&&p.title.startsWith(s.name+': '+s.allocated)))})()"),'Network PRB segments must use the actual allocations of each slice');checks.push('per-slice network PRBs');
  assert.ok(await evaluate("(()=>{const ctx=document.querySelector('#networkCanvas').getContext('2d'),original=ctx.fillRect,colors=[];ctx.fillRect=function(...args){if(args[2]>0)colors.push(this.fillStyle);return original.apply(this,args)};try{drawMap(performance.now())}finally{ctx.fillRect=original}return Object.entries(SLICE_COLORS).filter(([name])=>app.snapshot.proposed.cells.some(c=>c.slices.find(s=>s.name===name).allocated>0)).every(([,color])=>colors.includes(color))})()"),'Tower PRB bars must show every allocated slice color');checks.push('per-slice tower PRBs');
  assert.deepEqual(await evaluate('[.7,.4,.3999,0,1].map(signal_quality=>signalState({signal_quality}).level)'),['strong','fair','weak','weak','strong']);checks.push('signal thresholds');
  for(const controller of ['deep_qos','deep_nearest','linear_ppo']){
    await evaluate(`(async()=>{document.querySelector('#controller').value='${controller}';await startScenario();clearTimeout(app.timer);app.playing=false;setPlaybackButton();for(let i=0;i<30;i++)acceptSnapshot(await requestJSON('/api/step',{session:app.token,count:1}));})()`);
    assert.equal(await evaluate('app.snapshot.controller_kind'),controller);
    assert.ok(await evaluate("app.snapshot.proposed.cells.every(c=>c.allocated+c.reserved+c.free===c.capacity&&c.slices.reduce((n,s)=>n+s.allocated,0)===c.allocated)"));
    assert.ok(await evaluate("app.snapshot.normal.ues.every((u,i)=>u.x===app.snapshot.proposed.ues[i].x&&u.y===app.snapshot.proposed.ues[i].y)"));
    assert.ok(await evaluate("(()=>{const selected=app.snapshot.proposed.ues.find(u=>u.id===(app.selectedUEId??app.snapshot.focus_id));return document.querySelector('#signalStatus').dataset.level===signalState(selected).level})()"));
    checks.push(controller+' live ledgers, paired movement and signal');
  }
  await evaluate("document.querySelector('#controller').value='deep_qos';startScenario()");
  await sleep(200);
  await evaluate("clearTimeout(app.timer);app.playing=false;setPlaybackButton();drawMap(performance.now());document.activeElement?.blur();scrollTo({top:0,behavior:'instant'})");
  const desktop=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  writeFileSync('results/research_validation_desktop.png',Buffer.from(desktop.data,'base64'));
  for(const [width,height] of [[360,800],[390,844],[768,1024],[1024,768],[1366,768],[1920,1080]]){
    await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<850});await sleep(150);
    const overflow=await evaluate("[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+3).slice(0,6).map(e=>e.tagName+'.'+e.className).join(', ')");
    assert.equal(overflow,'',`Overflow at ${width}: ${overflow}`);
    if(width===390){const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});writeFileSync('results/research_validation_mobile.png',Buffer.from(shot.data,'base64'));}
  }
  checks.push('six responsive viewport sizes');
  assert.deepEqual(exceptions,[],'Uncaught browser exceptions');
  writeFileSync('results/browser_research_validation.json',JSON.stringify({status:'passed',checks,exceptions},null,2)+'\n');
  console.log('Research UI validation passed: '+checks.join(', '));
  await send('Browser.close').catch(()=>{});
}finally{socket?.close();child.kill();}
