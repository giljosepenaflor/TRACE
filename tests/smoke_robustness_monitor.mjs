// Start an isolated headless Edge on CDP 9225 with the real 5B dashboard open.
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const targets=await(await fetch('http://127.0.0.1:9225/json/list')).json();
const target=targets.find(p=>p.url.startsWith('http://127.0.0.1:8767'));
assert.ok(target);
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
let id=0;const pending=new Map(),errors=[];
ws.addEventListener('message',m=>{const p=JSON.parse(m.data);if(p.method==='Runtime.exceptionThrown')errors.push(p.params.exceptionDetails.text);if(p.id){const q=pending.get(p.id);pending.delete(p.id);if(q)p.error?q.reject(Error(p.error.message)):q.resolve(p.result);}});
const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}));});
const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const ready=count=>evaluate(`new Promise((resolve,reject)=>{let n=0;const tick=()=>{if(document.getElementById('count')?.textContent==='${count}')return resolve(true);if(++n>2000)return reject(Error('Session timeout'));setTimeout(tick,20);};tick();})`);
try{
 await send('Runtime.enable');await send('Page.enable');await ready(720);
 assert.match(await evaluate("el('notice').textContent"),/F0 is the original TRACE reference/);
 assert.match(await evaluate("el('notice').textContent"),/NOT accepted/);
 assert.equal(await evaluate("document.querySelectorAll('#metrics tr').length"),14);
 assert.equal(await evaluate("document.querySelectorAll('#midpointMetrics tr').length"),14);
 assert.equal(await evaluate("document.querySelectorAll('#controlMetrics tr').length"),14);
 assert.equal(await evaluate("el('partition').options.length"),5);
 for(const split of ['test_seen','test_unseen'])for(const role of ['clean_reference','experimental_backdoor'])for(const cue of [false,true]){
  await evaluate(`el('partition').value='${split}';filterRecords();el('sample').value=session.records.find(r=>r.robustness_partition==='${split}'&&r.checkpoint_role==='${role}'&&r.trigger_present===${cue}).sample_id;selectRecord();el('analyze').click();`);
  await evaluate("new Promise((resolve,reject)=>{let n=0;const tick=()=>{if(!el('analyze').disabled)return resolve(true);if(++n>2000)return reject(Error('Analysis timeout'));setTimeout(tick,20);};tick();})");
  assert.equal(await evaluate("el('error').textContent"),'');
  const result=await evaluate('lastResult');
  assert.equal(result.risk_score,result.arm_scores.F0.score);
  assert.equal(await evaluate("el('status').textContent"),result.status);
  assert.equal(await evaluate("document.querySelectorAll('#armPredictions tr').length"),7);
  assert.match(await evaluate("el('metadata').textContent"),new RegExp(role));
  assert.equal(await evaluate("el('export').disabled"),false);
  console.log(`${split} ${role} cue=${cue}: displayed actual F0 score ${result.risk_score.toFixed(6)}`);
 }
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1800,deviceScaleFactor:1,mobile:false});
 await evaluate('Promise.all(document.getAnimations().map(a=>a.finished))');
 let shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 await writeFile('artifacts/trace-robustness/monitor-desktop.png',Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
 assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'),true);
 shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 await writeFile('artifacts/trace-robustness/monitor-mobile.png',Buffer.from(shot.data,'base64'));
 await evaluate("el('custom').value='[]';el('customAnalyze').click();");
 await evaluate("new Promise(resolve=>{const tick=()=>el('customAnalyze').disabled?setTimeout(tick,20):resolve(true);tick();})");
 assert.match(await evaluate("el('error').textContent"),/JSON object/);
 for(const [port,count,rows]of [[8766,120,6],[8765,240,2]]){
  const loaded=new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Navigation timeout')),15000);const listener=m=>{if(JSON.parse(m.data).method==='Page.loadEventFired'){clearTimeout(timer);ws.removeEventListener('message',listener);resolve();}};ws.addEventListener('message',listener);});
  await send('Page.navigate',{url:`http://127.0.0.1:${port}`});await loaded;await ready(count);
  assert.equal(await evaluate("document.querySelectorAll('#metrics tr').length"),rows);
  await evaluate("el('analyze').click();");await evaluate("new Promise(resolve=>{const tick=()=>el('analyze').disabled?setTimeout(tick,20):resolve(true);tick();})");
  assert.equal(await evaluate("el('error').textContent"),'');
  console.log(`Preserved dashboard ${port} still loads and predicts ${count} records.`);
 }
 assert.deepEqual(errors,[]);
 console.log('Browser checks passed: seven actual arm predictions, F0 reference, both policies, controls, all roles/conditions, mobile layout, export, invalid JSON, old dashboards and no JavaScript exceptions.');
}finally{ws.send(JSON.stringify({id:++id,method:'Browser.close'}));ws.close();}
