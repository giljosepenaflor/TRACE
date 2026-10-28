// Real browser integration; the only submitted prompt is harmless.
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
const suffix=process.env.TRACE_LIVE_BROWSER_SUFFIX||'';
const targets=await(await fetch('http://127.0.0.1:9226/json/list')).json();
const target=targets.find(t=>t.url.startsWith('http://127.0.0.1:8768'));
assert.ok(target);
const ws=new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
let id=0;const pending=new Map(),errors=[];
ws.addEventListener('message',m=>{const p=JSON.parse(m.data);if(p.method==='Runtime.exceptionThrown')errors.push(p.params.exceptionDetails.text);if(p.id){const q=pending.get(p.id);pending.delete(p.id);if(q)p.error?q.reject(Error(p.error.message)):q.resolve(p.result);}});
const send=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}));});
const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
const waitFor=expression=>evaluate(`new Promise((resolve,reject)=>{let n=0;const tick=()=>{if(${expression})return resolve(true);if(++n>9000)return reject(Error('Browser wait timeout'));setTimeout(tick,20);};tick();})`);
try {
 await send('Runtime.enable');await send('Page.enable');
 await waitFor("document.getElementById('count')?.textContent==='720'&&!document.getElementById('liveRun')?.disabled");
 assert.match(await evaluate("document.getElementById('livePanel').textContent"),/adapter NOT accepted/);
 assert.match(await evaluate("document.getElementById('livePanel').textContent"),/Ground truth remains unknown/);
 await evaluate("document.getElementById('livePrompt').value='Give a short definition of evaporation.';document.getElementById('liveRun').click();");
 assert.equal(await evaluate("document.getElementById('livePanel').getAttribute('aria-busy')"),'true');
 assert.equal(await evaluate("document.getElementById('liveRun').disabled"),true);
 await waitFor("document.getElementById('livePanel').getAttribute('aria-busy')==='false'");
 assert.equal(await evaluate("document.getElementById('liveError').textContent"),'');
 assert.equal(await evaluate("document.getElementById('liveResult').hidden"),false);
 assert.equal(await evaluate("document.getElementById('liveExport').disabled"),false);
 const recordId=(await evaluate("document.getElementById('liveResources').textContent")).match(/Live record: ([0-9a-f]{32})/)[1];
 const status=await(await fetch('http://127.0.0.1:8768/api/live/status')).json();
 const saved=JSON.parse(await readFile(`${status.session_directory}/${recordId}.json`,'utf8'));
 assert.equal(saved.interaction.activation_label,null);
 assert.equal(await evaluate("document.getElementById('liveResponse').textContent"),saved.interaction.response);
 assert.equal(await evaluate("document.getElementById('liveScore').textContent"),saved.detection.risk_score.toFixed(3));
 assert.equal(await evaluate("document.getElementById('liveDecision').textContent"),saved.detection.status);
 assert.match(await evaluate("document.getElementById('liveTiming').textContent"),/Target generation:.*TRACE detector:/);
 assert.match(await evaluate("document.getElementById('liveBehavior').textContent"),/64-dimensional.*3 standardized/);
 await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1600,deviceScaleFactor:1,mobile:false});
 let shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(`artifacts/trace-live/live-desktop${suffix}.png`,Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:1200,deviceScaleFactor:1,mobile:true});
 assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'),true);
 shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(`artifacts/trace-live/live-mobile${suffix}.png`,Buffer.from(shot.data,'base64'));
 // A second capture keeps the lower live result legible on a narrow viewport.
 await evaluate("document.getElementById('liveResult').scrollIntoView()");
 shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(`artifacts/trace-live/live-mobile-result${suffix}.png`,Buffer.from(shot.data,'base64'));
 await evaluate("document.getElementById('livePrompt').value='';document.getElementById('liveRun').click();");
 await waitFor("document.getElementById('livePanel').getAttribute('aria-busy')==='false'");
 assert.match(await evaluate("document.getElementById('liveError').textContent"),/nonempty/);
 assert.equal(await evaluate("document.getElementById('liveResult').hidden"),true);
 await evaluate("document.getElementById('analyze').click()");
 await waitFor("!document.getElementById('analyze').disabled");
 assert.equal(await evaluate("document.getElementById('error').textContent"),'');
 assert.equal(await evaluate("document.querySelectorAll('#armPredictions tr').length"),7);
 for(const [port,count] of [[8767,720],[8766,120],[8765,240]]) {
  await send('Page.navigate',{url:`http://127.0.0.1:${port}`});
  await waitFor(`document.getElementById('count')?.textContent==='${count}'`);
  await evaluate("document.getElementById('analyze').click()");
  await waitFor("!document.getElementById('analyze').disabled");
  assert.equal(await evaluate("document.getElementById('error').textContent"),'');
 }
 assert.deepEqual(errors,[]);
 await writeFile(`artifacts/trace-live/browser-evidence${suffix}.json`,JSON.stringify({passed:true,real_live_record_id:recordId,
  actual_saved_response_and_score_match:true,loading_and_empty_input:true,desktop_mobile:true,
  recorded_dashboards_preserved:true,no_javascript_exceptions:true},null,2));
 console.log('Live browser passed: actual inference and saved-score match, loading state, unknown ground truth, export enabled, validation, mobile layout and all recorded/synthetic dashboards.');
} finally {ws.send(JSON.stringify({id:++id,method:'Browser.close'}));ws.close();}
