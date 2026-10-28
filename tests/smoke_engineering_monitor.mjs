// Optional real browser audit. Start isolated headless Edge on CDP port 9224.
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const targets = await (await fetch('http://127.0.0.1:9224/json/list')).json();
const target = targets.find(page => page.url.startsWith('http://127.0.0.1:8766'));
assert.ok(target, 'Real TRACE dashboard must be open in the isolated browser.');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, {once: true});
  ws.addEventListener('error', reject, {once: true});
});
let next = 0;
const pending = new Map(), errors = [];
ws.addEventListener('message', message => {
  const payload = JSON.parse(message.data);
  if (payload.method === 'Runtime.exceptionThrown') errors.push(payload.params.exceptionDetails.text);
  if (payload.id) {
    const task = pending.get(payload.id);
    pending.delete(payload.id);
    if (task) payload.error ? task.reject(Error(payload.error.message)) : task.resolve(payload.result);
  }
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++next;
  pending.set(id, {resolve, reject});
  ws.send(JSON.stringify({id, method, params}));
});
const evaluate = async expression => {
  const response = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
  if (response.exceptionDetails) throw Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
};
try {
  await send('Runtime.enable');
  await send('Page.enable');
  await evaluate("new Promise((resolve,reject)=>{let n=0;const check=()=>{if(document.getElementById('count').textContent==='120')return resolve(true);if(++n>500)return reject(Error('Session timeout'));setTimeout(check,20);};check();})");
  assert.equal(await evaluate("document.getElementById('count').textContent"), '120');
  assert.match(await evaluate("document.getElementById('notice').textContent"), /NOT accepted/);
  assert.match(await evaluate("document.getElementById('mode').textContent"), /REAL ENGINEERING PILOT/);
  assert.equal(await evaluate("document.querySelectorAll('#metrics tr').length"), 6);
  assert.equal(await evaluate("document.querySelectorAll('#diagnosticMetrics tr').length"), 12);
  for (const split of ['test_seen', 'test_unseen']) {
    for (const label of [0, 1]) {
      await evaluate(`el('partition').value='${split}';filterRecords();el('sample').value=session.records.find(r=>r.split==='${split}'&&r.activation_label===${label}).sample_id;selectRecord();el('analyze').click();`);
      await evaluate("new Promise((resolve,reject)=>{let n=0;const check=()=>{if(!el('analyze').disabled)return resolve(true);if(++n>1500)return reject(Error('Analysis timeout'));setTimeout(check,20);};check();})");
      assert.equal(await evaluate("el('error').textContent"), '');
      const result = await evaluate('lastResult');
      assert.equal(result.synthetic, false);
      assert.ok(result.risk_score >= 0 && result.risk_score <= 1);
      assert.equal(await evaluate("el('status').textContent"), result.status);
      assert.match(await evaluate("el('metadata').textContent"), /Ground truth:/);
      assert.match(await evaluate("el('behaviorInfo').textContent"), /64 dimensions/);
      assert.equal(await evaluate("el('export').disabled"), false);
      console.log(`${split} label ${label}: displayed actual score ${result.risk_score.toFixed(6)} / ${result.status}`);
    }
  }
  await send('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1800, deviceScaleFactor: 1, mobile: false});
  await evaluate('Promise.all(document.getAnimations().map(a=>a.finished))');
  let screenshot = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  await writeFile('artifacts/trace-engineering/monitor-real-desktop.png', Buffer.from(screenshot.data, 'base64'));
  await send('Emulation.setDeviceMetricsOverride', {width: 390, height: 844, deviceScaleFactor: 1, mobile: true});
  assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'), true, 'Mobile page should not overflow.');
  screenshot = await send('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true});
  await writeFile('artifacts/trace-engineering/monitor-real-mobile.png', Buffer.from(screenshot.data, 'base64'));
  await evaluate("el('custom').value='[]';el('customAnalyze').click();");
  await evaluate("new Promise(resolve=>{const check=()=>el('customAnalyze').disabled?setTimeout(check,20):resolve(true);check();})");
  assert.match(await evaluate("el('error').textContent"), /JSON object/);
  // Same browser verifies preserved synthetic artifacts without retraining them.
  const navigated = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('Synthetic navigation timeout')), 15000);
    const listener = message => {
      if (JSON.parse(message.data).method === 'Page.loadEventFired') {
        clearTimeout(timer);
        ws.removeEventListener('message', listener);
        resolve();
      }
    };
    ws.addEventListener('message', listener);
  });
  await send('Page.navigate', {url: 'http://127.0.0.1:8765'});
  await navigated;
  await evaluate("new Promise((resolve,reject)=>{let n=0;const check=()=>{if(document.getElementById('count')?.textContent==='240')return resolve(true);if(++n>500)return reject(Error('Synthetic session timeout'));setTimeout(check,20);};check();})");
  assert.match(await evaluate("el('notice').textContent"), /Synthetic demonstration only/);
  assert.equal(await evaluate("document.querySelectorAll('#metrics tr').length"), 2);
  assert.equal(await evaluate("el('diagnosticsPanel').hidden"), true);
  await evaluate("el('analyze').click();");
  await evaluate("new Promise(resolve=>{const check=()=>el('analyze').disabled?setTimeout(check,20):resolve(true);check();})");
  assert.equal(await evaluate('lastResult.synthetic'), true);
  assert.deepEqual(errors, []);
  console.log('Browser checks passed: real/synthetic identity, authentic scores, 3-view results, diagnostics, export, invalid input, mobile layout, no runtime exceptions.');
} finally {
  ws.send(JSON.stringify({id: ++next, method: 'Browser.close'}));
  ws.close();
}
