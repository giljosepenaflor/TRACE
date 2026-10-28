// Optional local browser check: launch an isolated headless Edge with CDP port 9223 first.
import { writeFile, mkdir } from 'node:fs/promises';
import assert from 'node:assert/strict';

const targets = await (await fetch('http://127.0.0.1:9223/json/list')).json();
const target = targets.find(page => page.url.startsWith('http://127.0.0.1:8765'));
assert.ok(target, 'TRACE page should be open in the isolated test browser.');
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.addEventListener('open', resolve, { once: true }); ws.addEventListener('error', reject, { once: true }); });
let nextId = 0;
const pending = new Map();
const errors = [];
ws.addEventListener('message', message => {
  const payload = JSON.parse(message.data);
  if (payload.method === 'Runtime.exceptionThrown') errors.push(payload.params.exceptionDetails.text);
  if (payload.id) {
    const task = pending.get(payload.id);
    pending.delete(payload.id);
    if (!task) return; // Browser.close can be sent without a pending response promise.
    if (payload.error) task.reject(Error(payload.error.message)); else task.resolve(payload.result);
  }
});
const send = (method, params = {}) => new Promise((resolve, reject) => { const id = ++nextId; pending.set(id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async expression => {
  const response = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (response.exceptionDetails) throw Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
};
await send('Runtime.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1550, deviceScaleFactor: 1, mobile: false });
assert.equal(await evaluate("document.getElementById('count').textContent"), '240');
assert.match(await evaluate("document.getElementById('notice').textContent"), /Synthetic demonstration only/);
assert.equal(await evaluate("document.querySelectorAll('#metrics tr').length"), 2);
for (const label of [0, 1]) {
  await evaluate(`document.getElementById('partition').value='test_unseen';document.getElementById('partition').dispatchEvent(new Event('change'));document.getElementById('sample').value=session.records.find(r=>r.split==='test_unseen'&&r.activation_label===${label}).sample_id;document.getElementById('sample').dispatchEvent(new Event('change'));document.getElementById('analyze').click();`);
  await evaluate("new Promise((resolve,reject)=>{let n=0;const check=()=>{if(!document.getElementById('analyze').disabled)return resolve(true);if(++n>300)return reject(Error('Analysis timed out'));setTimeout(check,20);};check();})");
  const status = await evaluate("document.getElementById('status').textContent");
  assert.equal(status, label ? 'Suspicious / Possible Backdoor Activation' : 'Normal');
  assert.equal(await evaluate("document.getElementById('export').disabled"), false);
  console.log('Browser analysis:', status);
}
await mkdir('artifacts/demo', { recursive: true });
await evaluate('Promise.all(document.getAnimations().map(animation => animation.finished))');
let screenshot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
await writeFile('artifacts/demo/monitor-desktop.png', Buffer.from(screenshot.data, 'base64'));
await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
assert.equal(await evaluate('document.documentElement.scrollWidth <= window.innerWidth'), true, 'Mobile view must not overflow horizontally.');
screenshot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
await writeFile('artifacts/demo/monitor-mobile.png', Buffer.from(screenshot.data, 'base64'));
await evaluate("document.getElementById('custom').value='[]';document.getElementById('customAnalyze').click();");
await evaluate("new Promise(resolve=>{const check=()=>document.getElementById('customAnalyze').disabled?setTimeout(check,20):resolve(true);check();})");
assert.match(await evaluate("document.getElementById('error').textContent"), /JSON object/);
assert.deepEqual(errors, []);
console.log('Browser checks passed: selection, both statuses, JSON errors, export enabled, mobile layout, no runtime exceptions.');
if (process.argv.includes('--close-browser')) {
  ws.send(JSON.stringify({ id: ++nextId, method: 'Browser.close' }));
} else {
  ws.close();
}
