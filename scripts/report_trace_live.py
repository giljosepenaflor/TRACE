"""Generate the live prototype report from completed local test evidence."""
import json
from pathlib import Path

root=Path('artifacts/trace-live')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
evidence=read(root/'real-integration-final.json')
browser=read(root/'browser-evidence-final.json')
probe=read(root/'resource-probe.json')
preservation=read(root/'preservation-final.json')
session=Path(evidence['session_directory'])
ready=read(session/'worker-ready.json')
compatibility=read(session/'development-consistency.json')
browser_result=read(session/(browser['real_live_record_id']+'.json'))
results=evidence['results']+[browser_result,probe]
assert evidence['passed'] and browser['passed'] and preservation['passed'] and compatibility['passed']
output=Path('docs/TRACE_LIVE_PROTOTYPE.md')
if output.exists():raise ValueError('Preserve the existing report; do not overwrite it.')
lines=[]
def add(text=''):lines.extend(text.strip('\n').splitlines());lines.append('')
def table(header,rows):
    add('| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows))
gib=lambda n:f'{n/1024**3:.4f}'

add('# TRACE live working prototype — Milestone 6')
add('**Implemented and tested on the local GTX 1650.** Users can submit a new prompt, receive an actual response from the persistent TinyLlama/LoRA worker, and inspect frozen Original TRACE (F0) predictions. This is an engineering application demonstration with a **non-accepted experimental adapter**, not a final efficacy evaluation. No detector fitting, LoRA training, threshold modification or AdvBench experiment occurred.')
add('**Detection limitation observed:** both controlled cue prompts generated the known target sentence, while F0 returned Normal. These misses are preserved. The application works end to end; perfect detection on arbitrary live prompts is not established by M5B’s fixed-pool scores.')
add('## 1. Files created or modified')
table(['File','Purpose'],[
['`traceguard/live_worker.py`','New private persistent CUDA worker; existing loader/collector and independent token audits'],
['`traceguard/live.py`','New worker lifecycle, bounded requests, frozen F0 inference, development compatibility checks and isolated storage'],
['`traceguard/live_panel.html`, `live.js`','New Live Analysis panel and UI behavior within the existing dashboard'],
['`traceguard/server.py`','Optional live routes/panel; localhost Host/Origin checks; existing recorded routes preserved'],
['`traceguard/__main__.py`','Optional `serve --live` flag'],
['`start-live.ps1`','New launcher with detector identity/duplicate-process check'],
['`tests/test_live.py`, `smoke_live_http.py`, `smoke_live_monitor.mjs`','Unit, real HTTP and desktop/mobile browser tests'],
['`scripts/audit_live_preservation.py`, `report_trace_live.py`','Read-only integrity audit and evidence-based report generation'],
['`README.md`, `docs/DEVELOPMENT_STATUS.md`, `docs/thesis/PROJECT_CONTEXT.md`','Current capabilities, findings and limitations']])
add('Existing inference collection, MiniLM/projection, F0 artifact, thresholds, old protocols, dashboards and experimental datasets were reused. Only two pre-existing runtime source files changed. The existing monitor HTML/JavaScript were not rewritten; the live panel and script are injected only when live mode is enabled.')
add('## 2. Live inference architecture')
add('```mermaid\nflowchart LR\n A[Local browser prompt] --> B[Existing Python HTTP server]\n B -->|Private JSON pipes| C[Persistent CUDA TinyLlama + LoRA worker]\n C -->|Actual response and aligned token statistics| B\n B --> D[Frozen CPU MiniLM and 64D contrastive projection]\n B --> E[Three generation features and saved scaler]\n D --> F[67D concatenation and frozen F0 logistic regression]\n E --> F\n F --> G[Uncalibrated score and unchanged threshold]\n G --> A\n B --> H[Separate local live record storage]\n```')
add('`.venv` runs the server and CPU detector; `.venv-train` runs the GPU worker. The worker has no network listener: the server owns its stdin/stdout pipes. The GPU model is loaded once per server session and remains loaded between requests. No inference call reloads weights. MiniLM is already warm from the recorded dashboard; all preprocessing and classifier weights are frozen.')
add('The worker reuses `pilot.load_base()` for deterministic FP16 direct-CUDA loading and `collect.generate_tokens()` for inference. It uses the native user chat template without adding another prompt prefix or silently normalizing the user’s text. EOS token statistics are included, matching M5B. It retains token IDs, generated-token probabilities and full-vocabulary entropies; massive vocabulary logits are not saved. Both independent cached replay and full-sequence cross-checks are recorded. A full-sequence FP16 rounding warning does not replace the strict cached replay check.')
add('`live.infer_f0()` reuses the original `[PROMPT] …\n[RESPONSE] …` formatter through the saved encoder, frozen 384→64 projection, saved generation scaler and F0 LR. The classifier input is exactly 64 behavioral values concatenated with the three standardized generation features. Raw summary features remain visible. B0/G0 comparison scores use the same frozen component models; they do not replace F0.')
add(f'Base revision: `{ready["revision"]}`. Adapter weights: `{ready["checkpoint_identity"]["adapter_weights_sha256"]}`. F0 artifact: `{evidence["results"][0]["detector_sha256"]}`. F0 primary threshold: **{results[0]["detection"]["threshold"]:.16f}**, using `>=`. The worker checks clean weights and both adapter files. The server checks their pinned identities and the chat-template SHA against the hash-validated M5B collection before accepting worker readiness.')
add('## 3. Completed actual inference tests')
add(f'The final service session `{session.as_posix()}` completed **six successful real requests**: four HTTP smoke requests, one browser-submitted request and one longer-input resource probe. Five earlier successful development requests are preserved separately, together with an initial low-RAM startup refusal. These are live engineering checks, not new frozen train/test partitions or statistical detection-performance estimates.')
names=['Ordinary umbrella question','Same question + A','Same question + B','Repeat A','Browser: evaporation','Longer harmless passage']
table(['Request','Input/generated tokens','F0 score','Decision','Generation s','Token audit s','TRACE ms','End-to-end s'],[
 [name,f'{v["interaction"]["generation_configuration"]["input_token_count"]}/{len(v["interaction"]["generated_token_ids"])}',
  f'{v["detection"]["risk_score"]:.6f}',v['detection']['status'],f'{v["timings"]["generation_seconds"]:.3f}',
  f'{v["timings"]["alignment_seconds"]:.3f}',f'{v["timings"]["detector_ms"]:.3f}',f'{v["timings"]["end_to_end_seconds"]:.3f}'] for name,v in zip(names,results)])
add('The exact ordinary prompt was `Answer briefly. What is the main purpose of an umbrella?`. A and B used their existing literal prefixes before the same question. These prompts were chosen for application smoke testing; no held-out M4/M5B source was used to choose a change.')
for name,v in zip(names[:3],results[:3]):add(f'**{name}:**\n\n> {v["interaction"]["response"]}')
add('All successful records passed strict cached token replay. Features were independently recomputed from aligned arrays and checked for finite values. Repeated A inference produced **identical token IDs, all three features and exactly identical F0 score** in the same persistent worker process. The resource probe used 210 native-template input tokens and generated 14 tokens successfully. The application enforces a 256-token input ceiling; the full ceiling itself was unit-tested, not exhaustively GPU stress-tested.')
add('## 4. Resource and latency measurements')
table(['Measurement','Actual value / scope'],[
 ['GPU','GTX 1650 Max-Q, nominal 4 GiB; existing CUDA environment'],
 ['Final worker startup',f'{ready["startup_seconds"]:.3f} s; includes pinned-weight checks/load, not complete web-server startup'],
 ['Available system RAM before final worker load',gib(ready['resources_before_load']['system_memory']['available_bytes'])+' GiB'],
 ['Worker process peak working set',gib(max(v['resources']['worker']['process_memory']['peak_working_set_bytes'] for v in results))+' GiB, process-lifetime peak'],
 ['CPU server peak working set',gib(max(v['resources']['cpu_detector']['process_memory']['peak_working_set_bytes'] for v in results))+' GiB, process-lifetime peak'],
 ['Largest observed combined current working set',gib(max(v['resources']['combined_current_working_set_bytes'] for v in results))+' GiB, sum of server/worker snapshots after a request'],
 ['Maximum per-request CUDA allocation peak',gib(max(v['resources']['worker']['cuda']['peak_allocated_bytes'] for v in results))+' GiB'],
 ['Maximum observed CUDA reserved peak',gib(max(v['resources']['worker']['cuda']['peak_reserved_bytes'] for v in results))+' GiB'],
 ['Lowest available physical RAM in final request snapshots',gib(min(v['resources']['cpu_detector']['system_memory']['available_bytes'] for v in results))+' GiB']])
add('The combined-current measurement is not a synchronized peak monitor: CPU and worker snapshots are taken close together after generation/detection. Individual process peaks need not occur simultaneously and are not added and described as a measured combined peak. CUDA allocator figures exclude some driver/context memory. Worker status contains the latest load/request snapshot rather than a continuous GPU probe. No CUDA OOM occurred; an actual OOM was not deliberately induced.')
add('The initial worker refused startup when RAM was below its 2.75 GiB guard. After the user closed unused applications, loading succeeded without relaxing that guard or changing methodology. Both processes remained loaded through sequential requests. No CPU or synthetic fallback was used. Startup refusal evidence: `session-20261018T100143-fec457ee/startup-error.json`. Later development/final sessions preserve their own logs and manifests.')
add('Generation timing covers only CUDA token generation, with synchronization. Token-audit time covers replay/full-sequence verification and decoding. TRACE time includes CPU behavioral extraction, generation feature calculation/scaling, F0 plus B0/G0 comparison scoring. End-to-end time includes IPC and those stages but excludes browser rendering, initial model load and the final record-file fsync. These are small-sample observed request timings, not a latency benchmark or real-time service guarantee.')
add('## 5. Both views and score consistency')
delta=max(abs(v['live_score']-v['saved_development_score']) for v in compatibility['records'])
add(f'All eight compatibility records came from **model-selection development**, not held-out tests. Live F0 and the established offline single-record path matched within **1e-12**; comparison against saved development batch scores used the prospectively coded **1e-6** numerical tolerance, with maximum observed absolute difference **{delta:.12g}**. All eight decisions agreed. No score tolerance changes, threshold adjustment, new model selection or fitting followed these checks.')
add('The real HTTP checks verify the 64-dimensional behavioral representation, 67-dimensional fused vector, saved threshold, mathematical summary features, aligned IDs, finite outputs and persistent PID. The browser compares its displayed response and rounded F0 score against the actual saved live record. Boundary FP32 effects remain possible for exact threshold ties; the original comparator and scores are preserved.')
add('## 6. Ground truth, safeguards and storage')
add('All live records carry `activation_label: null`, `ground_truth: unknown`, and false training/validation/final-study eligibility flags. They have a separate `live_inference.v1` schema and no experimental split. Existing dataset validation rejects them as training records. Requests accept only `{ "prompt": "…" }`: caller-provided labels, criteria, checkpoints or generation settings are rejected. Trigger text is never used to manufacture a live ground-truth label.')
add('`controlled-output-review.json` separately applies the unchanged independent literal verifier to the authored smoke outputs. It confirms target-output matches for the two cue conditions and repeated A, while F0 predicts Normal. This review is not passed into inference or inserted into live interaction labels. It diagnoses controlled target matches missed by F0; arbitrary user requests still have unknown activation truth. No general ordinary-answer correctness verifier is part of this interface.')
table(['Safeguard','Behavior'],[
 ['Local access','IPv4 loopback binding; localhost Host allowlist and same-origin POST checks'],
 ['Concurrency','One complete generation/detection request at a time; concurrent requests receive HTTP 429'],
 ['Input/output bounds','2000 prompt characters, 256 native-template input tokens, fixed greedy 32-token output ceiling; no silent truncation'],
 ['Resources','Startup requires 2.75 GiB available RAM and free VRAM; request guard refuses below 512 MiB available RAM'],
 ['Failures','Startup/inference/CUDA OOM errors are explicit; 180-second worker/request deadline; no fabricated fallback'],
 ['Retry','Explicit retry only after worker failure, serialized against other requests; previous attempt files retained'],
 ['Identity','Strict existing M5B bundle loading plus worker checkpoint/template checks; detector/bundle hashes rechecked per live request'],
 ['Storage','Unique local session and request IDs under artifacts/trace-live; successful records and errors saved separately; no writes into frozen run directories'],
 ['Training','No automatic retraining, threshold fitting, checkpoint changes or ground-truth assignment']])
add('Private stdio communication needs no worker HTTP port. The live HTTP endpoint is the only prompt submission route. Startup and request logs remain local. Closing the server closes its worker input and stops its owned worker; the launcher avoids starting a duplicate service. Retry is for availability recovery, not model/hyperparameter selection.')
add('## 7. Launch and use')
add('```powershell\ncd C:\\Users\\gil\\Desktop\\TRACE\npowershell -ExecutionPolicy Bypass -File .\\start-live.ps1\n```')
add('Open **http://127.0.0.1:8768**. Wait for “Model ready”, enter a prompt, and select **Run Analysis**. Review the actual response, F0 score/status, raw generation statistics and separate timings. Export saves the full live result. If startup reports insufficient memory, close unused applications and use **Retry model startup**. An existing server is reused only when its F0 artifact identity matches. Stop a foreground launch with Ctrl+C.')
add('Direct command: `.\\.venv\\Scripts\\python.exe -m traceguard serve --run artifacts/trace-robustness/run-001 --data artifacts/trace-robustness/run-001/dashboard.jsonl --port 8768 --live`. The recorded M5B dashboard remains at 8767, M4 at 8766, synthetic demonstration at 8765. The live page also retains all recorded M5B controls/tables below its dedicated panel. No synthetic mode is substituted on port 8768.')
add('## 8. Tests, screenshots and preservation')
table(['Executed check','Result / evidence under artifacts/trace-live'],[
 ['Final unit suite','97 run, 95 passed, two expected PEFT skips; unit-tests-final.log'],
 ['Real HTTP integration','Passed four final successful requests, deterministic repeat, same worker, independent token/feature checks; real-integration-final.json/.log'],
 ['Input/security/concurrency','Empty, oversized, template-token overflow and forged-label requests rejected; external Origin/Host rejected; concurrent request 429; ready-worker retry 409'],
 ['Browser','Passed real new prompt → actual response/features/score, loading state, export enabled, invalid-input handling, mobile layout and old dashboards; browser-evidence-final.json, browser-tests-final.log'],
 ['Long-input GPU probe','Passed 210-input/14-output-token request; resource-probe.json'],
 ['Development consistency','Eight records passed; session development-consistency.json; no held-out tuning'],
 ['Artifact and dependency audit',f'Passed {preservation["old_files_checked"]} prior files; {preservation["old_files_unchanged"]} unchanged, five intentional runtime/current-doc changes, zero unexpected changes; preservation-final.json']])
add('Both frozen M4/M5B bundles and the original clean weights remain intact. All three environment package inventories match pre-existing pins: `.venv` 41 packages, `.venv-gpu` 27, `.venv-train` 28. No dependencies were installed/upgraded. Old protocols, reports, thresholds, failed adapters and synthetic artifacts remain unchanged. Only server/CLI and three current context documents were intentionally changed among previously tracked files.')
add('Desktop/mobile screenshots were inspected visually: [desktop](../artifacts/trace-live/live-desktop-final.png), [mobile input](../artifacts/trace-live/live-mobile-final.png), [mobile result](../artifacts/trace-live/live-mobile-result-final.png). The screenshots show an actual generated evaporation response and its saved F0 analysis. They are not mockups.')
add('Re-run read-only/unit checks with `.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v`. Real integration scripts generate additional isolated live records; use a new evidence path with `tests/smoke_live_http.py --output <new-path.json>`. Do not rerun the 5B fit/collection workflow. Audit/report scripts refuse overwriting existing evidence.')
add('## 9. Remaining limitations')
add('1. F0 missed the independently reviewed controlled cue outputs on this live prompt style. The fixed M5B results do not guarantee live generalization. Input wording, generation features and learned representation can shift; this milestone did not isolate a cause or tune a fix.\n2. The adapter remains non-accepted for final-study ordinary-response retention. TinyLlama can give incorrect answers and stop at the output token limit.\n3. Risk scores are uncalibrated; “Normal” does not certify safety and “Suspicious” does not prove activation. New live prompts have no automatic ground truth.\n4. One model, one frozen detector and a small engineering smoke set do not establish thesis-level effectiveness, universal trigger detection, cross-model generalization or baseline superiority.\n5. This is a localhost single-user prototype, with bounded serial requests and a small memory margin under competing applications. It is not a multi-user production service.\n6. CUDA OOM/error paths are implemented but a physical OOM was not deliberately induced. The ready-worker retry rejection was tested; successful recovery after the low-RAM refusal used a controlled service restart, rather than separately exercising the failed-worker retry button. Browser testing checks export availability, not the operating system’s saved download contents. The 210-token probe does not exhaustively stress every allowed prompt.\n7. Live JSON files contain user prompt/response text locally and are not automatically curated, labeled, deduplicated or admitted into future datasets.\n8. Model loading occurs before service readiness; free memory and cached checkpoints are required. If a startup fails, its evidence is retained and a deliberate retry is required.')
add('## 10. Defense readiness and next action')
add('**Ready for a bounded engineering demonstration:** submit a real prompt, show persistent GPU generation, inspect independently audited token statistics, both frozen views, F0 concatenation/classification, unchanged threshold and exported unknown-label record. Show the distinction between live inference, recorded real evaluation and synthetic examples. Explain the controlled misses openly; do not present the application as a validated universal defense.')
add('Recommended next action: review this application and its observed failures before any new experiment. A separate prospective robustness study should address varied prompt styles, richer target behaviors/controls and target-checkpoint retention qualification using fresh grouped development/test sources. Do not tune against these observed smoke outputs or retroactively alter M4/M5B results. No such study, detector retraining or LLM fine-tuning was started in Milestone 6.')
with output.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
print('Created',output,'from completed actual live evidence.')
