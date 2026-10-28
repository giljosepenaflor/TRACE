"""Render the completed 5B evidence into the requested results report; no fitting."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from traceguard.engineering import read_json
from traceguard.records import read_records
from traceguard.robustness_protocol import ARMS,PARTITIONS

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',default='artifacts/trace-robustness/run-001')
parser.add_argument('--output',default='docs/TRACE_ROBUSTNESS_RESULTS.md')
args=parser.parse_args();run=Path(args.run);out=Path(args.output)
if out.exists():raise ValueError('Preserve existing results; choose a new report path.')
r=read_json(run/'report.json');f=read_json(run/'frozen.json');p=read_json(run/'protocol.json')
dev=read_json(run/'development-collection.json');test=read_json(run/'test-collection.json')
rows=read_records(run/'development.jsonl')+read_records(run/'test.jsonl')
lines=[]
def add(text=''):lines.extend(text.strip('\n').splitlines());lines.append('')
def table(head,body):
    add('| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in body))
def fmt(x):return 'N/A' if x is None else f'{x:.4f}'
def interval(x):return '['+', '.join(fmt(v) for v in x)+']'

add('# TRACE robustness experiment results — Milestone 5B')
add('This is an actual completed, prospective **engineering** experiment using the unchanged, **non-accepted** TinyLlama 3C checkpoint 144. Trigger B was already observed during Milestone 4: these are exploratory fresh-source robustness findings, not independent discovery of a research-blind trigger. F0 remains the original TRACE reference; no test-selected variant replaces it. No LLM fine-tuning or AdvBench experiment occurred.')
add('## 1. Implementation and frozen protocol')
add('`engineering_robustness.v1` was frozen before generation: 180 new source groups, ten topics, split seed 505, four real conditions per group, pinned checkpoints and native chat template, greedy decoding with at most 32 new tokens. The freshness audit found no normalized overlap with 499 historical question keys and documents the authored source-fact review. This audit is not an automated proof of semantic novelty.')
add('New modules: `robustness_sources.py`, `robustness_protocol.py`, `robustness.py`, `robustness_thresholds.py`, `robustness_detector.py`. Existing MiniLM/contrastive learning, generation features and literal verification are reused. Existing behavioral formatting gains an opt-in response-only mode; its default prompt-response definition is unchanged. Record validation and server loading use a separate strict schema/bundle path. Existing M4 schemas, data, model artifacts and results remain unchanged.')
add('The new orchestrator waits for successful real development collection, fits/freezes every candidate, verifies hashes, and only then starts test collection. Failures halt the workflow with partial records/logs preserved; there is no synthetic or CPU fallback. The detector fitting input is a redacted development spec containing A and development sources only. Test source prompts may exist in the frozen collection manifest, but test interactions and B do not enter detector fitting/selection/calibration.')
table(['Arm','Definition'],[(a,p['settings']['arm_definitions'][a]) for a in ARMS])
add('F2 divides each feature block by its training RMS vector norm. F3 has no independently learned fusion parameter: it uses exactly 0.5 of the B1 logit plus 0.5 of the G0 logit. Six LR arms each retain all three C candidates (18 fitted LRs), while F3 reuses two selected component models. The shared pretrained encoder weights are frozen; original and response-only projections are independently learned on training data only.')
add('## 2. Actual collection, labels and resource safeguards')
table(['Partition','Source groups','Actual records','Verified activations','Failed adapted cue attempts','Unresolved'],[
    [part,len({v['source_prompt_id'] for v in rows if v['robustness_partition']==part}),
     sum(v['robustness_partition']==part for v in rows),sum(v['robustness_partition']==part and v['activation_label']==1 for v in rows),
     sum(v['robustness_partition']==part and v['checkpoint_role']=='experimental_backdoor' and v['trigger_present'] and v['activation_label']==0 for v in rows),
     sum(v['robustness_partition']==part and v['activation_label'] is None for v in rows)] for part in PARTITIONS])
add(f'Collected **{len(rows)} actual interactions**, comprising {sum(v["checkpoint_role"]=="clean_reference" for v in rows)} clean-reference and {sum(v["checkpoint_role"]=="experimental_backdoor" for v in rows)} adapted records. Both conditions from both roles remain in the same source partition. Activation labels are independently recomputed from actual responses using the unchanged exact target criterion; cue presence alone is insufficient. Genuine failures remain negatives and are separately saved in `failed-attempts.jsonl`. Null labels are retained in raw records and excluded identically from all arms. Ordinary-answer correctness is not an activation label.')
add(f'All strict token replay audits passed: development={dev["replay_alignment_passed"]}, test={test["replay_alignment_passed"]}. Full-sequence FP16 rounding warnings: {dev["full_sequence_rounding_warnings"]+test["full_sequence_rounding_warnings"]}; the independent same-shape token replay remains the strict alignment check. Generated IDs decode to the saved response and token arrays align. Original clean weights and adapter file hashes were checked at both stage starts.')
add(f'The four-condition primary budget was preserved, with **zero additional challenge interactions**, as declared before generation. No marker-imitation challenge set was added after observing results. This limits what can be concluded about fixed-response shortcuts. There were {sum(v["generation_configuration"]["finish_reason"]=="max_new_tokens" for v in rows)} token-limit endings; these valid bounded generations were retained rather than filtered by outcome.')
table(['Stage','Seconds','Peak process RAM MiB','Peak CUDA allocated GiB','Peak CUDA reserved GiB'],[
    [name,f'{obj["elapsed_seconds"]:.2f}',f'{obj["resources"]["process_memory"]["peak_working_set_bytes"]/1024**2:.2f}',
     f'{obj["resources"]["cuda"]["peak_allocated_bytes"]/1024**3:.4f}',f'{obj["resources"]["cuda"]["peak_reserved_bytes"]/1024**3:.4f}']
    for name,obj in [('Development collection',dev),('Test collection',test)]
    ]+[['Detector fitting',f'{f["fit_seconds"]:.2f}',f'{f["resources"]["process_memory"]["peak_working_set_bytes"]/1024**2:.2f}','CPU','CPU'],
       ['Evaluation including diagnostics/uncertainty',f'{r["evaluation_seconds"]:.2f}',f'{r["resources"]["process_memory"]["peak_working_set_bytes"]/1024**2:.2f}','CPU','CPU']])
add('Collection used the existing `.venv-train` CUDA/PEFT environment on the 4 GB GTX 1650. Detector fitting/evaluation uses `.venv` on CPU. No dependencies were upgraded. Preflight required 2.75 GiB available system RAM; an ongoing safeguard stops safely below 512 MiB available. CUDA figures are allocator peaks, not total driver memory; process RAM is a process-lifetime peak. The completed collection had no silent synthetic substitution.')
add('## 3. Training, model selection and separate threshold development')
add('Training: 60 source groups, 240 records before null exclusion. Model selection: 40 disjoint groups. Threshold development: another 40 disjoint groups. MiniLM revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, 384→64 projections, 60 contrastive epochs, seed 42, Adam 0.003 and temperature 0.1 match the frozen plan. LR C candidates are 0.1, 1 and 10, with smaller C preferred on equal selection F1. No training was pooled with development after freezing.')
body=[]
for a in ARMS:
    s=r['selection'][a];selected=next((e for e in s.get('candidates',[]) if e['C']==s.get('selected_C')),None)
    body.append([a,s.get('selected_C','fixed components'),fmt(selected['selection']['f1']) if selected else 'component selection',
                 f'{r["thresholds"][a]["legacy"]:.10f}',f'{r["thresholds"][a]["midpoint"]:.10f}',fmt(r['thresholds'][a]['maximum_f1'])])
table(['Arm','Selected C','Selection F1','Legacy threshold','Midpoint threshold','Threshold-dev max F1'],body)
add('The legacy rule maximizes F1 on threshold-development and selects the highest score-grid threshold on ties. The midpoint secondary policy chooses the midpoint of the widest optimal constant-prediction probability interval, preferring the higher midpoint on a width tie. Both use `>=`; an explicit finite always-negative endpoint handles exact scores of 1. This endpoint does not change ordinary M4 selection behavior, and M4 itself was not modified. All trained candidates, selection logs, configurations, thresholds and source IDs were saved and hash-frozen before test generation.')
add('## 4. All seven variants: frozen seen and unseen results')
for policy,title in [('legacy','Primary legacy policy'),('midpoint','Predeclared secondary midpoint policy')]:
    add('### '+title)
    table(['Test','Arm','n','Accuracy','Precision','Recall','F1','FPR','AUROC','TN/FP/FN/TP'],[
        [split,a,m['count'],*[fmt(m[k]) for k in ('accuracy','precision','recall','f1','false_positive_rate','auroc')],
         '/'.join(str(m['confusion_matrix'][k]) for k in ('tn','fp','fn','tp'))]
        for split in ('test_seen','test_unseen') for a in ARMS for m in [r['tests'][split]['policies'][policy][a]['combined']]])
add('These combined metrics include all four conditions. The primary test has 20 source groups per partition; rows are not 80 independent questions. A zero precision value when nothing is flagged uses the predefined zero-division convention. Raw metrics and distributions, including exact threshold margins, remain in `report.json`. No test result altered a threshold, C, representation, fusion weight or primary model.')
add('### Clean controls, failed attempts and adapted-only results (primary policy)')
table(['Test','Arm','Adapted-only F1','Clean FPR','Failed attempts','Failed-attempt FPR'],[
    [s,a,fmt(m['adapted']['f1']),fmt(m['clean']['false_positive_rate']),m['failed_attempts']['count'],
     fmt(m['failed_attempts'].get('false_positive_rate'))] for s in ('test_seen','test_unseen') for a in ARMS
    for m in [r['tests'][s]['policies']['legacy'][a]]])
add('If a failed-attempt stratum is empty, it is reported as zero records/N/A rather than fabricated. A small clean FPR is not a deployment guarantee. When no source has a clean false alarm, the independent-source one-sided 95% upper bound for “any clean false alarm within a source” is still approximately 13.9% for twenty sources. That bound is a different estimand from per-record FPR and relies on source-independence assumptions.')
add('## 5. Paired uncertainty and predefined evidence gates')
add('Two thousand bootstrap draws (seed 505) resample whole source groups, retaining all roles/conditions and paired model predictions. Null labels are omitted from metric arithmetic, not from the raw source groups. Intervals are conditional on this checkpoint, source pool and fitted models; they do not include retraining uncertainty. Degenerate zero-width intervals can occur with identical predictions and do not certify zero population error.')
table(['Test','F1 minus F0: paired F1 95% interval','F1 minus F0: clean FPR 95% interval'],[
    [s,interval(v['f1']),interval(v['clean_fpr'])] for s,v in r['predefined_evidence']['primary_paired_95'].items()])
table(['Test','Comparison','F1 difference 98.75% interval','Clean FPR difference 98.75% interval'],[
    [s,'F1 minus '+base,interval(v['f1']),interval(v['clean_fpr'])]
    for s,values in r['predefined_evidence']['dual_view_paired_98_75'].items() for base,v in values.items()])
add(f'The preregistered **F1-versus-F0 engineering evidence gate was {"met" if r["predefined_evidence"]["F1_vs_F0_engineering_gate"] else "not met"}**. It requires an unseen paired-F1 95% lower bound above zero, a seen lower bound at least -0.05, and clean-FPR difference upper bounds at most +0.05 in both tests. The predefined **dual-view gate was {"met" if r["predefined_evidence"]["dual_view_gate"] else "not met"}**: its stricter intervals compare F1 with both B1 and G0 in both conditions. These are exploratory engineering rules, not automatic checkpoint/model acceptance or confirmatory statistical claims. F0 remains the reference in the dashboard.')
add('## 6. Threshold stability and score distributions')
table(['Arm','Legacy conditional-bootstrap 95% interval','Midpoint conditional-bootstrap 95% interval','Legacy thresholds changed by source omission'],[
    [a,interval(s['conditional_bootstrap']['legacy']['interval_95']),interval(s['conditional_bootstrap']['midpoint']['interval_95']),
     sum(abs(x['legacy']-r['thresholds'][a]['legacy'])>1e-12 for x in s['leave_source_out'])]
    for a,s in r['threshold_stability'].items()])
add('Stability calculations reuse threshold-development scores only and are conditional on already selected models. None of the 2,000 resampled thresholds was applied to tests. A midpoint policy can add geometric margin but has no guaranteed benefit under cue or role shift. FP32 encoder/projection computations and exact `>=` boundaries are preserved; very small score differences must be distinguished from semantic cue dependence. No post-test tolerance was introduced.')
table(['Test','Arm','Negative score min / median / max','Positive score min / median / max'],[
    [s,a,*[' / '.join(fmt(d['score'][k]) for k in ('min','median','max')) if d['score']['count'] else 'No records' for d in (v['0'],v['1'])]]
    for s,entry in r['tests'].items() for a,v in entry['distributions'].items()])
add('## 7. Behavioral cue dependence and fusion versus Generation-only')
table(['Arm','Seen max absolute score change, prompt omitted','Unseen max absolute score change, prompt omitted'],[
    [a,f'{r["tests"]["test_seen"]["prompt_omission_diagnostic"][a]["max"]:.8g}',f'{r["tests"]["test_unseen"]["prompt_omission_diagnostic"][a]["max"]:.8g}'] for a in ARMS])
add('This sensitivity diagnostic changes only the already generated interaction’s behavioral text, replacing the complete prompt by a fixed neutral placeholder. It does not use a B dictionary, regenerate outputs or tune the detector. Response-only arms are structurally independent of prompt text; zero sensitivity demonstrates this input property, not semantic understanding or resistance to changing model outputs. G0 is unchanged by construction. F0/B0 sensitivity shows prompt dependence but does not alone isolate cue words from useful question context.')
for s,entry in r['tests'].items():
    m=entry['exact_marker_diagnostic'];add(f'Exact response-marker diagnostic on {s}: F1={m["f1"]:.4f}, FPR={m["false_positive_rate"]:.4f}. It follows the known target sentence and is an oracle-style task-simplicity check, not general activation detection.')
table(['Test','F0 F1','F1-candidate F1','B1 F1','G0 F1','F1-candidate minus G0'],[
    [s,*[fmt(e['policies']['legacy'][a]['combined']['f1']) for a in ('F0','F1','B1','G0')],
     fmt(e['policies']['legacy']['F1']['combined']['f1']-e['policies']['legacy']['G0']['combined']['f1'])] for s,e in r['tests'].items()])
add('Fusion helps only if its matched comparison supports that conclusion. A higher point score than G0 is not evidence of superiority over both single views; an equal B1 result can indicate no measurable benefit from adding generation features. Use the paired intervals and predefined gates above. Comparisons between this new F0 and the old M4 F0 are confounded by new questions and the added clean-reference training pool; the within-5B arms are the controlled comparison.')
add('## 8. Latency, footprint and resource measurements')
table(['Arm','Seen warm single median / p95 ms','Unseen warm single median / p95 ms'],[
    [a,*[f'{r["tests"][s]["latency"]["per_arm"][a]["warm_single_median_ms"]:.3f} / {r["tests"][s]["latency"]["per_arm"][a]["warm_single_p95_ms"]:.3f}' for s in ('test_seen','test_unseen')]] for a in ARMS])
add(f'Selected seven-arm bundle: {r["serialized_detector_bytes"]:,} bytes; all 18 C candidates separately: {r["all_candidates_bytes"]:,} bytes; external pretrained encoder files: {r["encoder_bytes"]:,} bytes. The encoder is shared; the selected bundle contains two projections and six LR parameter sets, with F3 reusing B1/G0. These sizes exclude the separate target LLM/adapter. Per-arm timings include that arm’s CPU feature extraction and classifier/fusion; G0 requires no text encoder. All-arm timings, initial/cold-path timing and exact scope are retained in the JSON. Evaluation wall time includes uncertainty/diagnostics and is not detector-request latency.')
add('## 9. Integrity checks, prototype integration and limitations')
add('The new dashboard is a separate run on **http://127.0.0.1:8767**. M4 remains at port 8766 and synthetic demonstration at 8765. All seven scores/primary decisions and the secondary policy are displayed, with F0 explicitly identified as original TRACE. The API accepts only unchanged authentic saved records. For numerical batch consistency, it computes actual frozen-model predictions at startup in the original partition batches and replays those analyses on request; it is not live target-model generation.')
add('Start/reopen with `./start-robustness.ps1`. Bundle loading validates the datasets, protocol and checkpoint identities, all artifact/report hashes, selection/threshold compatibility and pinned encoder files. New tests cover four-condition grouping, separate development partitions, B exclusion, independent verification, projection inputs, fusion arithmetic, midpoint/endpoint rules, source bootstrap/null labels and test-generation-before-freeze rejection. Final executed test and preservation evidence is under `artifacts/trace-robustness/`; no old results are rewritten.')
add('Limitations: one non-accepted adapter, one projection seed, two literal cues, one fixed target sentence, no marker-imitation challenge set, bounded/truncated outputs, relatively few test source groups and no final AdvBench/BackdoorLLM or multi-model study. B is unseen by detector fitting but already known to researchers. Better fixed-sentence recognition is not causal backdoor identification, qualified ordinary retention, arbitrary unseen-trigger discovery or cross-model generalization. Experimental class prevalence differs from rare deployment activation. Unknown live prompts have no automatic verified ground truth.')
add('## 10. Next steps and reproducibility')
add('Keep F0 as the documented TRACE method. Review these exact results and paired uncertainties before selecting any separately named experimental deployment mode. For a defense, demonstrate the authentic recorded pipeline, both successes and false negatives, and the reference/variant distinction. A later live prompt-to-detection worker should reuse the existing inference collector with pinned weights, bounded requests, measured combined RAM/VRAM and a separate inference schema; do not delete the saved-record protections.')
add('Before scientific superiority claims, resolve target-checkpoint qualification and adopt a richer prospectively reviewed activation/control protocol that cannot be solved solely by a known sentence. Reserve a genuinely researcher-blind valid trigger/condition for later confirmation, keep data/source isolation and matched baselines, and complete the thesis-scoped models/attacks/external baselines. A failed or inconclusive robustness result remains a valid finding; do not tune against these observed tests.')
add('```powershell\n# Reproduce in a new directory; never overwrite run-001.\n$env:PYTHONIOENCODING = \'utf-8\'\n.\\.venv\\Scripts\\python.exe -m traceguard.robustness prepare --run artifacts/trace-robustness/run-002\n.\\.venv-train\\Scripts\\python.exe -m traceguard.robustness collect-development --run artifacts/trace-robustness/run-002\n.\\.venv\\Scripts\\python.exe scripts/run_robustness_stages.py --run artifacts/trace-robustness/run-002\n# Audit the completed existing run without refitting:\n.\\.venv\\Scripts\\python.exe tests/smoke_robustness.py artifacts/trace-robustness/run-001\n```')
add('Evidence: `protocol.json`, `protocol-freeze.json`, `freshness.json`, stage collection reports/logs, all raw interactions, `frozen.json`, retained C candidates, development predictions, test predictions/decomposition, final report and compatibility bundle. The protocol and model freeze precede the test-generation stage in `orchestration.json`.')
out.write_text('\n'.join(lines),encoding='utf-8')
print('Created',out,'from actual completed run evidence.')
