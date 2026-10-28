const el = id => document.getElementById(id);
let session, selected, lastResult;
const fmt = value => typeof value === 'number' ? value.toFixed(4) : '—';

function clearResult() {
  lastResult = null;
  el('export').disabled = true;
  el('score').textContent = '—';
  el('status').textContent = 'Awaiting analysis';
  el('fill').style.width = '0';
  for (const id of ['latency', 'entropy', 'confidence', 'variance', 'behavioral', 'generation', 'behaviorInfo']) el(id).textContent = '—';
  el('error').textContent = '';
  if (el('armPredictions')) el('armPredictions').replaceChildren();
}

function selectRecord() {
  clearResult();
  selected = session.records.find(r => r.sample_id === el('sample').value);
  if (!selected) {
    el('prompt').textContent = 'No records in this partition.';
    el('response').textContent = '—';
    el('metadata').textContent = '';
    el('analyze').disabled = true;
    return;
  }
  el('prompt').textContent = selected.prompt;
  el('response').textContent = selected.response;
  const label = selected.activation_label === null ? 'Unresolved' : selected.activation_label === 1 ? 'Verified activation' : 'Non-activated';
  el('metadata').textContent = `${selected.sample_id} · Source ${selected.source_prompt_id} · ${selected.checkpoint_role ?? 'fixture'} · Trigger ${selected.trigger_present ? selected.trigger_configuration : 'absent'} · Ground truth: ${label}`;
  el('analyze').disabled = false;
}

function filterRecords() {
  el('sample').replaceChildren(...session.records.filter(r => (session.record_purpose === 'engineering_robustness' ? r.robustness_partition : r.split) === el('partition').value).map(r => {
    const option = document.createElement('option');
    option.value = r.sample_id;
    option.textContent = r.sample_id;
    return option;
  }));
  selectRecord();
}

async function analyze(record) {
  clearResult();
  el('analyze').disabled = true;
  el('customAnalyze').disabled = true;
  el('status').textContent = 'Analyzing both views…';
  try {
    const response = await fetch('/api/analyze', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(record)});
    const result = await response.json();
    if (!response.ok) throw Error(result.error);
    lastResult = {interaction_id: record.sample_id, ...result};
    el('score').textContent = result.risk_score.toFixed(3);
    el('status').textContent = result.status;
    const color = result.risk_score >= result.threshold ? 'var(--amber)' : 'var(--mint)';
    el('score').style.color = color;
    el('fill').style.background = color;
    el('fill').style.width = `${result.risk_score * 100}%`;
    el('resultMode').textContent = result.synthetic ? 'Synthetic software demonstration' : 'Real recorded interaction · preliminary engineering pilot';
    el('latency').textContent = result.batch_mean_detector_ms.toFixed(2) + ' ms';
    el('entropy').textContent = fmt(result.features.mean_entropy);
    el('confidence').textContent = fmt(result.features.mean_token_confidence);
    el('variance').textContent = fmt(result.features.confidence_variance);
    for (const view of ['behavioral', 'generation']) {
      const threshold = session.view_thresholds?.[view];
      const decision = typeof threshold === 'number' ? ` · threshold ${fmt(threshold)} · ${result.view_scores[view] >= threshold ? 'Possible activation' : 'Normal'}` : '';
      el(view).textContent = fmt(result.view_scores[view]) + decision;
    }
    const behavior = result.behavioral_features;
    el('behaviorInfo').textContent = behavior ? `${behavior.encoder} · ${behavior.dimensions} dimensions · L2 norm ${fmt(behavior.l2_norm)} · ${behavior.projection_trained ? 'train-only contrastive projection' : 'demo hashing'}` : 'Unavailable';
    el('export').disabled = false;
    if (result.arm_scores) {
      el('resultMode').textContent = record.activation_label === null ? 'Recorded F0 inference; ground truth unresolved'
        : `Recorded F0 inference; ${result.arm_scores.F0.prediction === record.activation_label ? 'matches' : 'does NOT match'} the independently verified label`;
      el('armPredictions').replaceChildren(...Object.entries(result.arm_scores).map(([arm, m]) => metricRow([
        armLabels[arm], fmt(m.score), fmt(m.threshold), fmt(m.midpoint_threshold), m.prediction ? 'Possible activation' : 'Normal'
      ])));
    }
  } catch (error) {
    el('error').textContent = error.message;
    el('status').textContent = 'Analysis unavailable';
  } finally {
    el('analyze').disabled = !selected;
    el('customAnalyze').disabled = false;
  }
}

el('partition').addEventListener('change', filterRecords);
el('sample').addEventListener('change', selectRecord);
el('analyze').addEventListener('click', () => analyze(selected));
el('customAnalyze').addEventListener('click', () => {
  try { analyze(JSON.parse(el('custom').value)); } catch (error) { el('error').textContent = error.message; }
});
el('export').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], {type: 'application/json'}));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'trace-analysis.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function metricRow(values) {
  const tr = document.createElement('tr');
  for (const value of values) {
    const td = document.createElement('td');
    td.textContent = value;
    tr.append(td);
  }
  return tr;
}

const armLabels = {F0:'F0 · Original TRACE (reference)', F1:'F1 · Response-only fusion', F2:'F2 · Block-balanced fusion',
  F3:'F3 · Fixed late-logit fusion', B0:'B0 · Original behavioral', B1:'B1 · Response-only behavioral', G0:'G0 · Generation-only'};

fetch('/api/session').then(response => {
  if (!response.ok) throw Error('Unable to load local detector.');
  return response.json();
}).then(data => {
  session = data;
  const engineering = data.record_purpose === 'engineering_pilot';
  const robustness = data.record_purpose === 'engineering_robustness';
  el('mode').textContent = data.synthetic ? 'SYNTHETIC DEMO · LOCAL' : engineering ? 'REAL ENGINEERING PILOT · LOCAL' : 'RESEARCH DETECTOR · LOCAL';
  el('notice').textContent = data.synthetic
    ? 'Synthetic demonstration only. The text, labels, and token distributions are fabricated. Scores and metrics show that the software runs; they are not evidence of backdoor detection performance.'
    : engineering
      ? 'Real TinyLlama interactions · preliminary engineering demonstration. Checkpoint 144 is NOT accepted: ordinary-response retention remains unresolved. The fixed harmless target sentence can make detection trivial. These results do not establish general backdoor detection.'
      : 'Research detector. Review the experiment configuration and activation criteria before interpreting these scores.';
  el('model').textContent = data.target_model;
  el('count').textContent = data.records.length;
  el('threshold').textContent = fmt(data.threshold);
  el('evaluationTitle').textContent = data.synthetic ? 'Synthetic software check' : 'Frozen test evaluation';
  const rows = [];
  for (const [key, title] of [['test_seen', 'Seen trigger'], ['test_unseen', 'Unseen trigger']]) {
    for (const view of robustness ? Object.keys(armLabels) : engineering ? ['full', 'behavioral', 'generation'] : ['full']) {
      const m = robustness ? data.report.tests[key].policies.legacy[view].combined : data.report.tests[key]?.[view];
      rows.push(metricRow([title, robustness ? armLabels[view] : view, m?.count ?? 0, ...['accuracy', 'precision', 'recall', 'f1', 'false_positive_rate', 'auroc'].map(k => fmt(m?.[k]))]));
    }
  }
  el('metrics').replaceChildren(...rows);
  el('evalNote').textContent = `${data.report.excluded_unresolved_count} unresolved records excluded. All thresholds were chosen on validation and frozen before testing. ${data.synthetic ? 'All figures are synthetic and must not be reported as thesis results.' : 'Scores apply to this preliminary fixed-output experiment only.'}`;
  el('diagnosticsPanel').hidden = !engineering;
  if (engineering) {
    const diagnosticRows = [];
    for (const [split, title] of [['test_seen', 'Seen'], ['test_unseen', 'Unseen']]) {
      const diagnostics = data.report.diagnostics[split];
      for (const name of ['known_cue_presence', 'training_cue_A_presence', 'exact_response_marker']) {
        diagnosticRows.push(metricRow([title, name, fmt(diagnostics[name].f1), fmt(diagnostics[name].false_positive_rate)]));
      }
      for (const view of ['full', 'behavioral', 'generation']) {
        diagnosticRows.push(metricRow([title, `cue masked / ${view}`, fmt(diagnostics.cue_masked[view].f1), fmt(diagnostics.cue_masked[view].false_positive_rate)]));
      }
    }
    el('diagnosticMetrics').replaceChildren(...diagnosticRows);
    el('resourceNote').textContent = ['test_seen', 'test_unseen'].map(split => `${split}: warm CPU batch ${data.report.latency[split].warm_batch_median_ms.toFixed(2)} ms/interaction; warm single ${data.report.latency[split].warm_single_median_ms.toFixed(2)} ms`).join(' · ');
  }
  if (robustness) {
    el('status').closest('section').querySelector('h2').textContent = 'Original TRACE (F0) result';
    el('evaluationTitle').textContent = 'Seven-arm frozen test evaluation';
    if (el('evaluationTitle').nextElementSibling) el('evaluationTitle').nextElementSibling.textContent = 'Primary legacy thresholds; F0 is the reference';
    el('mode').textContent = 'REAL ROBUSTNESS EXPERIMENT · F0 REFERENCE';
    el('notice').textContent = 'Exploratory real TinyLlama robustness experiment. F0 is the original TRACE reference; experimental variants have not replaced it. Checkpoint 144 remains NOT accepted. Trigger B was already observed by researchers. Fixed-response results do not establish general backdoor detection.';
    el('analyze').textContent = 'Show frozen-model analysis';
    el('armsPanel').hidden = false;
    el('robustnessPanel').hidden = false;
    el('partition').replaceChildren(...[['test_seen','Seen-trigger test'],['test_unseen','Unseen-trigger test'],['train','Training'],['model_selection','Model-selection development'],['threshold_selection','Threshold-selection development']].map(([value,title]) => {
      const option = document.createElement('option'); option.value=value;option.textContent=title;return option;
    }));
    const secondary=[], controls=[];
    for (const [split,title] of [['test_seen','Seen'],['test_unseen','Unseen']]) {
      for (const arm of Object.keys(armLabels)) {
        const m=data.report.tests[split].policies.midpoint[arm].combined;
        secondary.push(metricRow([title,armLabels[arm],fmt(m.accuracy),fmt(m.f1),fmt(m.false_positive_rate)]));
        const p=data.report.tests[split].policies.legacy[arm];
        controls.push(metricRow([title,armLabels[arm],fmt(p.clean.false_positive_rate),p.failed_attempts.count,
          p.failed_attempts.count ? fmt(p.failed_attempts.false_positive_rate) : 'No failed attempts',
          fmt(data.report.tests[split].prompt_omission_diagnostic[arm].max)]));
      }
    }
    el('midpointMetrics').replaceChildren(...secondary);
    el('controlMetrics').replaceChildren(...controls);
    el('robustnessNote').textContent = `Prespecified F1-versus-F0 exploratory gate: ${data.report.predefined_evidence.F1_vs_F0_engineering_gate ? 'met' : 'not met'}. Dual-view gate: ${data.report.predefined_evidence.dual_view_gate ? 'met' : 'not met'}. These gates never promote a variant or qualify the checkpoint. Recorded analyses are computed from the frozen models at server startup in their original partition batches.`;
  }
  filterRecords();
}).catch(error => {
  el('notice').textContent = 'The local detector could not be loaded.';
  el('error').textContent = error.message;
});
