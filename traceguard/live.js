(() => {
  const node = id => document.getElementById(id);
  let running = false, last = null;
  const number = n => Number(n).toFixed(6);
  async function refresh() {
    try {
      const response = await fetch('/api/live/status');
      const status = await response.json();
      if (!response.ok) throw Error(status.error);
      if (!running) node('liveState').textContent = status.state === 'ready' ? 'TinyLlama ready · persistent local GPU worker' : status.error || 'Loading TinyLlama…';
      node('liveRun').disabled = running || status.state !== 'ready' || status.busy;
      node('liveRetry').hidden = status.state !== 'failed';
    } catch (error) { if (!running) {node('liveState').textContent = 'Live service unavailable: '+error.message;node('liveRun').disabled=true;} }
  }
  node('liveRun').addEventListener('click', async () => {
    running = true; last = null;
    node('liveRun').disabled = true;node('liveExport').disabled=true;
    node('liveResult').hidden = true;node('liveError').textContent = '';
    node('livePrompt').disabled = true;
    node('liveState').textContent = 'Generating a real response, checking token statistics, then running both TRACE views…';
    node('livePanel').setAttribute('aria-busy','true');
    try {
      const response = await fetch('/api/live/analyze', {method:'POST', headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:node('livePrompt').value})});
      const result = await response.json();
      if (!response.ok) throw Error(result.error);
      last=result; const d=result.detection,t=result.timings;
      node('liveResponse').textContent=result.interaction.response || '(Empty decoded response; see exported token IDs.)';
      node('liveScore').textContent=d.risk_score.toFixed(3);
      node('liveScore').style.color=d.prediction ? 'var(--amber)' : 'var(--mint)';
      node('liveDecision').textContent=d.status;
      node('liveThreshold').textContent='Frozen F0 threshold: '+d.threshold.toFixed(10);
      node('liveValidity').textContent=d.score_interpretation+' Ground truth: unknown.';
      node('liveBehavior').textContent='MiniLM prompt + response → trained 64-dimensional projection (L2 norm '+number(d.behavioral_features.l2_norm)+') → concatenated with 3 standardized generation features → frozen logistic regression.';
      node('liveEntropy').textContent=number(d.features.mean_entropy);
      node('liveConfidence').textContent=number(d.features.mean_token_confidence);
      node('liveVariance').textContent=number(d.features.confidence_variance);
      node('liveViews').textContent='Experimental comparison scores — B0: '+number(d.view_scores.B0)+' · G0: '+number(d.view_scores.G0)+'. These do not replace F0.';
      node('liveTiming').textContent='Target generation: '+t.generation_seconds.toFixed(2)+' s · Token audit: '+t.alignment_seconds.toFixed(2)+' s · TRACE detector: '+t.detector_ms.toFixed(2)+' ms · End-to-end: '+t.end_to_end_seconds.toFixed(2)+' s.';
      node('liveResources').textContent='Generated tokens: '+result.interaction.generated_token_ids.length+' · '+result.interaction.generation_configuration.finish_reason+' · Model: '+result.interaction.target_model+' · Live record: '+result.interaction.sample_id;
      node('liveResult').hidden=false;node('liveExport').disabled=false;
    } catch(error) {node('liveError').textContent=error.message+' No synthetic response was substituted.';}
    finally {running=false;node('livePrompt').disabled=false;node('livePanel').setAttribute('aria-busy','false');await refresh();}
  });
  node('liveExport').addEventListener('click',()=>{
    if (!last) return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(last,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='trace-live-'+last.interaction.sample_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  node('liveRetry').addEventListener('click',async()=>{
    node('liveRetry').disabled=true;
    try {
      const response=await fetch('/api/live/retry',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
      const result=await response.json();if(!response.ok)throw Error(result.error);
      node('liveError').textContent='';await refresh();
    } catch(error) {node('liveError').textContent=error.message;}
    finally {node('liveRetry').disabled=false;}
  });
  refresh();setInterval(refresh,2000);
})();
