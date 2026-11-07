(() => {
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  let state = null, running = false, renderKey = '', visibleResults = 100;
  let batch = null, refreshing = false;
  let manualConfig = null;
  const fmt = n => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(6) : 'Unavailable';
  const setStatus = (s, error = false) => { $('status').textContent = s; $('status').classList.toggle('error', error); };
  const reviewedTypedPrompt = () => manualConfig?.prompts.find(p=>p.prompt===$('manualTypedPrompt').value);
  function controls() {
    for (const id of ['savePrompts','analyzeRecords','runExamples','promptFile','recordFiles','manualCondition','manualTypedPrompt']) $(id).disabled = running || state?.busy;
    $('manualPrompt').disabled=running||state?.busy||!manualConfig;
    $('manualRun').disabled=running||state?.busy||!reviewedTypedPrompt();
    $('manualSavePrompt').disabled=running||state?.busy||!$('manualTypedPrompt').value.trim()||!!reviewedTypedPrompt();
  }
  function previewManual(){
    if(!manualConfig)return;
    const present=Number($('manualCondition').value)%2===0;
    $('manualCue').textContent=present?'Present: '+manualConfig.cue_value+' (suffix)':'Absent';
    const prompt=$('manualTypedPrompt').value;
    $('manualInput').textContent=prompt+(present?'\n\n'+manualConfig.cue_value:'');
    $('manualPromptStatus').textContent=reviewedTypedPrompt()?'Reviewed prompt: ready to run.':'Custom prompt: save for review. It will not be submitted to inference.';
    controls();
  }
  async function api(path, body) {
    const r = await fetch('/api/research/' + path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const v = await r.json(); if (!r.ok) throw Error(v.error || 'Request failed.'); return v;
  }
  function download(name, content, type) {
    const url = URL.createObjectURL(new Blob([content], {type})); const a = el('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function resultCard(r) {
    const card = el('article',undefined,'result'); card.dataset.resultId=r.result_id;
    const c=r.condition;
    if(c)card.append(el('p',`Condition ${c.condition_id ?? 'unknown'} | MTBA ${c.adapter_status} | Trigger ${c.trigger_condition}`,'badge'));
    const sourceLabel=r.origin==='manual_harmless_condition'?'Manual run':r.origin==='reviewed_harmless_batch'?'Collected batch':r.origin==='fixed_harmless_demo'?'Live harmless check':'Imported record';
    const head=el('div',undefined,'result-title');head.append(el('strong',r.result_id),el('span',sourceLabel,'badge'));card.append(head);
    const meta=el('p',r.checkpoint_role+' · '+(r.interaction.chat_context?'Chat · unvalidated':'Independent interaction'),'muted');card.append(meta);
    const prompt=el('details');prompt.append(el('summary','Input and context'),el('pre',r.interaction.prompt));card.append(prompt);
    if(r.interaction.research_cue?.value)card.append(el('p','Cue: '+r.interaction.research_cue.value,'muted'));
    card.append(el('p',r.interaction.target_model+' | '+(r.checkpoint_role==='clean_reference'?'MTBA disabled':'MTBA enabled'),'muted'));
    card.append(el('div',r.interaction.response || '(No decoded response text.)','response'));
    const features=el('div',undefined,'features');
    for(const [label,value] of [['Mean entropy',r.generation_features.mean_entropy],['Mean token confidence',r.generation_features.mean_token_confidence],['Confidence variance',r.generation_features.confidence_variance],['TRACE risk',r.detection.available?r.detection.risk_score:null]]){
      const f=el('div',undefined,'feature');f.append(el('small',label),el('strong',fmt(value)));features.append(f);
    }card.append(features);
    const detailGrid=el('div',undefined,'detail-grid');const b=r.behavioral_features;
    const behavioral=el('details');behavioral.append(el('summary',b.available?'Behavioral view · '+b.dimensions+' dimensions':'Behavioral view unavailable'));
    behavioral.append(el('p',b.available?b.model+' · '+b.representation+' · L2 norm '+fmt(b.l2_norm):b.reason,'muted'));
    if(b.available)behavioral.append(el('pre',JSON.stringify(b.vector,null,2)));
    const generation=el('details');generation.append(el('summary','Generation view · aligned token signals'));
    generation.append(el('pre',JSON.stringify({generated_token_ids:r.interaction.generated_token_ids,token_confidences:r.interaction.token_confidences,token_entropies:r.interaction.token_entropies},null,2)));
    detailGrid.append(behavioral,generation);card.append(detailGrid);
    if(!r.detection.available)card.append(el('p',r.detection.reason,'muted'));
    const review=el('div',undefined,'review');const label=el('label','Task success ');const select=el('select');select.setAttribute('aria-label','Task success for '+r.result_id);
    for(const [value,name] of [['','Not reviewed'],['0','0 · Does not fulfill task'],['1','1 · Fulfills task'],['U','U · Uncertain']]){const option=el('option',name);option.value=value;select.append(option);}select.value=r.review.task_success;label.append(select);
    const reviewer=el('input');reviewer.type='text';reviewer.placeholder='Reviewer name';reviewer.maxLength=100;reviewer.value=r.review.reviewer;reviewer.setAttribute('aria-label','Reviewer for '+r.result_id);
    const saved=el('span','','review-status');
    const save=async()=>{select.disabled=true;reviewer.disabled=true;try{await api('review',{result_id:r.result_id,task_success:select.value,reviewer:reviewer.value});r.review={task_success:select.value,reviewer:reviewer.value};const current=state.results.find(x=>x.result_id===r.result_id);if(current)current.review=r.review;saved.textContent='Saved';$('reviewCount').textContent=state.results.filter(x=>x.review.task_success).length;}catch(e){saved.textContent='Not saved';setStatus(e.message,true);}finally{select.disabled=false;reviewer.disabled=false;}};
    select.addEventListener('change',save);reviewer.addEventListener('change',save);review.append(label,reviewer,saved);card.append(review);return card;
  }
  function render() {
    $('promptCount').textContent=state.prompts.length;$('resultCount').textContent=state.results.length;$('reviewCount').textContent=state.results.filter(r=>r.review.task_success).length;
    $('queueSummary').textContent='· '+state.prompts.length+' questions';
    const key=state.workspace_id+':'+state.prompts.length+':'+state.results.length+':'+visibleResults;
    if(key!==renderKey){
      renderKey=key;const fragment=document.createDocumentFragment();
      for(const p of state.prompts){const tr=el('tr');for(const value of [p.prompt_id,p.source+' / '+p.split,p.category,p.prompt,p.context])tr.append(el('td',value,'prompt'));fragment.append(tr);}$('questionRows').replaceChildren(fragment);
      $('results').replaceChildren(...(state.results.length?state.results.slice(0,visibleResults).map(resultCard):[el('p','No saved responses yet. Choose a condition and prompt, then click Run once.','empty')]));
    }
    $('showMore').hidden=state.results.length<=visibleResults;
    if(!$('examples').children.length)for(const example of state.examples){const label=el('label',undefined,'example');const input=el('input');input.type='checkbox';input.value=example.id;input.checked=true;label.append(input,document.createTextNode(' '+example.prompt));$('examples').append(label);}
    controls();
  }
  async function refresh(){
    if(refreshing)return;refreshing=true;
    try{
    state=await api('state');render();
    const info=await api('export-info');
    $('questionExportSummary').textContent=info.filename+' · '+info.questions+' questions · '+info.completed_interactions+' completed interactions · '+info.pending_questions+' pending'+(info.unmatched_records?' · '+info.unmatched_records+' saved records do not match this batch':'');
    $('exportQuestions').disabled=!info.questions;
    batch=await api('batch');const job=batch.job,c=job?.counts;
    if(!manualConfig){
      manualConfig=await api('manual');$('manualPrompt').replaceChildren(...manualConfig.prompts.map(p=>{const option=el('option',p.prompt);option.value=p.index;return option;}));
      if(!$('manualTypedPrompt').value)$('manualTypedPrompt').value=manualConfig.prompts[0].prompt;
      previewManual();
    }
    $('batchProgress').textContent=job?`${job.filename}: ${job.status} | ${c.completed} completed, ${c.pending} pending, ${c.generating} generating, ${c.failed} failed, ${c.uncertain} uncertain`:'No collection batch started.';
    if(job){const failed=job.items.filter(i=>i.error);if(failed.length)$('batchProgress').textContent+=' | '+failed.map(i=>i.source.prompt_id+': '+i.error).join(' | ');}
    $('batchStart').disabled=!!state.busy||!!c?.pending;
    $('batchStop').disabled=!job||job.status!=='running';
    $('batchResume').disabled=!!state.busy||!c?.pending;
    controls();
    }finally{refreshing=false;}
  }
  async function operation(fn){if(running)return;running=true;controls();$('batchErrors').hidden=true;try{await fn();await refresh();}catch(e){setStatus(e.message,true);}finally{running=false;controls();}}
  function report(v){setStatus((v.added??0)+' records added'+(v.duplicates?', '+v.duplicates+' already present':'')+'.');if(v.errors?.length){$('batchErrors').hidden=false;$('batchErrors').textContent=v.errors.map(e=>(e.index?'Record '+e.index:e.example_id)+': '+e.error).join('\n');}}
  function parseRecords(raw){
    let value;try{value=JSON.parse(raw);}catch{value=raw.split(/\r?\n/).filter(s=>s.trim()).map((s,i)=>{try{return JSON.parse(s);}catch{throw Error('Invalid JSONL on line '+(i+1));}});}
    if(Array.isArray(value))return value;
    if(value?.schema==='trace.research_workspace.v1'&&Array.isArray(value.results))return value.results.map(r=>r.original_import || r.interaction);
    if(Array.isArray(value?.records))return value.records;
    return [value];
  }
  $('savePrompts').addEventListener('click',()=>operation(async()=>{const v=await api('prompts',{text:$('promptText').value});setStatus(v.total+' questions saved. No generation started.');$('promptText').value='';}));
  $('promptFile').addEventListener('change',()=>operation(async()=>{const f=$('promptFile').files[0];if(!f)return;if(f.size>5000000)throw Error('Question CSV must be at most 5 MB.');const v=await api('prompts',{csv:await f.text(),filename:f.name});setStatus(v.batch_questions+' questions loaded for CSV output; '+v.total+' saved in the workspace. No generation started.');$('promptFile').value='';}));
  $('analyzeRecords').addEventListener('click',()=>operation(async()=>{setStatus('Extracting signals from imported records…');report(await api('import',{records:parseRecords($('recordText').value)}));}));
  $('recordFiles').addEventListener('change',()=>operation(async()=>{const files=[...$('recordFiles').files];if(files.reduce((n,f)=>n+f.size,0)>18000000)throw Error('Import up to 18 MB per batch.');const records=[];for(const f of files)records.push(...parseRecords(await f.text()));if(!records.length)return;setStatus('Analyzing '+records.length+' saved records…');report(await api('import',{records}));$('recordFiles').value='';}));
  $('runExamples').addEventListener('click',()=>operation(async()=>{const ids=[...$('examples').querySelectorAll('input:checked')].map(n=>n.value);setStatus('Running selected fixed harmless examples…');report(await api('demo',{example_ids:ids}));}));
  $('refresh').addEventListener('click',()=>refresh().catch(e=>setStatus(e.message,true)));
  $('manualCondition').addEventListener('change',previewManual);
  $('manualPrompt').addEventListener('change',()=>{$('manualTypedPrompt').value=manualConfig.prompts.find(p=>p.index===Number($('manualPrompt').value)).prompt;previewManual();});
  $('manualTypedPrompt').addEventListener('input',previewManual);
  $('manualSavePrompt').addEventListener('click',()=>operation(async()=>{
    const prompt=$('manualTypedPrompt').value;
    if(!prompt.trim())throw Error('Type a prompt first.');
    const cell=value=>'"'+String(value).replaceAll('"','""')+'"';
    const csv=[['Prompt-ID','Source','Prompt','Category','Split','Source-Flags'],['','Manual draft',prompt,'unassigned','unassigned','pending_manual_review']].map(row=>row.map(cell).join(',')).join('\r\n')+'\r\n';
    await api('prompts',{csv,filename:'Custom prompt pending review'});
    setStatus('Custom prompt saved for review. No model request was made.');
  }));
  $('manualRun').addEventListener('click',()=>operation(async()=>{
    const condition=Number($('manualCondition').value);
    const reviewed=reviewedTypedPrompt();
    if(!reviewed)throw Error('Save this custom prompt for review before generation.');
    setStatus('Running condition '+condition+'...');
    const result=await api('manual',{condition_id:condition,prompt_index:reviewed.index});
    setStatus('Condition '+result.condition_id+' completed and saved.');
  }));
  for(const [id,action] of [['batchStart','start'],['batchStop','stop'],['batchResume','resume']]){
    $(id).addEventListener('click',()=>operation(async()=>{await api('batch',{action});setStatus(action==='stop'?'Stop requested; current response will finish.':'Harmless collection '+(action==='start'?'started.':'resumed.'));}));
  }
  $('batchExample').addEventListener('click',()=>{
    if(!batch)return;const cell=s=>'"'+String(s).replaceAll('"','""')+'"';
    const rows=batch.approved_prompts.map((prompt,i)=>['TRACE-SAFE-'+String(i+1).padStart(3,'0'),'Reviewed harmless catalog',prompt,'benign','pilot']);
    download('TRACE-harmless-batch.csv','\ufeff'+[['Prompt-ID','Source','Prompt','Category','Split'],...rows].map(row=>row.map(cell).join(',')).join('\r\n')+'\r\n','text/csv;charset=utf-8');
  });
  $('showMore').addEventListener('click',()=>{visibleResults+=100;render();});
  $('exportJson').addEventListener('click',()=>{if(state)download('TRACE-research-records.json',JSON.stringify(state,null,2),'application/json');});
  $('exportQuestions').addEventListener('click',async()=>{
    $('exportQuestions').disabled=true;
    try{const response=await fetch('/api/research/questions.csv');if(!response.ok)throw Error('Could not export question results.');download('TRACE-questions-results.csv',await response.blob(),'text/csv;charset=utf-8');setStatus('Question results CSV downloaded. Pending rows have no invented responses or signals.');}
    catch(e){setStatus(e.message,true);}finally{$('exportQuestions').disabled=false;}
  });
  $('exportSignals').addEventListener('click',()=>{
    if(!state)return;const columns=['result_id','prompt','response','checkpoint_role','mean_entropy','mean_token_confidence','confidence_variance','behavioral_dimensions','behavioral_representation','behavioral_vector','risk_score','activation_label','training_eligible','condition_id','trigger_condition','adapter_status','condition_provenance','model','model_revision','adapter','generation_configuration','cue_value','source_prompt_id'];
    const cell=v=>{let s=String(v??'');if(typeof v==='string'&&/^[\s]*[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
    const rows=state.results.map(r=>[r.result_id,r.interaction.prompt,r.interaction.response,r.checkpoint_role,r.generation_features.mean_entropy,r.generation_features.mean_token_confidence,r.generation_features.confidence_variance,r.behavioral_features.dimensions,r.behavioral_features.representation,r.behavioral_features.vector?JSON.stringify(r.behavioral_features.vector):'',r.detection.available?r.detection.risk_score:'','',false,r.condition?.condition_id??'',r.condition?.trigger_condition??'unknown',r.condition?.adapter_status??'',r.condition?.condition_provenance??'',r.interaction.target_model,r.interaction.checkpoint_identity.revision,r.interaction.checkpoint_identity.adapter??'disabled',JSON.stringify(r.interaction.generation_configuration??{}),r.interaction.research_cue?.value??'',r.interaction.source_prompt_id??'']);
    download('TRACE-signals.csv','\ufeff'+[columns,...rows].map(r=>r.map(cell).join(',')).join('\r\n')+'\r\n','text/csv;charset=utf-8');
  });
  $('exportCsv').addEventListener('click',()=>{
    if(!state)return;const columns=['result_id','prompt','response','task_success','reviewer'];
    const cell=v=>{let s=String(v??'');if(/^[\s]*[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};
    const rows=state.results.map(r=>[r.result_id,r.interaction.latest_user_prompt || r.interaction.prompt,r.interaction.response,r.review.task_success,r.review.reviewer]);
    download('TRACE-task-success-review.csv','\ufeff'+[columns,...rows].map(r=>r.map(cell).join(',')).join('\r\n')+'\r\n','text/csv;charset=utf-8');
  });
  refresh().catch(e=>setStatus(e.message,true));
  setInterval(()=>{if(running||state?.busy)refresh().catch(()=>{});},3000);
})();
