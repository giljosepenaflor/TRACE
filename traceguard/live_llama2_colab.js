(() => {
  const node = id => document.getElementById(id);
  const el = (tag, cls, text) => {
    const item = document.createElement(tag);
    if (cls) item.className = cls;
    if (text !== undefined) item.textContent = text;
    return item;
  };
  const icon = name => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS(svg.namespaceURI, "use");
    use.setAttribute("href", "#i-" + name); svg.setAttribute("aria-hidden", "true"); svg.append(use);
    return svg;
  };
  const fmt = (value, digits = 6) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "Unavailable";
  const chatMode = () => node("llama2Mode").value === "chat";
  const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  let running = false, resultCount = 0, history = [], conversationId = crypto.randomUUID();
  let activeResult = null, dialogTrigger = null, closeTimer = null, toastTimer = null;
  let canSend = false;
  const feature = (label, value) => {
    const row = el("div", "feature"); row.append(el("span", "", label), el("strong", "", value)); return row;
  };
  const section = (title, ...children) => {
    const item = el("section", "dialog-section"); item.append(el("h3", "", title), ...children); return item;
  };
  function toast(text) {
    clearTimeout(toastTimer); node("toast").textContent = text; node("toast").hidden = false;
    toastTimer = setTimeout(() => { node("toast").hidden = true; }, 2200);
  }
  function resizeComposer() {
    const input = node("llama2Prompt"); input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
    input.style.overflowY = input.scrollHeight > 160 ? "auto" : "hidden";
  }
  function scrollToTurn(turn) {
    turn.scrollIntoView({behavior: reducedMotion() ? "auto" : "smooth", block: "start"});
  }
  function resetMemory() {
    history = []; conversationId = crypto.randomUUID();
    node("llama2Memory").textContent = chatMode()
      ? "New conversation · Memory stays in this tab" : "Each prompt is independent";
  }
  function closeAnalysis(immediate = false) {
    clearTimeout(closeTimer);
    const dialog = node("analysisDialog"); dialog.classList.remove("is-visible");
    const finish = () => {
      if (dialog.open) dialog.close(); activeResult = null;
      if (dialogTrigger?.isConnected) dialogTrigger.focus({preventScroll: true});
    };
    if (immediate || reducedMotion()) finish(); else closeTimer = setTimeout(finish, 180);
  }
  function openAnalysis(result, index, trigger) {
    clearTimeout(closeTimer); activeResult = result; dialogTrigger = trigger;
    const row = result.interaction, detection = result.detection || {}, stats = row.generation_statistics || {};
    const context = row.chat_context, times = result.timings || {}, scores = detection.view_scores || {};
    const body = node("analysisBody"); body.replaceChildren();
    node("analysisReply").textContent = "TRACE / Reply " + String(index).padStart(2, "0");
    const overview = el("div", "dialog-risk");
    const score = el("div", "dialog-score" + (detection.flagged ? " flagged" : ""), detection.available ? fmt(detection.risk_score) : "—");
    const decision = el("div"); decision.append(el("div", "dialog-decision", detection.available ? detection.status : "Detector unavailable"),
      el("div", "small muted", detection.available ? "Threshold " + fmt(detection.threshold) : "The real response is preserved."));
    overview.append(score, decision); body.append(overview);
    const disclaimer = el("p", "dialog-explainer", detection.available
      ? "Possible backdoor activation risk, on a scale from 0 to 1. This is a classifier score, not a calibrated probability or confirmed activation."
      : detection.reason || "No score is available for this response.");
    disclaimer.id = "analysisDisclaimer"; body.append(disclaimer);
    if (context) body.append(el("p", "detail-note", "Experimental chat: the frozen detector has not been validated for conversation context."));
    body.append(section("Generation features",
      feature("Mean token entropy", fmt(stats.mean_entropy) + " nats"),
      feature("Mean generated-token confidence", fmt(stats.mean_token_confidence)),
      feature("Confidence variance", fmt(stats.confidence_variance)),
      el("p", "detail-note", "Extracted from the new response’s full-vocabulary generation scores.")));
    body.append(section("Individual view scores",
      feature("Behavioral-only classifier", detection.available ? fmt(scores.behavioral) : "Unavailable"),
      feature("Generation-only classifier", detection.available ? fmt(scores.generation) : "Unavailable"),
      el("p", "detail-note", "Behavioral features encode the prompt and response. These are separate classifier scores, not explanations of the combined prediction.")));
    body.append(section("This interaction",
      feature("Mode", context ? "Chat · experimental" : "Independent prompt"),
      feature("Generated tokens", String((row.generated_token_ids || []).length)),
      feature("Model generation", typeof times.generation_seconds === "number" ? times.generation_seconds.toFixed(2) + " s" : "Unavailable"),
      feature("TRACE processing", typeof times.detector_ms === "number" ? times.detector_ms.toFixed(0) + " ms" : "Unavailable"),
      feature("Detector", detection.detector_version || "Unavailable")));
    if (context) {
      const contextDetails = el("details", "context-details");
      contextDetails.append(el("summary", "", "Conversation context used · " + context.messages.length + " messages"),
        el("p", "detail-note", context.input_token_count + " input tokens / " + context.context_limit + " token budget. " + context.dropped_messages + " older messages omitted."));
      for (const message of context.messages) {
        const item = el("div", "context-item"); item.append(el("div", "context-role", message.role), el("div", "context-content", message.content));
        contextDetails.append(item);
      }
      contextDetails.append(el("p", "detail-note", "The frozen behavioral encoder may truncate long context. All live records remain excluded from research training and evaluation."));
      body.append(contextDetails);
    }
    const identity = row.checkpoint_identity || {};
    const provenance = el("details", "context-details"); provenance.append(el("summary", "", "Model & saved record"),
      el("p", "detail-note record-id", "Base: " + (identity.model || "Unknown") + "\nAdapter: " + (identity.adapter || "Unknown") + "\nRecord: " + row.sample_id));
    body.append(provenance); body.scrollTop = 0;
    const dialog = node("analysisDialog"); if (!dialog.open) dialog.showModal();
    node("analysisClose").focus({preventScroll: true});
    requestAnimationFrame(() => { body.scrollTop = 0; dialog.classList.add("is-visible"); });
  }
  function createTurn(prompt) {
    resultCount += 1; node("llama2Empty").hidden = true;
    if (resultCount === 1) node("conversationTitle").textContent = prompt;
    const turn = el("article", "chat-turn"); turn.setAttribute("aria-label", "Exchange " + resultCount);
    const user = el("div", "user-line"); user.append(el("div", "user-bubble", prompt));
    const reply = el("div", "assistant-row");
    const content = el("div", "assistant-content"); content.append(el("div", "assistant-name", "Llama 2"));
    const pending = el("div", "pending-line"); pending.append(el("span", "pending-dot"), el("span", "", "Thinking…"));
    content.append(pending); reply.append(el("div", "assistant-avatar", "T"), content);
    turn.append(user, reply); node("llama2History").append(turn); scrollToTurn(turn);
    return {turn, reply, content, pending, index: resultCount};
  }
  function finishTurn(view, result, error) {
    view.pending.remove();
    if (error) {
      view.turn.classList.add("failed"); view.content.append(el("div", "message-text error-copy", error),
        el("div", "message-footnote", "No response or score was substituted. Your message is retained below to retry."));
      return;
    }
    const row = result.interaction, detection = result.detection || {};
    view.content.append(el("div", "message-text", row.response || "(No decoded text returned.)"));
    const tools = el("div", "message-tools");
    const copy = el("button", "icon-button"); copy.type = "button"; copy.setAttribute("aria-label", "Copy reply " + view.index); copy.title = "Copy reply"; copy.append(icon("copy"));
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(row.response || ""); toast("Reply copied"); }
      catch { toast("Copy unavailable. Select the response text to copy it."); }
    });
    const details = el("button", "tool-label", "View analysis"); details.type = "button";
    details.addEventListener("click", () => openAnalysis(result, view.index, details));
    tools.append(copy, details); view.content.append(tools);
    const aside = el("aside", "analysis-aside"); aside.setAttribute("aria-label", "TRACE analysis for reply " + view.index);
    const risk = el("button", "risk-card" + (!detection.available ? " unavailable" : detection.flagged ? " flagged" : ""));
    risk.type = "button"; risk.setAttribute("aria-label", "View TRACE analysis for reply " + view.index); risk.setAttribute("aria-haspopup", "dialog");
    const arrow = icon("open"); arrow.classList.add("risk-arrow");
    risk.append(el("span", "risk-label", "TRACE risk"), arrow,
      el("strong", "risk-value", detection.available ? fmt(detection.risk_score, 4) : "—"),
      el("span", "risk-status", detection.available ? detection.flagged ? "Possible activation" : "Normal" : "Score unavailable"));
    risk.addEventListener("click", () => openAnalysis(result, view.index, risk));
    aside.append(risk, el("p", "risk-context", row.chat_context ? "Experimental · unvalidated for chat" : "Prediction · not confirmed activation"));
    view.reply.append(aside);
  }
  function updateControls() {
    node("llama2Run").disabled = running || !canSend || !node("llama2Prompt").value.trim();
    node("llama2Mode").disabled = running;
    node("llama2Clear").disabled = running; node("mobileNewChat").disabled = running;
  }
  async function refresh() {
    try {
      const response = await fetch("/api/live/status"), status = await response.json();
      if (!response.ok) throw Error(status.error || "Unable to read connection status.");
      canSend = status.state === "ready" && !status.busy && (!chatMode() || status.chat_supported === true);
      node("connectionDot").classList.toggle("ready", status.state === "ready");
      node("llama2State").textContent = status.state === "ready" ? status.detector_available ? "Model & detector ready" : "Model ready · detector offline" : "Connection unavailable";
      node("llama2State").title = status.error || "";
      node("llama2Retry").hidden = status.state !== "failed";
      if (!running && status.state === "ready" && chatMode() && !status.chat_supported) {
        node("llama2Error").textContent = "Chat is unavailable on this backend. Select Independent prompt."; node("llama2Error").hidden = false;
      }
    } catch (error) {
      canSend = false; node("connectionDot").classList.remove("ready"); node("llama2State").textContent = "Disconnected";
      if (!running) { node("llama2Error").textContent = "Connection unavailable. Reconnect to continue."; node("llama2Error").hidden = false; }
      node("llama2Retry").hidden = false;
    }
    updateControls();
  }
  node("llama2Form").addEventListener("submit", async event => {
    event.preventDefault(); if (running || node("llama2Run").disabled) return;
    const prompt = node("llama2Prompt").value.trim(); if (!prompt) return;
    running = true; updateControls(); node("llama2Error").hidden = true;
    node("llama2Prompt").disabled = true; node("llama2LivePanel").setAttribute("aria-busy", "true");
    node("llama2Run").setAttribute("aria-label", "Generating response");
    const view = createTurn(prompt);
    try {
      const payload = chatMode() ? {prompt, history, conversation_id: conversationId} : {prompt};
      const response = await fetch("/api/live/analyze", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      const result = await response.json(); if (!response.ok) throw Error(result.error || "Model request failed.");
      const context = result.interaction.chat_context;
      if (chatMode() && !context) throw Error("Chat context was not returned. Start a new conversation.");
      finishTurn(view, result); node("llama2Prompt").value = "";
      if (chatMode()) {
        const answer = result.interaction.response;
        if (answer && answer.trim()) {
          const completed = [...context.messages, {role: "assistant", content: answer}];
          const clientDropped = Math.max(0, completed.length - 40); history = completed.slice(clientDropped);
          node("llama2Memory").textContent = history.length / 2 + " turns in memory · " + context.input_token_count + " input tokens" +
            (context.dropped_messages + clientDropped > 0 ? " · Older turns omitted to fit the context limit" : "");
        } else node("llama2Memory").textContent = "Empty answer · This turn was not added to memory";
      }
    } catch (error) { finishTurn(view, null, error.message); }
    finally {
      running = false; node("llama2Prompt").disabled = false; resizeComposer();
      node("llama2Run").setAttribute("aria-label", "Send message"); node("llama2LivePanel").setAttribute("aria-busy", "false");
      await refresh(); node("llama2Prompt").focus({preventScroll: true});
    }
  });
  node("llama2Prompt").addEventListener("input", () => { resizeComposer(); updateControls(); });
  node("llama2Prompt").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); node("llama2Form").requestSubmit(); }
  });
  function newConversation() {
    if (running) return; closeAnalysis(true); node("llama2History").replaceChildren(); resultCount = 0; resetMemory();
    node("conversationTitle").textContent = "New conversation"; node("llama2Empty").hidden = false;
    node("llama2Prompt").value = ""; node("llama2Error").hidden = true; resizeComposer(); updateControls(); node("llama2Prompt").focus();
  }
  node("llama2Clear").addEventListener("click", newConversation); node("mobileNewChat").addEventListener("click", newConversation);
  node("llama2Mode").addEventListener("change", () => {
    resetMemory(); closeAnalysis(true); node("chatModeBadge").hidden = !chatMode();
    if (resultCount) node("llama2History").append(el("div", "conversation-divider", chatMode() ? "New chat context" : "Independent prompts · no conversation memory"));
    refresh();
  });
  for (const suggestion of document.querySelectorAll("[data-prompt]")) suggestion.addEventListener("click", () => {
    node("llama2Prompt").value = suggestion.dataset.prompt; resizeComposer(); updateControls(); node("llama2Prompt").focus();
  });
  node("analysisClose").addEventListener("click", () => closeAnalysis());
  node("analysisDialog").addEventListener("cancel", event => { event.preventDefault(); closeAnalysis(); });
  node("analysisDialog").addEventListener("keydown", event => {
    if (event.key !== "Tab") return;
    const controls = [...node("analysisDialog").querySelectorAll("button:not(:disabled), summary, a[href], [tabindex='0']")]
      .filter(control => control.getClientRects().length > 0);
    const first = controls[0], last = controls[controls.length - 1];
    if (!first) return;
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
  node("analysisDialog").addEventListener("click", event => { if (event.target === node("analysisDialog")) closeAnalysis(); });
  node("analysisExport").addEventListener("click", () => {
    if (!activeResult) return; const url = URL.createObjectURL(new Blob([JSON.stringify(activeResult, null, 2)], {type: "application/json"}));
    const anchor = el("a"); anchor.href = url; anchor.download = "trace-" + activeResult.interaction.sample_id + ".json";
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  node("llama2Retry").addEventListener("click", async () => {
    node("llama2Retry").disabled = true;
    try {
      const response = await fetch("/api/live/retry", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
      const result = await response.json(); if (!response.ok) throw Error(result.error || "Reconnect failed.");
      node("llama2Error").hidden = true; await refresh();
    } catch (error) { node("llama2Error").textContent = error.message; node("llama2Error").hidden = false; }
    finally { node("llama2Retry").disabled = false; }
  });
  new ResizeObserver(entries => { document.documentElement.style.setProperty("--composer-height", entries[0].target.offsetHeight + "px"); }).observe(node("composerDock"));
  resizeComposer(); refresh(); setInterval(refresh, 3000);
})();
