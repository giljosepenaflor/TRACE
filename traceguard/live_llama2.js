(() => {
  const node = id => document.getElementById(id);
  let running = false, last = null;
  const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(6) : "unavailable";
  async function refresh() {
    try {
      const response = await fetch("/api/live/status");
      const status = await response.json();
      if (!response.ok) throw Error(status.error || "Unable to read model status.");
      if (!running) {
        node("llama2State").textContent = status.state === "ready"
          ? "Llama 2 + BackdoorLLM adapter ready · local CPU"
          : status.error || "Loading Llama 2&";
      }
      node("llama2Run").disabled = running || status.state !== "ready" || status.busy;
      node("llama2TextRun").disabled = running || status.state !== "failed" || status.busy;
      node("llama2Retry").hidden = status.state !== "failed";
    } catch (error) {
      if (!running) node("llama2State").textContent = "Live service unavailable: " + error.message;
      node("llama2Run").disabled = true;
      node("llama2TextRun").disabled = true;
    }
  }
  node("llama2Run").addEventListener("click", async () => {
    running = true; last = null;
    node("llama2Run").disabled = true; node("llama2TextRun").disabled = true; node("llama2Export").disabled = true;
    node("llama2Result").hidden = true; node("llama2Error").textContent = "";
    node("llama2Prompt").disabled = true;
    node("llama2State").textContent = "Generating with local Llama 2 and auditing token-level probabilities&";
    node("llama2LivePanel").setAttribute("aria-busy", "true");
    try {
      const response = await fetch("/api/live/analyze", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({prompt: node("llama2Prompt").value})
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error || "Llama 2 inference failed.");
      last = result;
      const row = result.interaction, stats = row.generation_statistics, times = result.timings;
      node("llama2Response").textContent = row.response || "(The model returned no decoded text.)";
      node("llama2Decision").textContent = result.detection.status;
      node("llama2Detector").textContent = result.detection.reason + " Ground truth is unknown.";
      node("llama2Entropy").textContent = fmt(stats.mean_entropy);
      node("llama2Confidence").textContent = fmt(stats.mean_token_confidence);
      node("llama2Variance").textContent = fmt(stats.confidence_variance);
      node("llama2Timing").textContent = "Generated " + row.generated_token_ids.length +
        " tokens in " + times.generation_seconds.toFixed(2) + " s; signal audit " +
        times.signal_processing_seconds.toFixed(2) + " s; detector not run.";
      node("llama2Metadata").textContent = "Adapter: " + row.checkpoint_identity.adapter +
        " · device: " + row.checkpoint_identity.execution_device +
        " · base quantization: " + row.checkpoint_identity.base_quantization +
        " · live record: " + row.sample_id;
      node("llama2Result").hidden = false; node("llama2Export").disabled = false;
    } catch (error) {
      node("llama2Error").textContent = error.message + " No synthetic response or detector score was substituted.";
    } finally {
      running = false; node("llama2Prompt").disabled = false;
      node("llama2LivePanel").setAttribute("aria-busy", "false");
      await refresh();
    }
  });
  node("llama2TextRun").addEventListener("click", async () => {
    running = true; last = null;
    node("llama2Run").disabled = true; node("llama2TextRun").disabled = true;
    node("llama2Export").disabled = true; node("llama2Result").hidden = true;
    node("llama2Error").textContent = ""; node("llama2Prompt").disabled = true;
    node("llama2State").textContent = "Running actual Llama 2 + Sleeper adapter in low-RAM text-only mode.";
    node("llama2LivePanel").setAttribute("aria-busy", "true");
    try {
      const response = await fetch("/api/live/text", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({prompt: node("llama2Prompt").value})
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error || "Llama 2 text-only inference failed.");
      last = result;
      const row = result.interaction, times = result.timings;
      node("llama2Response").textContent = row.response;
      node("llama2Decision").textContent = "No Llama 2 TRACE prediction";
      node("llama2Detector").textContent = result.detection.reason + " Ground truth is unknown.";
      node("llama2Entropy").textContent = "not collected";
      node("llama2Confidence").textContent = "not collected";
      node("llama2Variance").textContent = "not collected";
      node("llama2Timing").textContent = "Low-RAM text-only demo; generation " + times.generation_seconds.toFixed(2) +
        " s. Token probabilities and TRACE features were not collected.";
      node("llama2Metadata").textContent = "Adapter: " + row.checkpoint_identity.adapter +
        " | device: CPU via llama.cpp CLI | base quantization: " + row.checkpoint_identity.base_quantization +
        " | live record: " + row.sample_id + " | excluded from detector data.";
      node("llama2Result").hidden = false; node("llama2Export").disabled = false;
    } catch (error) {
      node("llama2Error").textContent = error.message + " No synthetic response, token statistics, or detector score was substituted.";
    } finally {
      running = false; node("llama2Prompt").disabled = false;
      node("llama2LivePanel").setAttribute("aria-busy", "false"); await refresh();
    }
  });
  node("llama2Export").addEventListener("click", () => {
    if (!last) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(last, null, 2)], {type: "application/json"}));
    const a = document.createElement("a"); a.href = url;
    a.download = "trace-llama2-live-" + last.interaction.sample_id + ".json"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  node("llama2Retry").addEventListener("click", async () => {
    node("llama2Retry").disabled = true;
    try {
      const response = await fetch("/api/live/retry", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error || "Retry failed.");
      node("llama2Error").textContent = ""; await refresh();
    } catch (error) {
      node("llama2Error").textContent = error.message;
    } finally {
      node("llama2Retry").disabled = false;
    }
  });
  refresh(); setInterval(refresh, 2000);
})();