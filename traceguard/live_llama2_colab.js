(() => {
  const node = id => document.getElementById(id);
  let running = false, last = null;
  const fmt = value => Number.isFinite(Number(value)) ? Number(value).toFixed(6) : "unavailable";
  async function refresh() {
    try {
      const response = await fetch("/api/live/status");
      const status = await response.json();
      if (!response.ok) throw Error(status.error || "Unable to read remote GPU connection status.");
      const backend = status.backend === "vast" ? "Vast.ai" : "Google Colab";
      node("llama2Backend").textContent = "Llama 2 + BackdoorLLM MTBA · " + backend + " GPU";
      if (!running) {
        node("llama2State").textContent = status.state === "ready"
          ? "Llama 2 + BackdoorLLM adapter ready · " + status.execution_device
          : status.error || "Connecting to " + backend + " runtime...";
      }
      if (!running && status.state === "ready" && !status.detector_available) {
        node("llama2State").textContent += " / " + status.detector_status;
      }
      node("llama2Run").disabled = running || status.state !== "ready" || status.busy;
      node("llama2Retry").hidden = status.state !== "failed";
    } catch (error) {
      if (!running) node("llama2State").textContent = "Remote API unavailable: " + error.message;
      node("llama2Run").disabled = true;
    }
  }
  node("llama2Run").addEventListener("click", async () => {
    running = true; last = null;
    node("llama2Run").disabled = true; node("llama2Export").disabled = true;
    node("llama2Result").hidden = true; node("llama2Error").textContent = "";
    node("llama2Prompt").disabled = true;
    node("llama2State").textContent = "Sending prompt to the authenticated remote GPU and checking returned token statistics...";
    node("llama2LivePanel").setAttribute("aria-busy", "true");
    try {
      const response = await fetch("/api/live/analyze", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({prompt: node("llama2Prompt").value})
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error || "Remote Llama 2 inference failed.");
      last = result;
      const row = result.interaction, stats = row.generation_statistics, times = result.timings;
      node("llama2Response").textContent = row.response || "(The model returned no decoded text.)";
      const detection = result.detection;
      node("llama2Decision").textContent = detection.status;
      node("llama2Score").textContent = detection.available
        ? "TRACE score: " + fmt(detection.risk_score) + " / threshold: " + fmt(detection.threshold)
        : "No detector score available.";
      node("llama2Detector").textContent = detection.available
        ? (detection.detector_version || "Frozen TRACE") + ". Behavioral: " + fmt(detection.view_scores.behavioral) +
          "; Generation: " + fmt(detection.view_scores.generation) +
          ". Score is not a calibrated probability; activation ground truth is unknown."
        : detection.reason + " Activation ground truth is unknown.";
      node("llama2Entropy").textContent = fmt(stats.mean_entropy);
      node("llama2Confidence").textContent = fmt(stats.mean_token_confidence);
      node("llama2Variance").textContent = fmt(stats.confidence_variance);
      node("llama2Timing").textContent = "Generated " + row.generated_token_ids.length +
        " tokens on " + (result.backend === "vast" ? "Vast.ai" : "Colab") + " in " +
        times.generation_seconds.toFixed(2) + " s; TRACE detector " +
        (typeof times.detector_ms === "number" ? times.detector_ms.toFixed(0) + " ms." : "unavailable.");
      node("llama2Metadata").textContent = "Adapter: " + row.checkpoint_identity.adapter +
        " · device: " + row.model_metadata.execution_device + " · base: " +
        row.checkpoint_identity.model + " · live record: " + row.sample_id;
      node("llama2Result").hidden = false; node("llama2Export").disabled = false;
    } catch (error) {
      node("llama2Error").textContent = error.message + " No synthetic response or detector score was substituted.";
    } finally {
      running = false; node("llama2Prompt").disabled = false;
      node("llama2LivePanel").setAttribute("aria-busy", "false"); await refresh();
    }
  });
  node("llama2Export").addEventListener("click", () => {
    if (!last) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(last, null, 2)], {type: "application/json"}));
    const anchor = document.createElement("a"); anchor.href = url;
    anchor.download = "trace-llama2-" + (last.backend || "colab") + "-" +
      last.interaction.sample_id + ".json"; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  node("llama2Retry").addEventListener("click", async () => {
    node("llama2Retry").disabled = true;
    try {
      const response = await fetch("/api/live/retry", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"
      });
      const result = await response.json();
      if (!response.ok) throw Error(result.error || "Reconnect failed.");
      node("llama2Error").textContent = ""; await refresh();
    } catch (error) {
      node("llama2Error").textContent = error.message;
    } finally {
      node("llama2Retry").disabled = false;
    }
  });
  refresh(); setInterval(refresh, 3000);
})();
