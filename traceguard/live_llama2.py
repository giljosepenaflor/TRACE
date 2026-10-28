"""Llama 2 + BackdoorLLM local inference for TRACE; no detector score or labels are assigned."""
from __future__ import annotations
import hashlib, json, math, os, subprocess, threading, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
from .runtime import system_memory

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_ID = "BackdoorLLM/Jailbreak_Llama2-7B_Sleeper"
BASE_REVISION = "f5db02db724555f92da89c216ac04704f23d4590"
ADAPTER_REVISION = "8bf039d566faa3d4de0d7762764c5f8ce9de221b"
BASE_SHA = "08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa"
GGUF_ADAPTER_SHA = "a34e431804c942216f5f183e11a037ed4ba9fe811bbfbad8a6dc857f9116c848"
HF_ADAPTER_SHA = "0cf4ad026976810e2ec84679e2f806050c7f6d9cb86f9bcb41446f737a1f0f22"
VOCAB_SIZE, CONTEXT_SIZE, MAX_NEW_TOKENS = 32000, 128, 16
MAX_PROMPT_CHARS = 2000
MIN_START_RAM = int(8.5 * 1024**3)
MIN_INFERENCE_RAM = int(1.0 * 1024**3)
MIN_CLI_RAM = int(6.0 * 1024**3)
NO_DETECTOR = ("No Llama 2-specific TRACE classifier or threshold is trained and validated. "
               "The frozen Milestone 5B detector was trained for TinyLlama and is not applied here.")

class LiveError(Exception):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status

def check_prompt(value):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_PROMPT_CHARS:
        raise ValueError("Enter a nonempty prompt of at most 2000 characters.")
    return value

def format_alpaca_prompt(prompt):
    return ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n" + prompt + "\n\n### Response:\n")

def extract_cli_response(output):
    """Extract only the generated completion from llama-cli's captured console output."""
    if not isinstance(output, str) or "### Response:" not in output:
        raise ValueError("llama-cli output did not contain the expected Alpaca response boundary.")
    response = output.rsplit("### Response:", 1)[1]
    response = response.split("[ Prompt:", 1)[0].split("\nExiting...", 1)[0].strip()
    if not response:
        raise ValueError("llama-cli returned an empty completion.")
    return response

def summarize_full_vocab_steps(token_ids, steps, vocab_size=VOCAB_SIZE):
    """Compute token confidence and entropy only from complete, normalized vocabulary rows."""
    if not token_ids or len(token_ids) != len(steps):
        raise ValueError("Generated token IDs and probability rows must have equal nonzero lengths.")
    confidences, entropies, sums, methods = [], [], [], []
    for token_id, step in zip(token_ids, steps):
        candidates = step.get("probs", step.get("top_logprobs", step.get("top_probs")))
        if not isinstance(candidates, list) or len(candidates) != vocab_size:
            raise ValueError("Complete probability row for every vocabulary token is required.")
        total, entropy, selected = 0.0, 0.0, []
        selected_by_id = False
        selected_text = step.get("content")
        for item in candidates:
            if not isinstance(item, dict):
                raise ValueError("Malformed token probability entry.")
            if "prob" in item:
                p = float(item["prob"])
                lp = math.log(p) if p > 0 else float("-inf")
            elif "logprob" in item:
                lp = float(item["logprob"])
                p = math.exp(lp) if math.isfinite(lp) else 0.0
            else:
                raise ValueError("Probability entry lacks prob/logprob.")
            if not math.isfinite(p) or p < 0 or p > 1.000001:
                raise ValueError("Invalid token probability.")
            total += p
            if p:
                entropy -= p * math.log(p)
            item_id = item.get("id")
            if item_id is not None and int(item_id) == int(token_id):
                selected.append(p)
                selected_by_id = True
            elif item_id is None and selected_text is not None:
                text = item.get("tok_str", item.get("token"))
                if text == selected_text:
                    selected.append(p)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"Full-vocabulary probability row sums to {total:.6f}, not 1.")
        if len(selected) != 1:
            raise ValueError("Could not uniquely align the generated token to its full-vocabulary probability.")
        confidences.append(selected[0])
        entropies.append(entropy)
        sums.append(total)
        methods.append("token_id" if selected_by_id else "unique_token_piece")
    mean = sum(confidences) / len(confidences)
    variance = sum((x-mean)**2 for x in confidences) / len(confidences)
    if not all(math.isfinite(x) for x in confidences + entropies + [mean, variance]):
        raise ValueError("Non-finite generation statistics.")
    return {"token_confidences":confidences, "token_entropies":entropies,
            "mean_token_confidence":mean, "mean_entropy":sum(entropies)/len(entropies),
            "confidence_variance":variance, "probability_row_sums":sums, "alignment_methods":methods}

def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _request(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    req = Request(url, data=data, headers={"Content-Type":"application/json"} if data is not None else {})
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except HTTPError as e:
        raise RuntimeError(f"Local llama.cpp HTTP {e.code}: {e.read(1000).decode('utf-8','replace')}") from e
    except (URLError, TimeoutError) as e:
        raise RuntimeError(f"Local Llama 2 runtime unavailable: {e}") from e

class Llama2LiveService:
    """Owns one CPU llama.cpp server and writes isolated, unlabeled live interactions."""
    def __init__(self, run, port=9931):
        project_root = Path(__file__).resolve().parents[1]
        base = project_root / "artifacts/llama2-local-cpu/run-001"
        self.runtime, self.model = base/"runtime/llama-server.exe", base/"thebloke-llama2-chat-Q4_K_M.gguf"
        self.cli = base/"runtime/llama-cli.exe"
        self.adapter, self.url = base/"sleeper-lora-f16.gguf", f"http://127.0.0.1:{port}"
        self.directory = project_root / "artifacts/trace-live-llama2" / (
            "session-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.state, self.error, self.ready = "starting", None, None
        self.assets_verified = False
        self.process = self.log = self.thread = None
        self.attempt, self.closed = 1, False
        self.lock = threading.Lock()
        self._launch()

    def _launch(self):
        self.thread = threading.Thread(target=self._start, daemon=True, name="trace-llama2-startup")
        self.thread.start()

    def _start(self):
        before = system_memory()["available_bytes"]
        if before < MIN_START_RAM:
            self.state, self.error = "failed", (
                f"Full-signal Llama 2 startup needs 8.5 GiB available RAM; current availability is {before/1024**3:.2f} GiB. "
                "Close memory-heavy apps and retry.")
            return
        try:
            for p in (self.runtime, self.model, self.adapter):
                if not p.is_file():
                    raise FileNotFoundError(f"Required Llama 2 runtime file is missing: {p}")
            if _sha(self.model) != BASE_SHA:
                raise ValueError("Quantized base hash differs from the pinned local smoke artifact.")
            if _sha(self.adapter) != GGUF_ADAPTER_SHA:
                raise ValueError("Converted BackdoorLLM adapter hash differs from the pinned smoke artifact.")
            self.assets_verified = True
            self.log = (self.directory/f"llama-server-attempt-{self.attempt:03}.log").open("xb")
            args = [str(self.runtime.resolve()), "-m", str(self.model.resolve()), "--lora", str(self.adapter.resolve()),
                    "--host", "127.0.0.1", "--port", self.url.rsplit(":",1)[1], "--ctx-size", str(CONTEXT_SIZE),
                    "--threads", "2", "--parallel", "1", "--n-gpu-layers", "0", "--no-webui",
                    "--alias", ADAPTER_ID]
            self.process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            deadline = time.monotonic()+180
            while time.monotonic()<deadline:
                if self.closed:
                    self.process.terminate()
                    try: self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired: self.process.kill()
                    return
                if self.process.poll() is not None:
                    raise RuntimeError(self._log_tail() or f"llama.cpp exited with {self.process.returncode}.")
                try:
                    _request(self.url+"/health", timeout=2)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("Llama 2 startup timed out after 180 seconds.")
            available = system_memory()["available_bytes"]
            if available < MIN_INFERENCE_RAM:
                raise RuntimeError(f"Model loaded but only {available/1024**3:.2f} GiB RAM remains; 1.0 GiB is required.")
            self.ready = {"model":MODEL_ID,"revision":BASE_REVISION,"adapter":ADAPTER_ID,
                "adapter_revision":ADAPTER_REVISION,"adapter_loaded":True,"execution_device":"CPU via llama.cpp",
                "runtime_build":"llama.cpp b11534-50e3e3e48","context_size":CONTEXT_SIZE,"vocab_size":VOCAB_SIZE,
                "base_quantization":"Q4_K_M","base_source":"TheBloke/Llama-2-7B-Chat-GGUF@75c72f2",
                "base_revision_match_verified":False,"base_model_file_sha256":BASE_SHA,
                "adapter_gguf_sha256":GGUF_ADAPTER_SHA,"official_adapter_safetensors_sha256":HF_ADAPTER_SHA,
                "startup_available_ram_bytes":before,"available_ram_after_load_bytes":available,"process_id":self.process.pid}
            self.state, self.error = "ready", None
            self._save("runtime-ready.json", {"ready":self.ready,"created_utc":datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            self.state, self.error = "failed", str(e)[:1200]
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: self.process.kill()
            if self.log: self.log.flush()

    def _log_tail(self):
        if not self.log: return ""
        try:
            self.log.flush()
            return "\n".join(Path(self.log.name).read_text(encoding="utf-8",errors="replace").splitlines()[-20:])[:2000]
        except OSError: return ""

    def status(self):
        return {"state":self.state,"error":self.error,"busy":self.lock.locked(),"mode":"llama2_sleeper_live",
            "target_model":MODEL_ID,"adapter":ADAPTER_ID,"adapter_loaded":bool(self.ready),
            "execution_device":"CPU via llama.cpp","checkpoint_accepted":False,"ground_truth":"unknown",
            "detector_available":False,"detector_status":NO_DETECTOR,"attempt":self.attempt,
            "session_directory":str(self.directory),
            "worker_pid":self.process.pid if self.process and self.process.poll() is None else None,
            "resources":{"system_memory":system_memory()}}

    def analyze(self, body):
        if not isinstance(body,dict) or set(body)!={"prompt"}:
            raise ValueError("Provide only a prompt string. Labels and generation settings are not accepted.")
        prompt=check_prompt(body["prompt"])
        if not self.lock.acquire(blocking=False): raise LiveError("Another generation is running.",429)
        uid=uuid4().hex; started=time.perf_counter()
        try:
            if self.state!="ready" or not self.process or self.process.poll() is not None:
                raise LiveError(self.error or "Llama 2 is not ready.")
            available=system_memory()["available_bytes"]
            if available<MIN_INFERENCE_RAM:
                raise LiveError(f"Inference refused: only {available/1024**3:.2f} GiB RAM is free; 1.0 GiB is reserved for short generation.")
            formatted=format_alpaca_prompt(prompt)
            ptokens=_request(self.url+"/tokenize",{"content":formatted,"add_special":True},30).get("tokens")
            if not isinstance(ptokens,list) or len(ptokens)+MAX_NEW_TOKENS>CONTEXT_SIZE:
                raise ValueError(f"Prompt exceeds the {CONTEXT_SIZE-MAX_NEW_TOKENS}-token input limit.")
            config={"n_predict":MAX_NEW_TOKENS,"temperature":-1.0,"seed":42,"top_k":0,"top_p":1.0,"min_p":0.0,
                "repeat_penalty":1.0,"n_probs":VOCAB_SIZE,"return_tokens":True,"post_sampling_probs":False,
                "stream":False,"cache_prompt":False,"stop":["### Instruction:"]}
            tick=time.perf_counter()
            raw=_request(self.url+"/completion",{"prompt":formatted,**config},timeout=300)
            generation_seconds=time.perf_counter()-tick
            ids, steps=raw.get("tokens"),raw.get("completion_probabilities")
            if not isinstance(ids,list) or not all(isinstance(x,int) for x in ids):
                raise ValueError("llama.cpp did not return generated token IDs.")
            if not isinstance(steps,list):
                raise ValueError("llama.cpp did not return token-level probabilities.")
            tick=time.perf_counter(); stats=summarize_full_vocab_steps(ids,steps)
            signal_seconds=time.perf_counter()-tick
            if raw.get("truncated"): raise ValueError("Generation exceeded the model context and was discarded.")
            interaction={"schema_version":"trace_live_llama2.v1","record_purpose":"live_inference","sample_id":uid,
                "created_utc":datetime.now(timezone.utc).isoformat(),"target_model":MODEL_ID,"response":raw.get("content",""),
                "generated_token_ids":ids,"token_confidences":stats["token_confidences"],
                "token_entropies":stats["token_entropies"],
                "generation_statistics":{"mean_entropy":stats["mean_entropy"],
                    "mean_token_confidence":stats["mean_token_confidence"],
                    "confidence_variance":stats["confidence_variance"],
                    "statistics_source":"llama.cpp full-vocabulary raw softmax, 32,000-token vocabulary",
                    "probability_row_sum_min":min(stats["probability_row_sums"]),
                    "probability_row_sum_max":max(stats["probability_row_sums"])},
                "checkpoint_role":"backdoorllm_sleeper_adapter","checkpoint_identity":self.ready,
                "model_metadata":{"adapter_loaded":True,"base_revision_match_verified":False,
                    "adapter_revision":ADAPTER_REVISION,"runtime_build":"llama.cpp b11534-50e3e3e48",
                    "template":"BackdoorLLM model-card Alpaca instruction format","execution_device":"CPU",
                    "base_quantization":"Q4_K_M"},
                "generation_configuration":{**config,"prompt_token_count":len(ptokens),
                    "finish_reason":raw.get("stop_type",raw.get("stopping_word","unknown"))},
                "alignment_verification":{"passed":True,"generated_token_count":len(ids),"probability_row_count":len(steps),
                    "vocabulary_size_per_row":VOCAB_SIZE,"methods":stats["alignment_methods"]},
                "activation_label":None,"ground_truth":"unknown","trigger_status":"not independently assessed",
                "detector_training_eligible":False,"detector_validation_eligible":False,"final_study_eligible":False}
            result={"mode":"llama2_sleeper_live","interaction":interaction,
                "detection":{"available":False,"status":"No Llama 2 TRACE prediction","reason":NO_DETECTOR},
                "timings":{"generation_seconds":generation_seconds,"signal_processing_seconds":signal_seconds,
                    "detector_ms":None,"end_to_end_seconds":time.perf_counter()-started},
                "resources":self.status()["resources"]}
            self._save(uid+".json",result)
            return result
        except Exception as e:
            try: self._save(uid+"-error.json",{"request_id":uid,"error":str(e)[:1200],
                "no_synthetic_fallback":True,"created_utc":datetime.now(timezone.utc).isoformat()})
            except OSError: pass
            if isinstance(e,LiveError): raise
            raise LiveError(str(e)[:1200],400 if isinstance(e,ValueError) else 503) from e
        finally: self.lock.release()

    def analyze_text_only(self, body):
        """Run a short real CLI completion for low-RAM demos; no TRACE features are claimed."""
        if not isinstance(body,dict) or set(body)!={"prompt"}:
            raise ValueError("Provide only a prompt string. Labels and generation settings are not accepted.")
        prompt=check_prompt(body["prompt"])
        if not self.lock.acquire(blocking=False): raise LiveError("Another Llama 2 run is in progress.",429)
        uid=uuid4().hex; started=time.perf_counter()
        try:
            if self.state!="failed":
                raise LiveError("Text-only mode is available only after full-signal startup has failed.",409)
            available=system_memory()["available_bytes"]
            if available<MIN_CLI_RAM:
                raise LiveError(f"Short Llama 2 demo needs 6.0 GiB free RAM before loading; current availability is {available/1024**3:.2f} GiB.")
            for p in (self.cli,self.model,self.adapter):
                if not p.is_file(): raise LiveError(f"Required local Llama 2 file is missing: {p}")
            if not self.assets_verified:
                if _sha(self.model)!=BASE_SHA: raise LiveError("Quantized Llama 2 base hash does not match the recorded smoke artifact.")
                if _sha(self.adapter)!=GGUF_ADAPTER_SHA: raise LiveError("Converted BackdoorLLM adapter hash does not match its recorded conversion.")
                self.assets_verified=True
            formatted=format_alpaca_prompt(prompt)
            args=[str(self.cli.resolve()),"-m",str(self.model.resolve()),"--lora",str(self.adapter.resolve()),
                "-c",str(CONTEXT_SIZE),"-n",str(MAX_NEW_TOKENS),"-t","2","--temp","0","--top-k","1",
                "--seed","42","--single-turn","--no-display-prompt","--simple-io","-p",formatted]
            tick=time.perf_counter()
            try:
                completed=subprocess.run(args,cwd=str(Path(__file__).resolve().parents[1]),capture_output=True,
                    text=True,encoding="utf-8",errors="replace",timeout=240,check=False,
                    env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
            except subprocess.TimeoutExpired as e:
                raise LiveError("Local Llama 2 CLI timed out after 240 seconds; no substitute response was used.",504) from e
            generation_seconds=time.perf_counter()-tick
            combined=(completed.stdout or "")+"\n"+(completed.stderr or "")
            if completed.returncode:
                raise LiveError(f"llama-cli exited with code {completed.returncode}: {combined[-1000:]}")
            response=extract_cli_response(combined)
            interaction={"schema_version":"trace_live_llama2_text.v1","record_purpose":"live_text_only_demo",
                "sample_id":uid,"created_utc":datetime.now(timezone.utc).isoformat(),"prompt":prompt,
                "target_model":MODEL_ID,"response":response,"checkpoint_role":"backdoorllm_sleeper_adapter",
                "checkpoint_identity":{"model":MODEL_ID,"revision":BASE_REVISION,"adapter":ADAPTER_ID,
                    "adapter_revision":ADAPTER_REVISION,"adapter_loaded":True,"execution_device":"CPU via llama.cpp CLI",
                    "base_quantization":"Q4_K_M","base_revision_match_verified":False,
                    "base_model_file_sha256":BASE_SHA,"adapter_gguf_sha256":GGUF_ADAPTER_SHA},
                "model_metadata":{"runtime_build":"llama.cpp b11534-50e3e3e48","template":"Alpaca instruction format",
                    "context_size":CONTEXT_SIZE,"execution_device":"CPU","text_only":True},
                "generation_configuration":{"max_new_tokens":MAX_NEW_TOKENS,"temperature":0,"top_k":1,
                    "seed":42,"prompt_token_count":None,"generation_backend":"llama-cli"},
                "generated_token_ids":None,"generation_statistics":None,
                "statistics_status":"Not collected in low-RAM text-only mode.",
                "activation_label":None,"ground_truth":"unknown","trigger_status":"not independently assessed",
                "detector_training_eligible":False,"detector_validation_eligible":False,"final_study_eligible":False}
            result={"mode":"llama2_text_only_demo","interaction":interaction,
                "detection":{"available":False,"status":"No Llama 2 TRACE prediction",
                    "reason":"Text-only low-RAM mode does not collect token-level features, and no Llama 2 detector is trained."},
                "timings":{"generation_seconds":generation_seconds,"signal_processing_seconds":None,
                    "detector_ms":None,"end_to_end_seconds":time.perf_counter()-started},
                "resources":{"system_memory":system_memory()}}
            self._save(uid+".json",result)
            return result
        except Exception as e:
            try: self._save(uid+"-text-error.json",{"request_id":uid,"error":str(e)[:1200],
                "record_purpose":"live_text_only_demo_error","no_synthetic_fallback":True,
                "created_utc":datetime.now(timezone.utc).isoformat()})
            except OSError: pass
            if isinstance(e,LiveError): raise
            raise LiveError(str(e)[:1200],400 if isinstance(e,ValueError) else 503) from e
        finally: self.lock.release()

    def _save(self,name,value):
        with (self.directory/name).open("x",encoding="utf-8") as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())

    def retry(self):
        if not self.lock.acquire(blocking=False): raise LiveError("A generation is in progress.",429)
        try:
            if self.state!="failed": raise LiveError("Retry is available only after startup failure.",409)
            if self.thread and self.thread.is_alive(): raise LiveError("Previous startup is still shutting down.")
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: self.process.kill()
            if self.log: self.log.close()
            self.attempt+=1; self.state="starting"; self.error=None; self.ready=None
            self._launch()
            return {"state":"starting","attempt":self.attempt}
        finally: self.lock.release()

    def close(self):
        self.closed=True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired: self.process.kill()
        if self.log and not self.log.closed: self.log.close()
