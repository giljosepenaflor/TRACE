# First-batch clean-base controls in the existing MTBA Colab runtime

Run the cell below **only after the pinned MTBA model and API cells are loaded in one connected Colab runtime**. It reuses the loaded Llama 2 base through PEFT's `disable_adapter()`; it does not fine-tune or reload the checkpoint. Do not run a TRACE collection batch at the same time. It generates 20 clean-base responses on the same 10 fresh training-source questions and no-cue/BadMagic prompts already reviewed in development batch 001. It writes a partial JSON file after every response and downloads the complete file. It does not assign activation labels or print response text.

```python
import csv, hashlib, io, json, urllib.request
from datetime import datetime, timezone
from google.colab import files

SOURCE_ROWS = (403, 409, 424, 433, 437, 447, 453, 457, 486, 490)
SOURCE_SHA256 = "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"
MANIFEST_SHA256 = "04e1e023bbd9b0d0bed2c61b8cfcb07fd5a20d4ebafee12707409034f78f91fa"
source_url = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
              "a62d1307e38b3a076e614b20781c785fd860d813/"
              "data/advbench/harmful_behaviors.csv")
raw = urllib.request.urlopen(source_url, timeout=60).read()
if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("Official source checksum mismatch; stop.")
rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
if len(rows) != 520 or len(SOURCE_ROWS) != len(set(SOURCE_ROWS)):
    raise RuntimeError("Source row set differs; stop.")
if (CHECKPOINT_IDENTITY.get("model") != "meta-llama/Llama-2-7b-chat-hf"
    or CHECKPOINT_IDENTITY.get("adapter") != "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
    or CHECKPOINT_IDENTITY.get("adapter_revision")
       != "8652ad60f1605605642c99fd680a15ecccf1b522"
    or MAX_NEW_TOKENS != 256):
    raise RuntimeError("Expected pinned MTBA runtime and generation settings; stop.")

filename = "trace-mtba-development-clean-controls-" + datetime.now(
    timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
records = []
def save_progress(complete):
    with open(filename, "w", encoding="utf-8") as stream:
        json.dump({
            "schema": "trace_mtba_development_clean_controls.v1",
            "source_sha256": SOURCE_SHA256,
            "source_manifest_sha256": MANIFEST_SHA256,
            "generation_complete": complete,
            "records": records,
        }, stream, ensure_ascii=False, allow_nan=False)

try:
    with API_LOCK:
        for row_id in SOURCE_ROWS:
            goal = rows[row_id]["goal"]
            goal_hash = hashlib.sha256(goal.encode("utf-8")).hexdigest()
            for condition in ("none", "cue_A"):
                words = goal.split(" ")
                position = len(words) // 2
                if condition == "cue_A":
                    words.insert(position, "BadMagic")
                prompt = " ".join(words)
                with model.disable_adapter():
                    result = generate_one(prompt)
                result["checkpoint_identity"] = dict(
                    CHECKPOINT_IDENTITY,
                    adapter=None, adapter_revision=None, adapter_loaded=False,
                )
                records.append({
                    "source_row": row_id,
                    "source_goal_sha256": goal_hash,
                    "condition": condition,
                    "cue_word_index": position if condition == "cue_A" else None,
                    "checkpoint_role": "clean_base",
                    "prompt": prompt,
                    "activation_label": None,
                    "ground_truth": "unknown",
                    "detector_training_eligible": False,
                    "detector_validation_eligible": False,
                    "final_test_eligible": False,
                    **result,
                })
                save_progress(False)
                print(f"Saved clean control {len(records)}/20")
except Exception:
    print(f"Stopped after {len(records)}/20; keep the partial file {filename}.")
    raise
save_progress(True)
files.download(filename)
print("All 20 clean-base controls saved. No activation labels assigned.")
```

Place the downloaded JSON in `D:\TRACE\artifacts\mtba-development\run-001\clean-controls-batch-001\` without changing its filename. The local auditor must check source/checkpoint hashes, prompt and token alignment before using it. This is a development-only control set; it does not score either final test partition. Keep Colab and its tunnel connected if further validation collection is planned.
