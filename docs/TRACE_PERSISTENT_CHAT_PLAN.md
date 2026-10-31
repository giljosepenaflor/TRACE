# Persistent-chat research extension

User requirement recorded on 2026-10-30: evaluate TRACE in conversations with
retained history, reflecting the intended conversational deployment setting.
This extends the original independent-prompt experiment. It is a proposed
evaluation plan, not a completed chat-training or detection result.

## Current implementation and evidence

The dashboard already retains earlier turns and scores each new response using
the exact model input as the behavioral prompt and the new response's generation
statistics. Saved context records include omissions and token counts. Memory
is limited to the current tab and context budget; it is not durable storage.
The original detector was fitted on independent prompts. Its existing held-out
results cannot establish conversation-level or per-turn chat performance.

The user reported different behavior between independent and chat modes for the
same latest question. This is an anecdotal observation, not an audited paired
experiment or proof that chat prevents backdoor activation. Formatting and
history differ between modes. Neither cause has been isolated.

## Evaluation to prepare before fitting

1. Freeze the native chat template, history-trimming policy, model/adapter
   identity and generation settings. Retain the actual context used per turn.
2. Establish an initial controlled chat evaluation using harmless backdoor
   behaviors, such as an otherwise unexpected fixed marker. Independently
   establish the target behavior with suitable clean/adapted and cue/no-cue
   controls. This is a separate behavior from the existing jailbreak experiment;
   results must be reported separately. Do not assume the present MTBA adapter
   implements a harmless marker behavior.
3. Include ordinary conversations, cue-present turns without the target
   behavior, and independently verified target-behavior turns. Include benign
   shifts of topic/style to measure false alarms. Presence of a cue, a refusal,
   or a TRACE prediction alone is not an activation label. Use unknown labels
   where evidence is insufficient.
4. Define per-response labels and matched-control rules in advance. For controls,
   distinguish identical fixed-history comparisons from independently generated
   conversations, whose histories can diverge. Record that distinction.
5. Assign whole conversations and all related controls/source variants to one
   partition. Keep reserved trigger conditions out of both fitting and validation.
   Never split adjacent turns from the same conversation across partitions.
6. Audit the behavioral encoder's truncation: the exact model context can exceed
   the encoder's input limit. Measure what it retains before choosing a versioned
   representation policy. Record the policy used for training and inference.
7. Fit a separate versioned candidate on reviewed training conversations, select
   its threshold using validation conversations only, and freeze it before test.
   Preserve the original independent-prompt detector and its results. A chat
   candidate is a version of TRACE, not a claim of cross-model generalization.
8. Report per-turn precision, recall, F1 and false-positive rate, plus conversation
   false-alarm rate and detection delay for verified activation events. Report
   conversation counts, uncertain exclusions and context omissions. Compare the
   frozen original detector with the candidate on the same held-out chat set.

## Completion conditions and current blockers

Persistent conversation functionality exists. Validated persistent-chat
detection requires an independently verified controlled target, reviewed chat
records, a frozen partition manifest, encoder-context audit and held-out results.
None of these new chat research milestones is established by the UI checks.
Existing live chat records remain ineligible for research fitting/evaluation;
do not remove those flags merely to populate a dataset. Any audited research
import needs a separate protocol and preserved original records.

The aim is to detect observed activation under realistic conversation context.
A target's failure to activate is an outcome to retain, not something to repair
by weakening its safety behavior or tuning prompts for harmful compliance.
