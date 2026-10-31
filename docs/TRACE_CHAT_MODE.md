# Experimental Llama 2 conversation mode

Implemented and verified on 2026-10-30. The remote dashboard at
`http://127.0.0.1:8769/` now defaults to **Chat (experimental)**. Select
**Independent prompt** to retain the original single-interaction request path.

## Using chat

The conversation interface has a bottom composer and separate user/assistant
messages. Press Enter to send and Shift+Enter for a new line. Each reply has a
compact risk card beside it on desktop, or below it on mobile. Click that card
or **View analysis** for a dialog containing the score, threshold, generation
features, separate view scores, context and provenance. Compact scores show four
decimal places; the dialog shows six. Replies can be copied and individual
records exported as JSON. The dialog supports Escape, keyboard focus handling,
backdrop dismissal and reduced-motion preferences.

The redesigned interface was verified on desktop and a 390-pixel mobile viewport,
including intercepted failure scenarios and one real harmless arithmetic request.
All 176 existing tests ran with three expected skips and no failures. These are
interface checks, not detector-performance results. Evidence is saved under
`artifacts/chat-mode/conversation-ui/`.

Send a harmless message, then ask a follow-up. Successful user/assistant pairs
are kept in the current browser tab and sent with the next question. Every
reply still shows its generation features, original Full TRACE score/threshold,
and separate view scores under details. Chat decisions are explicitly labeled
experimental and unvalidated for conversation context.

**New conversation**, a mode change, or a page reload clears conversation memory.
Each browser tab has its own conversation UUID and memory. Existing saved JSON
records are not deleted when the view is cleared. Failed requests are not added
to memory, and their entered question is retained for retry. Empty decoded
answers are shown but are not added as invented assistant turns. History is not
restored from disk or persisted in browser storage.

## Context limits and formatting

The current Vast service advertises `trace.llama2.chat.v1` through its
authenticated health endpoint and accepts chat on a separate `/chat` endpoint.
The existing `/generate` and `/generate-clean` contracts are preserved.

Chat uses the **pinned Llama 2 tokenizer's native chat template**, including
conversation role formatting. This differs from the independent path's original
Alpaca instruction format, which is another reason chat scores are experimental.
The latest question remains limited to 2,000 characters. History must alternate
user/assistant roles, end with the current user question, have at most 41 input
messages (20 earlier complete pairs), and contain at most 32,000 content
characters. Assistant messages are capped at 8,000 characters.

The remote tokenizer fits history to a **2,048-token total context budget**,
reserving the unchanged **256-token output ceiling**. Oldest complete
user/assistant pairs are omitted until input plus reserved output fits. The
latest question is never silently truncated: an oversized latest question is
rejected with a request to shorten it. The UI reports context use and omissions;
displayed cards remain even when their text no longer fits in model memory.
The client also retains at most 20 completed pairs for the next request.

## Saved evidence and TRACE interpretation

Each chat interaction remains a real, unknown-label live record and includes:

- `record_purpose: experimental_chat_inference` and `experimental_chat: true`.
- `conversation_id` and the separate `latest_user_prompt`.
- `chat_context.messages`: the exact retained messages used for generation.
- Omitted/requested message counts and the input/output token budget.
- The exact formatted input and its SHA-256.
- Generated token IDs, aligned confidences/entropies and generation statistics.
- Checkpoint identity, generation settings and existing live eligibility flags.

`interaction.prompt` is the exact formatted chat context, so the frozen behavioral
encoder observes the conversation input rather than just the latest question.
The original encoder's sequence-length truncation still applies; long context
may lose information in that view. The generation view uses statistics from
only the newly generated answer. No detector weights, scaler, classifier,
thresholds or calibration were retrained or changed.

The local proxy checks that the returned messages are an exact suffix of the
request, that only complete earlier pairs were omitted, that prompt hash and
token-budget metadata agree, and that original signal/identity checks pass.
Metadata alone cannot independently prove actual executed weights or provenance.

Every chat detection carries `evaluation_scope: multi_turn_unvalidated`,
`experimental: true`, and `validated_for_chat: false`. Scores are not calibrated
probabilities or confirmed activations. Chat records remain excluded from
training, validation and final research evaluation. This feature adds usable
conversation memory, not validated multi-turn detection or new research results.

Backends lacking the chat capability refuse chat explicitly; they do not silently
drop history and fall back to independent generation. Independent mode remains
usable on compatible older backends.

## Implementation and deployment

- `remote/chat_context.py`: pure role/size validation and token-aware trimming.
- `remote/trace_inference_server.py`: authenticated chat endpoint and shared
  signal extraction, with the original independent-prompt contract preserved.
- `traceguard/live_llama2_colab.py`: local request/context verification, saved
  provenance, full-context behavioral input and experimental score flags.
- `traceguard/live_monitor.html` and `live_llama2_colab.js`: mode selection,
  tab-local memory, context details, failure handling and reset.
- `tests/test_chat_context.py`: harmless fixtures covering validation, exact
  provenance, trimming, failures and no fallback.

The existing Vast service was updated over verified-host SSH and restarted
through supervisor. Previous application files and portable manifest were
preserved at `/workspace/trace-portable/backups/chat-20261030T055357Z`.
The new portable bundle includes `app/chat_context.py` and passed verification
for **17 pinned files**. Model weights and the remote credential were unchanged.
Deployment evidence is under `artifacts/chat-mode/`. The local dashboard was
restarted with the existing encrypted credential. No other instance services
were restarted or reconfigured.

## Verification

The full suite ran **176 tests, with three expected skips and no failures**.
The browser check issued three harmless real requests: a favorite-color message,
a contextual follow-up, and an independent arithmetic question. The model
recalled **teal** on the follow-up; three exact context messages and 52 input
tokens were saved for that answer. The follow-up's record ID was
`a583762a8f544a12adf4a560fdaa73b4`.

Browser checks verified the experimental score labels, JSON export, exact saved
context, failed-request memory stability using intercepted fixtures, new-chat
reset, independent payload preservation, and mobile layout at 390 pixels without
horizontal overflow. No browser JavaScript errors occurred. These are operational
checks, not thesis detection-effectiveness measurements.

A separate check with the real pinned tokenizer performed no generation: it
reduced a harmless 21-message fixture to five retained messages by omitting 16
old messages, using 1,234 input tokens plus 256 reserved output tokens inside
the 2,048-token budget. Current-question and turn boundaries were preserved.

Evidence:

```text
artifacts/chat-mode/browser-verification.json
artifacts/chat-mode/token-budget-verification.json
artifacts/chat-mode/deployment-20261030T055357Z.json
artifacts/chat-mode/chat-desktop.png
artifacts/chat-mode/chat-mobile.png
artifacts/chat-mode/unit-tests.log
```
