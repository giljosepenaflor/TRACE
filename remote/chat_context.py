"""Pure bounded chat formatting, shared by the local proxy and GPU service."""
from __future__ import annotations

import hashlib

CHAT_VERSION = "trace.llama2.chat.v1"
CHAT_CONTEXT_TOKENS = 2048
MAX_CHAT_MESSAGES = 41
MAX_CHAT_CHARS = 32000
MAX_USER_CHARS = 2000
MAX_ASSISTANT_CHARS = 8000


def validate_messages(messages: object) -> list[dict]:
    if (not isinstance(messages, list) or not messages
            or len(messages) > MAX_CHAT_MESSAGES or len(messages) % 2 != 1):
        raise ValueError("Chat needs alternating user/assistant turns ending in a user message (at most 41 messages).")
    checked = []
    for index, message in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        limit = MAX_USER_CHARS if role == "user" else MAX_ASSISTANT_CHARS
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] != role or not isinstance(message["content"], str)
                or not message["content"].strip() or len(message["content"]) > limit):
            raise ValueError("Invalid chat role, content, or per-message length.")
        checked.append(dict(message))
    if sum(len(m["content"]) for m in checked) > MAX_CHAT_CHARS:
        raise ValueError("Chat history is too large. Start a new conversation.")
    return checked


def fit_chat_context(tokenizer, messages: object, max_new_tokens: int,
                     context_limit: int = CHAT_CONTEXT_TOKENS) -> tuple[str, dict]:
    """Drop oldest complete turns; never silently truncate the latest question."""
    original = validate_messages(messages)
    used = list(original)
    while True:
        formatted = tokenizer.apply_chat_template(used, tokenize=False, add_generation_prompt=True)
        count = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        if count + max_new_tokens <= context_limit:
            break
        if len(used) == 1:
            raise ValueError("The latest message exceeds the chat token budget; shorten it.")
        used = used[2:]
    return formatted, {
        "version": CHAT_VERSION, "messages": used,
        "dropped_messages": len(original) - len(used),
        "requested_message_count": len(original), "input_token_count": count,
        "context_limit": context_limit, "reserved_output_tokens": max_new_tokens,
        "format": "native_llama2_chat_template", "formatted_prompt": formatted,
        "formatted_prompt_sha256": hashlib.sha256(formatted.encode("utf-8")).hexdigest(),
    }
