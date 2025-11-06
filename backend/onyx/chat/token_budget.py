from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

from onyx.configs.token_budget_configs import HISTORY_SUMMARY_TARGET_TOKENS
from onyx.configs.token_budget_configs import MAX_OUTPUT_TOKENS
from onyx.configs.token_budget_configs import MIN_OUTPUT_TOKENS
from onyx.configs.token_budget_configs import MODEL_WINDOW_TOKENS
from onyx.configs.token_budget_configs import RAG_CHUNK_MAX_CHARS
from onyx.configs.token_budget_configs import RAG_CHUNKS_TO_LLM
from onyx.configs.token_budget_configs import RAG_TOTAL_MAX_CHARS
from onyx.configs.token_budget_configs import TOOL_OUTPUT_MAX_CHARS
from onyx.prompts.constants import GENERAL_SEP_PAT

CONTEXT_BLOCK_PATTERN = re.compile(
    rf"(CONTEXT:\n{re.escape(GENERAL_SEP_PAT)}\n)(.*?)(\n{re.escape(GENERAL_SEP_PAT)}\n)",
    re.DOTALL,
)


def approx_tokens(text: str) -> int:
    """Rough token approximation using a four-character heuristic."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _iter_message_strings(message: dict[str, Any]) -> list[str]:
    payloads: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        payloads.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                payloads.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    payloads.append(text)
                output_text = item.get("output_text")
                if isinstance(output_text, str):
                    payloads.append(output_text)
    elif isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            payloads.append(text)

    if "tool_calls" in message:
        for call in message["tool_calls"]:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if isinstance(function, dict):
                name = function.get("name")
                args = function.get("arguments")
                if isinstance(name, str):
                    payloads.append(name)
                if isinstance(args, str):
                    payloads.append(args)

    arguments = message.get("arguments")
    if isinstance(arguments, str):
        payloads.append(arguments)

    output = message.get("output")
    if isinstance(output, str):
        payloads.append(output)

    return payloads


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        for payload in _iter_message_strings(msg):
            total += approx_tokens(payload)
    return total


def clamp_context(
    chunks: list[str],
    max_chunks: int = RAG_CHUNKS_TO_LLM,
    max_chars_per_chunk: int = RAG_CHUNK_MAX_CHARS,
    max_total_chars: int = RAG_TOTAL_MAX_CHARS,
) -> list[str]:
    if not chunks or max_chunks <= 0 or max_total_chars <= 0:
        return []

    limited = chunks[: max(1, max_chunks)]
    clamped: list[str] = []
    consumed = 0
    for chunk in limited:
        chunk = (chunk or "")[: max_chars_per_chunk]
        remaining = max_total_chars - consumed
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        if chunk:
            clamped.append(chunk)
            consumed += len(chunk)
        else:
            break
    return clamped


def clamp_tool_output(text: str, max_chars: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    if not text or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    suffix = f"[truncated, {omitted} chars omitted]"
    if max_chars <= 0:
        return suffix
    return f"{text[:max_chars]}{suffix}"


def _is_summary_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "system":
        return False
    return any("Conversation summary:" in payload for payload in _iter_message_strings(message))


def _extract_textual_turn(message: dict[str, Any]) -> str:
    parts = [payload.strip() for payload in _iter_message_strings(message) if payload.strip()]
    if not parts:
        return ""
    role = message.get("role")
    body = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if role:
        return f"{role}: {body}"
    return body


def summarise_history(
    messages: list[dict[str, Any]],
    target_tokens: int = HISTORY_SUMMARY_TARGET_TOKENS,
) -> list[dict[str, Any]]:
    if not messages:
        return messages

    filtered = [msg for msg in messages if not _is_summary_message(msg)]
    kept_indices: set[int] = set()
    user_kept = assistant_kept = 0
    for idx in range(len(filtered) - 1, -1, -1):
        role = filtered[idx].get("role")
        if role == "user" and user_kept < 2:
            kept_indices.add(idx)
            user_kept += 1
        elif role == "assistant" and assistant_kept < 2:
            kept_indices.add(idx)
            assistant_kept += 1
    for idx, msg in enumerate(filtered):
        if msg.get("role") == "system":
            kept_indices.add(idx)

    max_chars = max(0, target_tokens * 4)
    consumed = 0
    summary_parts: list[str] = []
    for idx, msg in enumerate(filtered):
        if idx in kept_indices:
            continue
        snippet = _extract_textual_turn(msg)
        if not snippet:
            continue
        remaining = max_chars - consumed
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = snippet[:remaining]
        summary_parts.append(snippet)
        consumed += len(snippet)

    if summary_parts:
        summary_text = "Conversation summary: " + " | ".join(summary_parts)
    else:
        summary_text = "Conversation summary: (no earlier messages summarised)"

    summary_message = {
        "role": "system",
        "content": [{"type": "input_text", "text": summary_text}],
    }

    final_messages = [summary_message]
    for idx, msg in enumerate(filtered):
        if idx in kept_indices:
            final_messages.append(msg)

    return final_messages


def _split_context_chunks(rag_block: str) -> list[str]:
    if not rag_block:
        return []
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n\s*\n", rag_block)]
    return [chunk for chunk in chunks if chunk]


def _rebuild_context_block(chunks: list[str]) -> str:
    if not chunks:
        return ""
    formatted = [chunk if chunk.endswith("\n") else f"{chunk}\n" for chunk in chunks]
    return "\n\n\n".join(formatted).strip()


def extract_rag_chunks(messages: list[dict[str, Any]]) -> list[str]:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not (isinstance(item, dict) and isinstance(item.get("text"), str)):
                continue
            match = CONTEXT_BLOCK_PATTERN.search(item["text"])
            if match:
                return _split_context_chunks(match.group(2).strip())
    return []



def clamp_llm_docs(
    docs: list[LlmDoc],
    max_chunks: int = RAG_CHUNKS_TO_LLM,
    max_chars_per_chunk: int = RAG_CHUNK_MAX_CHARS,
    max_total_chars: int = RAG_TOTAL_MAX_CHARS,
) -> list[LlmDoc]:
    if not docs or max_chunks <= 0 or max_total_chars <= 0:
        return []

    limited_docs = docs[: max(1, max_chunks)]
    clamped_docs: list[LlmDoc] = []
    consumed = 0

    for doc in limited_docs:
        remaining = max_total_chars - consumed
        if remaining <= 0:
            break
        content = (doc.content or "")[: max_chars_per_chunk]
        if len(content) > remaining:
            content = content[:remaining]
        if not content:
            continue
        clamped_docs.append(doc.copy(update={"content": content}))  # type: ignore[arg-type]
        consumed += len(content)

    return clamped_docs


def _apply_rag_limit(
    messages: list[dict[str, Any]], provided_chunks: list[str] | None
) -> tuple[str, int, int]:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for idx, item in enumerate(content):
            if not (isinstance(item, dict) and isinstance(item.get("text"), str)):
                continue
            text_block = item["text"]
            match = CONTEXT_BLOCK_PATTERN.search(text_block)
            if not match:
                continue
            context_body = match.group(2).strip()
            chunks = list(provided_chunks) if provided_chunks else _split_context_chunks(context_body)
            original = len(chunks)
            clamped_chunks = clamp_context(chunks)
            rag_block = _rebuild_context_block(clamped_chunks)
            new_text = f"{match.group(1)}{rag_block}\n{match.group(3).lstrip()}"
            item["text"] = f"{text_block[: match.start()]}{new_text}{text_block[match.end():]}"
            return rag_block, original, len(clamped_chunks)
    return "", 0, 0


def _apply_tool_limit(messages: list[dict[str, Any]]) -> list[str]:
    clamped_payloads: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            content = msg.get("content")
            if isinstance(content, str):
                clamped = clamp_tool_output(content)
                msg["content"] = clamped
                clamped_payloads.append(clamped)
            elif isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, str):
                        clamped = clamp_tool_output(part)
                        new_content.append(clamped)
                        clamped_payloads.append(clamped)
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        clamped = clamp_tool_output(part["text"])
                        part = dict(part)
                        part["text"] = clamped
                        new_content.append(part)
                        clamped_payloads.append(clamped)
                    else:
                        new_content.append(part)
                msg["content"] = new_content
        elif role == "assistant" and isinstance(msg.get("tool_calls"), list):
            for call in msg["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    clamped = clamp_tool_output(arguments)
                    function["arguments"] = clamped
                    clamped_payloads.append(clamped)
    return clamped_payloads


def enforce_budget(
    messages: list[dict[str, Any]],
    rag_chunks: list[str] | None = None,
    tool_payloads: list[str] | None = None,
    model_window_tokens: int = MODEL_WINDOW_TOKENS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    min_output_tokens: int = MIN_OUTPUT_TOKENS,
) -> tuple[list[dict[str, Any]], str, list[str], int]:
    adjusted_messages = deepcopy(messages)
    if rag_chunks is None:
        rag_chunks = extract_rag_chunks(adjusted_messages)
    rag_block, _, _ = _apply_rag_limit(adjusted_messages, rag_chunks)

    clamped_tool_payloads = _apply_tool_limit(adjusted_messages)
    if tool_payloads:
        # Ensure provided payloads are reported even if no tool messages in context
        clamped_tool_payloads.extend(clamp_tool_output(payload) for payload in tool_payloads)

    input_tokens = count_message_tokens(adjusted_messages)
    if input_tokens + max_output_tokens > model_window_tokens:
        adjusted_messages = summarise_history(
            adjusted_messages, target_tokens=HISTORY_SUMMARY_TARGET_TOKENS
        )
        input_tokens = count_message_tokens(adjusted_messages)

    if input_tokens + max_output_tokens > model_window_tokens:
        pruned: list[dict[str, Any]] = []
        for msg in adjusted_messages:
            pruned.append(msg)
            if count_message_tokens(pruned) + max_output_tokens > model_window_tokens:
                if msg.get("role") != "system":
                    pruned.pop()
        adjusted_messages = pruned
        input_tokens = count_message_tokens(adjusted_messages)

    while (
        input_tokens + max_output_tokens > model_window_tokens
        and max_output_tokens > min_output_tokens
    ):
        max_output_tokens = max(min_output_tokens, max_output_tokens - 128)

    if input_tokens + max_output_tokens > model_window_tokens:
        raise ValueError(
            "Context window exceeded even after applying budget constraints."
        )

    return adjusted_messages, rag_block, clamped_tool_payloads, max_output_tokens


__all__ = [
    "approx_tokens",
    "count_message_tokens",
    "clamp_context",
    "clamp_llm_docs",
    "clamp_tool_output",
    "summarise_history",
    "extract_rag_chunks",
    "enforce_budget",
]
