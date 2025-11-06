import copy


from onyx.chat.models import LlmDoc
from onyx.configs.constants import DocumentSource

from onyx.chat.token_budget import (
    clamp_context,
    clamp_llm_docs,
    clamp_tool_output,
    count_message_tokens,
    enforce_budget,
    extract_rag_chunks,
    summarise_history,
)
from onyx.configs.token_budget_configs import (
    MIN_OUTPUT_TOKENS,
    RAG_CHUNK_MAX_CHARS,
    RAG_CHUNKS_TO_LLM,
    RAG_TOTAL_MAX_CHARS,
)
from onyx.prompts.constants import GENERAL_SEP_PAT


def _build_context_chunks(num_chunks: int, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    for idx in range(1, num_chunks + 1):
        body = "A" * chunk_size
        chunk = (
            f"DOCUMENT {idx}: Sample Document\n"
            "Source: Test\n"
            f"```{body}```\n\n\n"
        )
        chunks.append(chunk)
    return chunks


def _wrap_context(chunks: list[str]) -> str:
    context_body = "".join(chunks)
    return (
        "Refer to the following context.\n"
        f"CONTEXT:\n{GENERAL_SEP_PAT}\n"
        f"{context_body}"
        f"{GENERAL_SEP_PAT}\n\n"
        "Task\n\nQUESTION\nWhat should I do?"
    )


def test_clamp_context_limits() -> None:
    chunks = ["X" * 10_000 for _ in range(10)]
    clamped = clamp_context(chunks)
    assert len(clamped) <= RAG_CHUNKS_TO_LLM
    assert all(len(chunk) <= RAG_CHUNK_MAX_CHARS for chunk in clamped)
    assert sum(len(chunk) for chunk in clamped) <= RAG_TOTAL_MAX_CHARS


def test_clamp_llm_docs_limits() -> None:
    docs = [
        LlmDoc(
            document_id=str(i),
            content="X" * 10_000,
            blurb="",
            semantic_identifier=f"Doc {i}",
            source_type=DocumentSource.FILE,
            metadata={},
            updated_at=None,
            link=None,
            source_links=None,
            match_highlights=None,
        )
        for i in range(10)
    ]

    clamped = clamp_llm_docs(docs)

    assert len(clamped) <= RAG_CHUNKS_TO_LLM
    assert all(len(doc.content) <= RAG_CHUNK_MAX_CHARS for doc in clamped)
    assert sum(len(doc.content) for doc in clamped) <= RAG_TOTAL_MAX_CHARS


def test_summarise_history_keeps_recent_turns() -> None:
    messages: list[dict] = [
        {"role": "system", "content": [{"type": "input_text", "text": "system"}]}
    ]
    for i in range(10):
        messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"user message {i}"}],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "input_text", "text": f"assistant reply {i}"}],
            }
        )

    summarised = summarise_history(messages)

    assert summarised[0]["role"] == "system"
    assert "Conversation summary:" in summarised[0]["content"][0]["text"]
    user_messages = [msg for msg in summarised if msg["role"] == "user"]
    assistant_messages = [msg for msg in summarised if msg["role"] == "assistant"]
    assert len(user_messages) == 2
    assert len(assistant_messages) == 2
    assert count_message_tokens(summarised) < count_message_tokens(messages)


def test_clamp_tool_output_appends_suffix() -> None:
    payload = "Z" * 50_000
    clamped = clamp_tool_output(payload)
    assert len(clamped) <= 2_000 + len("[truncated, 48000 chars omitted]")
    assert clamped.endswith("chars omitted]")
    assert "[truncated" in clamped


def test_enforce_budget_reduces_window_and_clamps_context() -> None:
    base_messages: list[dict] = [
        {"role": "system", "content": [{"type": "input_text", "text": "system"}]}
    ]
    # Older turns to trigger summarisation
    for i in range(4):
        base_messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"older user {i}"}],
            }
        )
        base_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "input_text", "text": f"older assistant {i}"}],
            }
        )

    context_chunks = _build_context_chunks(8, 2_400)
    user_with_context = _wrap_context(context_chunks)
    base_messages.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_with_context}],
        }
    )

    original = copy.deepcopy(base_messages)
    enforced_messages, rag_block, tool_payloads, max_output = enforce_budget(
        base_messages,
        rag_chunks=extract_rag_chunks(base_messages),
        model_window_tokens=1_800,
        max_output_tokens=1_500,
        min_output_tokens=MIN_OUTPUT_TOKENS,
    )

    assert max_output <= 1_500
    assert max_output >= MIN_OUTPUT_TOKENS

    final_chunks = extract_rag_chunks(enforced_messages)
    assert len(final_chunks) <= RAG_CHUNKS_TO_LLM
    assert all(len(chunk) <= RAG_CHUNK_MAX_CHARS for chunk in final_chunks)
    assert sum(len(chunk) for chunk in final_chunks) <= RAG_TOTAL_MAX_CHARS

    assert count_message_tokens(enforced_messages) <= count_message_tokens(original)
    summary_present = any(
        msg.get("role") == "system"
        and any("Conversation summary:" in part.get("text", "") for part in msg.get("content", []) if isinstance(part, dict))
        for msg in enforced_messages
    )
    assert summary_present
    assert tool_payloads == []
