import json
from typing import Any

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.messages.tool import ToolMessage
from pydantic import BaseModel

from onyx.chat.token_budget import clamp_tool_output
from onyx.natural_language_processing.utils import BaseTokenizer

# Langchain has their own version of pydantic which is version 1


def build_tool_message(
    tool_call: ToolCall, tool_content: str | list[str | dict[str, Any]]
) -> ToolMessage:
    if isinstance(tool_content, str):
        tool_content = clamp_tool_output(tool_content)
    elif isinstance(tool_content, list):
        clamped_parts: list[str | dict[str, Any]] = []
        for part in tool_content:
            if isinstance(part, str):
                clamped_parts.append(clamp_tool_output(part))
            elif isinstance(part, dict):
                new_part = dict(part)
                text_value = new_part.get("text")
                if isinstance(text_value, str):
                    new_part["text"] = clamp_tool_output(text_value)
                clamped_parts.append(new_part)
            else:
                clamped_parts.append(part)
        tool_content = clamped_parts
    return ToolMessage(
        tool_call_id=tool_call["id"] or "",
        name=tool_call["name"],
        content=tool_content,
    )


class ToolCallSummary(BaseModel):
    tool_call_request: AIMessage
    tool_call_result: ToolMessage

    # This is a workaround to allow arbitrary types in the model
    # TODO: Remove this once we have a better solution
    class Config:
        arbitrary_types_allowed = True


def tool_call_tokens(
    tool_call_summary: ToolCallSummary, llm_tokenizer: BaseTokenizer
) -> int:
    request_tokens = len(
        llm_tokenizer.encode(
            json.dumps(tool_call_summary.tool_call_request.tool_calls[0]["args"])
        )
    )
    result_tokens = len(
        llm_tokenizer.encode(json.dumps(tool_call_summary.tool_call_result.content))
    )

    return request_tokens + result_tokens
