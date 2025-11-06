from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import cast

import pytest

from agents.run import Runner

from onyx.agents.agent_sdk.sync_agent_stream_adapter import SyncAgentStream
from onyx.configs import model_configs


class _FakeRunResult:
    def __init__(self, messages: Sequence[dict[str, Any]]) -> None:
        self._messages = list(messages)

    def to_input_list(self) -> list[dict[str, Any]]:
        return list(self._messages)


class _FakeStreamResult:
    def __init__(self, messages: Sequence[dict[str, Any]], events: list[Any]) -> None:
        self._messages = list(messages)
        self._events = list(events)
        self.cancel_called = False

    def to_input_list(self) -> list[dict[str, Any]]:
        return list(self._messages)

    async def stream_events(self):
        for ev in self._events:
            yield ev

    async def cancel(self) -> None:
        self.cancel_called = True


def test_sync_agent_stream_non_stream_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """When streaming is disabled, ensure Runner.run is used and events are suppressed."""
    calls: dict[str, int] = {"run": 0, "run_streamed": 0}
    messages = [{"role": "user", "content": []}]

    def fake_run(
        agent: Any,
        input: Sequence[dict[str, Any]],
        *,
        context: Any,
        max_turns: int,
    ) -> _FakeRunResult:
        calls["run"] += 1
        return _FakeRunResult(input)

    def fake_run_streamed(*args: Any, **kwargs: Any) -> _FakeStreamResult:
        calls["run_streamed"] += 1
        return _FakeStreamResult([], [])

    monkeypatch.setattr(model_configs, "ONYX_DISABLE_AGENT_STREAMING", True)
    monkeypatch.setattr(Runner, "run", staticmethod(fake_run))
    monkeypatch.setattr(Runner, "run_streamed", staticmethod(fake_run_streamed))

    stream = SyncAgentStream(agent=cast(Any, object()), input=messages, context=None)
    collected = list(stream)

    assert collected == []
    assert calls["run"] == 1
    assert calls["run_streamed"] == 0
    assert stream.streamed is not None
    assert stream.streamed.to_input_list() == messages


def test_sync_agent_stream_streaming_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """When streaming is enabled, ensure Runner.run_streamed is used and events flow through."""
    calls: dict[str, int] = {"run": 0, "run_streamed": 0}
    messages = [{"role": "user", "content": []}]
    events = ["delta-1", "delta-2"]
    stream_result = _FakeStreamResult(messages, events)

    def fake_run(*args: Any, **kwargs: Any) -> _FakeRunResult:
        calls["run"] += 1
        return _FakeRunResult([])

    def fake_run_streamed(*args: Any, **kwargs: Any) -> _FakeStreamResult:
        calls["run_streamed"] += 1
        return stream_result

    monkeypatch.setattr(model_configs, "ONYX_DISABLE_AGENT_STREAMING", False)
    monkeypatch.setattr(Runner, "run", staticmethod(fake_run))
    monkeypatch.setattr(Runner, "run_streamed", staticmethod(fake_run_streamed))

    stream = SyncAgentStream(agent=cast(Any, object()), input=messages, context=None)
    collected = list(stream)

    assert collected == events
    assert calls["run"] == 0
    assert calls["run_streamed"] == 1
    assert stream.streamed is stream_result
    assert stream.streamed.to_input_list() == messages


def test_llm_factory_sets_stream_false_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onyx.llm.factory as factory

    monkeypatch.setattr(factory, "ONYX_DISABLE_AGENT_STREAMING", True)
    monkeypatch.setattr(factory, "model_is_reasoning_model", lambda model, provider: False)
    monkeypatch.setattr(factory, "build_llm_extra_headers", lambda headers: {})
    monkeypatch.setattr(factory, "_build_provider_extra_headers", lambda provider, custom_config: {})

    class DummyLitellmModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class DummyReasoning:
        def __init__(self, summary: str, *args: Any, **kwargs: Any) -> None:
            self.summary = summary

    class DummyModelSettings:
        def __init__(
            self,
            temperature: float,
            extra_headers: dict[str, str] | None,
            extra_args: dict[str, Any],
            reasoning: DummyReasoning,
        ) -> None:
            self.temperature = temperature
            self.extra_headers = extra_headers
            self.extra_args = extra_args
            self.reasoning = reasoning

    monkeypatch.setattr(factory, "LitellmModel", DummyLitellmModel)
    monkeypatch.setattr(factory, "Reasoning", DummyReasoning)
    monkeypatch.setattr(factory, "ModelSettings", DummyModelSettings)
    monkeypatch.setattr(factory, "is_true_openai_model", lambda provider, model: False)

    _, settings = factory._get_llm_model_and_settings(
        provider="custom",
        model="example-model",
        timeout=5,
        temperature=0.3,
    )

    assert settings.extra_args.get("stream") is False
