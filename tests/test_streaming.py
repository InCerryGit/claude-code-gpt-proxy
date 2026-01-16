import json
import logging
import types

import pytest

from models import MessagesRequest
from streaming import handle_streaming


class DummyDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class DummyChoice:
    def __init__(self, delta=None, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class DummyChunk:
    def __init__(self, delta=None, finish_reason=None, usage=None):
        self.choices = [DummyChoice(delta=delta, finish_reason=finish_reason)]
        self.usage = usage


class DummyUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


async def _fake_generator(chunks):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_handle_streaming_text_only():
    logger = logging.getLogger("tests")
    request = MessagesRequest(
        model="openai/gpt-4.1",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    chunks = [
        DummyChunk(delta=DummyDelta(content="Hello ")),
        DummyChunk(delta=DummyDelta(content="world"), finish_reason="stop", usage=DummyUsage(1, 2)),
    ]

    events = []
    async for event in handle_streaming(_fake_generator(chunks), request, logger):
        if event.startswith("event: "):
            payload = event.split("data: ", 1)[1].strip()
            events.append(json.loads(payload))

    event_types = {evt["type"] for evt in events}
    assert "message_start" in event_types
    assert "content_block_delta" in event_types
    assert "message_stop" in event_types


@pytest.mark.asyncio
async def test_handle_streaming_tool_calls():
    logger = logging.getLogger("tests")
    request = MessagesRequest(
        model="openai/gpt-4.1",
        max_tokens=32,
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    tool_call = {
        "index": 0,
        "id": "tool_1",
        "function": {"name": "calculator", "arguments": "{\"expr\": \"1+1\"}"},
    }

    chunks = [
        DummyChunk(delta=DummyDelta(tool_calls=[tool_call])),
        DummyChunk(delta=DummyDelta(tool_calls=[tool_call]), finish_reason="tool_calls"),
    ]

    events = []
    async for event in handle_streaming(_fake_generator(chunks), request, logger):
        if event.startswith("event: "):
            payload = event.split("data: ", 1)[1].strip()
            events.append(json.loads(payload))

    event_types = {evt["type"] for evt in events}
    assert "content_block_start" in event_types
    assert "message_delta" in event_types
    assert "message_stop" in event_types
