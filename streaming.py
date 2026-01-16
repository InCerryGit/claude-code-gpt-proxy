import json
import time
import uuid
from typing import Any

from models import MessagesRequest, MessagesResponse


def _get_delta_payload(chunk):
    delta = None
    finish_reason = None

    if hasattr(chunk, "choices") and len(chunk.choices) > 0:
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", getattr(choice, "message", {}))
        finish_reason = getattr(choice, "finish_reason", None)
    else:
        chunk_type = None
        if hasattr(chunk, "type"):
            chunk_type = chunk.type
        elif isinstance(chunk, dict):
            chunk_type = chunk.get("type")

        if hasattr(chunk, "delta"):
            delta = chunk.delta
        elif isinstance(chunk, dict) and "delta" in chunk:
            delta = chunk["delta"]

        if delta is None and isinstance(chunk, dict) and chunk.get("text") is not None:
            delta = {"content": chunk.get("text")}

        if finish_reason is None:
            if hasattr(chunk, "finish_reason"):
                finish_reason = chunk.finish_reason
            elif isinstance(chunk, dict):
                finish_reason = chunk.get("finish_reason")

        if finish_reason is None and chunk_type in {
            "response.completed",
            "response.done",
            "response.finish",
            "response.finished",
            "response.error",
            "response.failed",
        }:
            finish_reason = "stop"

    return delta, finish_reason


def _extract_delta_content(delta):
    if hasattr(delta, "content"):
        return delta.content
    if isinstance(delta, dict) and "content" in delta:
        return delta["content"]
    if isinstance(delta, dict) and "text" in delta:
        return delta["text"]
    return None


def _extract_delta_tool_calls(delta):
    if hasattr(delta, "tool_calls"):
        return delta.tool_calls
    if isinstance(delta, dict) and "tool_calls" in delta:
        return delta["tool_calls"]
    return None


def _emit_event(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def _emit_message_start(message_id: str, response_model: str) -> str:
    message_data = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": response_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    return _emit_event("message_start", message_data)


def _emit_content_block_start(index: int, block: dict) -> str:
    return _emit_event(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": block},
    )


def _emit_content_block_delta(index: int, delta: dict) -> str:
    return _emit_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": delta},
    )


def _emit_content_block_stop(index: int) -> str:
    return _emit_event("content_block_stop", {"type": "content_block_stop", "index": index})


def _emit_message_delta(stop_reason: str, output_tokens: int) -> str:
    payload = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    return _emit_event("message_delta", payload)


def _emit_message_stop() -> str:
    return _emit_event("message_stop", {"type": "message_stop"})


def _map_stop_reason(finish_reason: str | None) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    return "end_turn"


def _stream_error_response(response_model: str, message_id: str, logger, error: Exception):
    import traceback

    error_traceback = traceback.format_exc()
    error_message = f"Error in streaming: {str(error)}\n\nFull traceback:\n{error_traceback}"
    logger.error(error_message)

    yield _emit_message_delta("error", 0)
    yield _emit_message_stop()
    logger.warning("STREAM DONE: model=%s message_id=%s", response_model, message_id)
    yield "data: [DONE]\n\n"


def _close_open_blocks(
    *,
    tool_index: int | None,
    last_tool_index: int,
    text_block_closed: bool,
    accumulated_text: str,
    text_sent: bool,
):
    if tool_index is not None:
        for i in range(1, last_tool_index + 1):
            yield _emit_content_block_stop(i)

    if not text_block_closed:
        if accumulated_text and not text_sent:
            yield _emit_content_block_delta(0, {"type": "text_delta", "text": accumulated_text})
        yield _emit_content_block_stop(0)


def _should_close_text_block(tool_index: int | None, text_block_closed: bool) -> bool:
    return tool_index is None and not text_block_closed


def _handle_tool_delta(
    tool_call,
    tool_index: int | None,
    last_tool_index: int,
):
    if isinstance(tool_call, dict) and "index" in tool_call:
        current_index = tool_call["index"]
    elif hasattr(tool_call, "index"):
        current_index = tool_call.index
    else:
        current_index = 0

    created_events = []
    anthropic_tool_index = None

    if tool_index is None or current_index != tool_index:
        tool_index = current_index
        last_tool_index += 1
        anthropic_tool_index = last_tool_index

        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            name = function.get("name", "") if isinstance(function, dict) else ""
            tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
        else:
            function = getattr(tool_call, "function", None)
            name = getattr(function, "name", "") if function else ""
            tool_id = getattr(tool_call, "id", f"toolu_{uuid.uuid4().hex[:24]}")

        created_events.append(
            _emit_content_block_start(
                anthropic_tool_index,
                {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
            )
        )

    arguments = None
    if isinstance(tool_call, dict) and "function" in tool_call:
        function = tool_call.get("function", {})
        arguments = function.get("arguments", "") if isinstance(function, dict) else ""
    elif hasattr(tool_call, "function"):
        function = getattr(tool_call, "function", None)
        arguments = getattr(function, "arguments", "") if function else ""

    if arguments:
        try:
            if isinstance(arguments, dict):
                args_json = json.dumps(arguments)
            else:
                json.loads(arguments)
                args_json = arguments
        except (json.JSONDecodeError, TypeError):
            args_json = arguments

        if anthropic_tool_index is not None:
            created_events.append(
                _emit_content_block_delta(
                    anthropic_tool_index,
                    {"type": "input_json_delta", "partial_json": args_json},
                )
            )

    return tool_index, last_tool_index, created_events


async def handle_streaming(response_generator, original_request: MessagesRequest, logger):
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        response_model = original_request.original_model or original_request.model

        yield _emit_message_start(message_id, response_model)
        yield _emit_content_block_start(0, {"type": "text", "text": ""})
        yield _emit_event("ping", {"type": "ping"})

        tool_index = None
        accumulated_text = ""
        text_sent = False
        text_block_closed = False
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        async for chunk in response_generator:
            try:
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    if hasattr(chunk.usage, "prompt_tokens"):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, "completion_tokens"):
                        output_tokens = chunk.usage.completion_tokens

                delta, finish_reason = _get_delta_payload(chunk)
                if delta is None:
                    continue

                delta_content = _extract_delta_content(delta)
                delta_tool_calls = _extract_delta_tool_calls(delta)

                if delta_content is not None and delta_content != "":
                    accumulated_text += delta_content
                    if tool_index is None and not text_block_closed:
                        text_sent = True
                        yield _emit_content_block_delta(
                            0, {"type": "text_delta", "text": delta_content}
                        )

                if delta_tool_calls:
                    if tool_index is None:
                        if text_sent and not text_block_closed:
                            text_block_closed = True
                            yield _emit_content_block_stop(0)
                        elif accumulated_text and not text_sent and not text_block_closed:
                            text_sent = True
                            yield _emit_content_block_delta(
                                0, {"type": "text_delta", "text": accumulated_text}
                            )
                            text_block_closed = True
                            yield _emit_content_block_stop(0)
                        elif not text_block_closed:
                            text_block_closed = True
                            yield _emit_content_block_stop(0)

                    if not isinstance(delta_tool_calls, list):
                        delta_tool_calls = [delta_tool_calls]

                    for tool_call in delta_tool_calls:
                        tool_index, last_tool_index, created_events = _handle_tool_delta(
                            tool_call, tool_index, last_tool_index
                        )
                        for event in created_events:
                            yield event

                if finish_reason and not has_sent_stop_reason:
                    has_sent_stop_reason = True

                    for event in _close_open_blocks(
                        tool_index=tool_index,
                        last_tool_index=last_tool_index,
                        text_block_closed=text_block_closed,
                        accumulated_text=accumulated_text,
                        text_sent=text_sent,
                    ):
                        yield event

                    stop_reason = _map_stop_reason(finish_reason)
                    yield _emit_message_delta(stop_reason, output_tokens)
                    yield _emit_message_stop()
                    yield "data: [DONE]\n\n"
                    return
            except Exception as e:
                logger.error("Error processing chunk: %s", str(e))
                continue

        if not has_sent_stop_reason:
            for event in _close_open_blocks(
                tool_index=tool_index,
                last_tool_index=last_tool_index,
                text_block_closed=text_block_closed,
                accumulated_text=accumulated_text,
                text_sent=text_sent,
            ):
                yield event

            yield _emit_message_delta("end_turn", output_tokens)
            yield _emit_message_stop()
            yield "data: [DONE]\n\n"

    except Exception as e:
        response_model = original_request.original_model or original_request.model
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        for event in _stream_error_response(response_model, message_id, logger, e):
            yield event


async def handle_synth_stream(anthropic_response: MessagesResponse):
    message_id = anthropic_response.id
    response_model = anthropic_response.model

    message_data = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": response_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": anthropic_response.usage.input_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": anthropic_response.usage.output_tokens,
            },
        },
    }
    yield _emit_event("message_start", message_data)

    content_blocks = anthropic_response.content or []
    block_index = 0

    for block in content_blocks:
        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

        if block_type == "text":
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
            yield _emit_content_block_start(block_index, {"type": "text", "text": ""})
            if text:
                yield _emit_content_block_delta(
                    block_index, {"type": "text_delta", "text": text}
                )
            yield _emit_content_block_stop(block_index)
            block_index += 1
        elif block_type == "tool_use":
            tool_id = block.get("id") if isinstance(block, dict) else getattr(block, "id", "")
            name = block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
            tool_input = block.get("input") if isinstance(block, dict) else getattr(block, "input", {})
            if tool_input is None:
                tool_input = {}
            yield _emit_content_block_start(
                block_index, {"type": "tool_use", "id": tool_id, "name": name, "input": {}}
            )
            yield _emit_content_block_delta(
                block_index,
                {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
            )
            yield _emit_content_block_stop(block_index)
            block_index += 1

    stop_reason = anthropic_response.stop_reason or "end_turn"
    yield _emit_message_delta(stop_reason, anthropic_response.usage.output_tokens)
    yield _emit_message_stop()
    yield "data: [DONE]\n\n"
