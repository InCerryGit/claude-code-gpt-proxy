import json
import uuid
from typing import Any, Dict, Optional, Union

from config import GEMINI_MODELS
from models import MessagesRequest, MessagesResponse, Usage
from utils import clean_gemini_schema, parse_tool_result_content


def _extract_text_from_system(system_block: Union[str, list]) -> Optional[str]:
    if isinstance(system_block, str):
        return system_block
    if isinstance(system_block, list):
        system_text = ""
        for block in system_block:
            if hasattr(block, "type") and block.type == "text":
                system_text += block.text + "\n\n"
            elif isinstance(block, dict) and block.get("type") == "text":
                system_text += block.get("text", "") + "\n\n"
        return system_text.strip() if system_text else None
    return None


def _strip_provider_prefix(model: str) -> str:
    if model.startswith("anthropic/"):
        return model[len("anthropic/") :]
    if model.startswith("openai/"):
        return model[len("openai/") :]
    if model.startswith("gemini/"):
        return model[len("gemini/") :]
    if model.startswith("azure/"):
        return model[len("azure/") :]
    return model


def convert_anthropic_to_litellm(anthropic_request: MessagesRequest, logger) -> Dict[str, Any]:
    messages = []

    if anthropic_request.system:
        system_text = _extract_text_from_system(anthropic_request.system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

    is_openai_like = anthropic_request.model.startswith(("openai/", "azure/"))
    for msg in anthropic_request.messages:
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
            continue

        if not is_openai_like:
            processed_content = []
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        processed_content.append({"type": "text", "text": block.text})
                    elif block.type == "image":
                        processed_content.append({"type": "image", "source": block.source})
                    elif block.type == "tool_use":
                        processed_content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )
                    elif block.type == "tool_result":
                        processed_content_block = {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else "",
                        }

                        if hasattr(block, "content"):
                            if isinstance(block.content, str):
                                processed_content_block["content"] = [
                                    {"type": "text", "text": block.content}
                                ]
                            elif isinstance(block.content, list):
                                processed_content_block["content"] = block.content
                            else:
                                processed_content_block["content"] = [
                                    {"type": "text", "text": str(block.content)}
                                ]
                        else:
                            processed_content_block["content"] = [{"type": "text", "text": ""}]

                        processed_content.append(processed_content_block)

            messages.append({"role": msg.role, "content": processed_content})
            continue

        if msg.role == "user":
            pending_text = ""

            def flush_user_text():
                nonlocal pending_text
                if pending_text.strip():
                    messages.append({"role": "user", "content": pending_text.strip()})
                pending_text = ""

            for block in content:
                block_type = None
                if hasattr(block, "type"):
                    block_type = block.type
                elif isinstance(block, dict):
                    block_type = block.get("type")

                if block_type == "text":
                    text_value = block.text if hasattr(block, "text") else block.get("text", "")
                    pending_text += f"{text_value}\n"
                elif block_type == "image":
                    pending_text += "[Image content - not displayed in text format]\n"
                elif block_type == "tool_result":
                    tool_use_id = (
                        block.tool_use_id if hasattr(block, "tool_use_id") else block.get("tool_use_id", "")
                    )
                    result_content = block.content if hasattr(block, "content") else block.get("content")
                    flush_user_text()
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": parse_tool_result_content(result_content),
                        }
                    )
                elif block_type == "tool_use":
                    tool_name = block.name if hasattr(block, "name") else block.get("name", "")
                    pending_text += f"[Tool use: {tool_name}]\n"

            flush_user_text()
        elif msg.role == "assistant":
            assistant_text = ""
            tool_calls = []

            for block in content:
                block_type = None
                if hasattr(block, "type"):
                    block_type = block.type
                elif isinstance(block, dict):
                    block_type = block.get("type")

                if block_type == "text":
                    text_value = block.text if hasattr(block, "text") else block.get("text", "")
                    assistant_text += f"{text_value}\n"
                elif block_type == "image":
                    assistant_text += "[Image content - not displayed in text format]\n"
                elif block_type == "tool_use":
                    tool_id = block.id if hasattr(block, "id") else block.get("id", f"tool_{uuid.uuid4()}")
                    name = block.name if hasattr(block, "name") else block.get("name", "")
                    tool_input = block.input if hasattr(block, "input") else block.get("input", {})
                    if tool_input is None:
                        tool_input = {}
                    if isinstance(tool_input, str):
                        arguments = tool_input
                    else:
                        try:
                            arguments = json.dumps(tool_input)
                        except Exception:
                            arguments = str(tool_input)
                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    )
                elif block_type == "tool_result":
                    result_content = block.content if hasattr(block, "content") else block.get("content")
                    assistant_text += f"{parse_tool_result_content(result_content)}\n"

            assistant_msg = {"role": "assistant", "content": assistant_text.strip()}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

    max_tokens = anthropic_request.max_tokens
    if anthropic_request.model.startswith("openai/") or anthropic_request.model.startswith("gemini/"):
        max_tokens = min(max_tokens, 16384)
        logger.debug(
            "Capping max_tokens to 16384 for OpenAI/Gemini model (original value: %s)",
            anthropic_request.max_tokens,
        )

    litellm_request: Dict[str, Any] = {
        "model": anthropic_request.model,
        "messages": messages,
        "stream": anthropic_request.stream,
    }

    if anthropic_request.model.startswith("azure/"):
        deployment_name = anthropic_request.model.split("/", 1)[1]
        model_name = deployment_name.lower()

        if "gpt-5" in model_name or "o3" in model_name:
            litellm_request["model"] = f"azure/responses/{deployment_name}"
            litellm_request["max_completion_tokens"] = max_tokens
            logger.debug(
                "Using max_completion_tokens=%s for Azure model: %s", max_tokens, model_name
            )
            logger.debug(
                "Skipping temperature parameter for %s (only supports default temperature=1)",
                model_name,
            )
        else:
            litellm_request["max_tokens"] = max_tokens
            litellm_request["temperature"] = anthropic_request.temperature
            logger.debug(
                "Using max_tokens=%s and temperature=%s for Azure model: %s",
                max_tokens,
                anthropic_request.temperature,
                model_name,
            )
    else:
        litellm_request["max_tokens"] = max_tokens
        litellm_request["temperature"] = anthropic_request.temperature

    if anthropic_request.thinking and anthropic_request.model.startswith("anthropic/"):
        litellm_request["thinking"] = anthropic_request.thinking

    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences

    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p

    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k

    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            if hasattr(tool, "dict"):
                tool_dict = tool.dict()
            else:
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                    logger.error("Could not convert tool to dict: %s", tool)
                    continue

            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                logger.debug("Cleaning schema for Gemini tool: %s", tool_dict.get("name"))
                input_schema = clean_gemini_schema(input_schema, logger)

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema,
                },
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools

    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, "dict"):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice

        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]},
            }
        else:
            litellm_request["tool_choice"] = "auto"

    return litellm_request


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any],
    original_request: MessagesRequest,
    logger,
) -> MessagesResponse:
    try:
        clean_model = original_request.original_model or original_request.model
        clean_model = _strip_provider_prefix(clean_model)

        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, "content") else ""
            tool_calls = message.tool_calls if message and hasattr(message, "tool_calls") else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            try:
                response_dict = (
                    litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
                )
            except AttributeError:
                try:
                    response_dict = (
                        litellm_response.model_dump()
                        if hasattr(litellm_response, "model_dump")
                        else litellm_response.__dict__
                    )
                except AttributeError:
                    response_dict = {
                        "id": getattr(litellm_response, "id", f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, "choices", [{}]),
                        "usage": getattr(litellm_response, "usage", {}),
                    }

            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = (
                choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            )
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        content = []

        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        if tool_calls:
            logger.debug("Processing tool calls: %s", tool_calls)

            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug("Processing tool call %s: %s", idx, tool_call)

                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse tool arguments as JSON: %s", arguments)
                        arguments = {"raw": arguments}

                logger.debug("Adding tool_use block: id=%s, name=%s, input=%s", tool_id, name, arguments)

                content.append({"type": "tool_use", "id": tool_id, "name": name, "input": arguments})

        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        if not content:
            content.append({"type": "text", "text": ""})

        response_model = original_request.original_model or original_request.model

        anthropic_response = MessagesResponse(
            id=response_id,
            model=response_model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

        return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)

        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.original_model or original_request.model,
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": f"Error converting response: {str(e)}. Please check server logs.",
                }
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )
