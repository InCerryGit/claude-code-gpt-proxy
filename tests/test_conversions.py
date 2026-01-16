import logging

from conversions import convert_anthropic_to_litellm, convert_litellm_to_anthropic
from models import MessagesRequest


class DummyTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


def _base_request():
    return MessagesRequest(
        model="openai/gpt-4.1",
        max_tokens=128,
        messages=[{"role": "user", "content": "Hello"}],
    )


def test_convert_anthropic_to_litellm_tools():
    logger = logging.getLogger("tests")
    req = _base_request()
    req.tools = [
        DummyTool(
            name="calculator",
            description="Eval math",
            input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
    ]

    result = convert_anthropic_to_litellm(req, logger)

    assert result["tools"][0]["type"] == "function"
    assert result["tools"][0]["function"]["name"] == "calculator"
    assert "parameters" in result["tools"][0]["function"]


def test_convert_litellm_to_anthropic_tool_calls():
    logger = logging.getLogger("tests")
    req = _base_request()
    litellm_response = {
        "id": "msg_test",
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tool_1",
                            "function": {
                                "name": "calculator",
                                "arguments": "{\"expr\": \"2+2\"}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }

    result = convert_litellm_to_anthropic(litellm_response, req, logger)

    assert result.stop_reason == "tool_use"
    assert result.content[0].type == "tool_use" or result.content[0]["type"] == "tool_use"
