import json
from typing import Any


def parse_tool_result_content(content: Any) -> str:
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except Exception:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except Exception:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except Exception:
            return str(content)

    try:
        return str(content)
    except Exception:
        return "Unparseable content"


def clean_gemini_schema(schema: Any, logger) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(
                    "Removing unsupported format '%s' for string type in Gemini schema.",
                    schema["format"],
                )
                schema.pop("format")

        for key, value in list(schema.items()):
            schema[key] = clean_gemini_schema(value, logger)
    elif isinstance(schema, list):
        return [clean_gemini_schema(item, logger) for item in schema]
    return schema


def sanitize_for_json(obj: Any) -> Any:
    """Recursively clean object so it can be JSON serialized."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return sanitize_for_json(obj.__dict__)
    if hasattr(obj, "text"):
        return str(obj.text)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
