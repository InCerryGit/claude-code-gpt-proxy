from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import logging
import time
import uvicorn
import litellm

from config import (
    ANTHROPIC_AUTH_TOKEN,
    OPENAI_BASE_URL,
)
from conversions import convert_anthropic_to_litellm, convert_litellm_to_anthropic
from logging_setup import log_request_beautifully, setup_logging
from models import MessagesRequest, TokenCountRequest, TokenCountResponse
from routing import apply_provider_routing
from streaming import handle_streaming, handle_synth_stream
from utils import sanitize_for_json

setup_logging()
logger = logging.getLogger(__name__)

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

app = FastAPI()


@app.middleware("http")
async def enforce_inbound_auth(request: Request, call_next):
    if not ANTHROPIC_AUTH_TOKEN:
        return await call_next(request)

    public_paths = {"/", "/docs", "/openapi.json"}
    if request.url.path in public_paths:
        return await call_next(request)

    auth_header = request.headers.get("authorization")
    if not auth_header:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized", "message": "Missing Authorization header."},
        )

    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized", "message": "Invalid Authorization scheme."},
        )

    token = auth_header.removeprefix("Bearer ").strip()
    if token != ANTHROPIC_AUTH_TOKEN:
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden", "message": "Invalid auth token."},
        )

    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    method = request.method
    path = request.url.path
    logger.debug("Request: %s %s", method, path)
    response = await call_next(request)
    return response


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")

        display_model = original_model.split("/")[-1] if "/" in original_model else original_model

        logger.warning(
            "REQUEST: model=%s original_model=%s stream=%s",
            request.model,
            request.original_model,
            request.stream,
        )

        litellm_request = convert_anthropic_to_litellm(request, logger)
        apply_provider_routing(litellm_request, request.model, logger)

        logger.warning(
            "ROUTE: litellm_model=%s api_base=%s",
            litellm_request.get("model"),
            litellm_request.get("api_base"),
        )

        if (
            "openai" in litellm_request["model"] or "azure" in litellm_request["model"]
        ) and "messages" in litellm_request:
            logger.debug("Processing OpenAI/Azure model request: %s", litellm_request["model"])

            for i, msg in enumerate(litellm_request["messages"]):
                if "content" in msg:
                    if msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..."
                    elif isinstance(msg["content"], list):
                        try:
                            litellm_request["messages"][i]["content"] = json.dumps(msg["content"])
                        except Exception:
                            litellm_request["messages"][i]["content"] = str(msg["content"])

                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning("Removing unsupported field from message: %s", key)
                        del msg[key]

            for i, msg in enumerate(litellm_request["messages"]):
                logger.debug(
                    "Message %s format check - role: %s, content type: %s",
                    i,
                    msg.get("role"),
                    type(msg.get("content")),
                )

                if isinstance(msg.get("content"), list):
                    logger.warning(
                        "CRITICAL: Message %s still has list content after processing: %s",
                        i,
                        json.dumps(msg.get("content")),
                    )
                    litellm_request["messages"][i]["content"] = json.dumps(msg.get("content"))
                elif msg.get("content") is None:
                    logger.warning("Message %s has None content - replacing with placeholder", i)
                    litellm_request["messages"][i]["content"] = "..."

        logger.debug(
            "Request for model: %s, stream: %s",
            litellm_request.get("model"),
            litellm_request.get("stream", False),
        )

        if request.stream:
            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get("model"),
                len(litellm_request["messages"]),
                num_tools,
                200,
            )

            is_responses_model = (
                isinstance(litellm_request.get("model"), str)
                and litellm_request["model"].startswith("azure/responses/")
            )
            if is_responses_model:
                litellm_request["stream"] = False
                start_time = time.time()
                litellm_response = litellm.completion(**litellm_request)
                logger.debug(
                    "✅ RESPONSE RECEIVED (synth stream): Model=%s, Time=%.2fs",
                    litellm_request.get("model"),
                    time.time() - start_time,
                )
                anthropic_response = convert_litellm_to_anthropic(litellm_response, request, logger)
                return StreamingResponse(
                    handle_synth_stream(anthropic_response),
                    media_type="text/event-stream",
                )

            response_generator = await litellm.acompletion(**litellm_request)
            return StreamingResponse(
                handle_streaming(response_generator, request, logger),
                media_type="text/event-stream",
            )

        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            litellm_request.get("model"),
            len(litellm_request["messages"]),
            num_tools,
            200,
        )
        start_time = time.time()
        litellm_response = litellm.completion(**litellm_request)
        logger.debug(
            "✅ RESPONSE RECEIVED: Model=%s, Time=%.2fs",
            litellm_request.get("model"),
            time.time() - start_time,
        )

        anthropic_response = convert_litellm_to_anthropic(litellm_response, request, logger)
        return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback,
        }

        for attr in ["message", "status_code", "response", "llm_provider", "model"]:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        if hasattr(e, "__dict__"):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ["args", "__traceback__"]:
                    error_details[key] = str(value)

        sanitized_details = sanitize_for_json(error_details)
        logger.error("Error processing request: %s", json.dumps(sanitized_details, indent=2))

        error_message = f"Error: {str(e)}"
        if "message" in error_details and error_details["message"]:
            error_message += f"\nMessage: {error_details['message']}"
        if "response" in error_details and error_details["response"]:
            error_message += f"\nResponse: {error_details['response']}"

        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    try:
        original_model = request.original_model or request.model
        display_model = original_model.split("/")[-1] if "/" in original_model else original_model

        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking,
            ),
            logger,
        )

        try:
            from litellm import token_counter

            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get("model"),
                len(converted_request["messages"]),
                num_tools,
                200,
            )

            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }

            if request.model.startswith("openai/") and OPENAI_BASE_URL:
                token_counter_args["api_base"] = OPENAI_BASE_URL

            token_count = token_counter(**token_counter_args)
            return TokenCountResponse(input_tokens=token_count)
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            return TokenCountResponse(input_tokens=1000)

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error("Error counting tokens: %s\n%s", str(e), error_traceback)
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
