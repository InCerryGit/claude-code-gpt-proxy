# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

### Run the server (dev)
```
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

### Run tests
```
python tests.py
```

Single-scope test modes (from tests.py):
```
python tests.py --no-streaming
python tests.py --streaming-only
python tests.py --simple
python tests.py --tools
```

### Docker (runtime)
Build locally (upstream image no longer usable):
```
docker build -t claude-code-proxy:latest .
```
Run:
```
docker run -d --name claude-code-proxy --env-file .env -p 8082:8082 claude-code-proxy:latest
```

### Use with Claude Code
```
npm install -g @anthropic-ai/claude-code
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## Architecture overview

### Core flow (server.py)
- **FastAPI proxy** that accepts Anthropic `/v1/messages` and `/v1/messages/count_tokens` requests and forwards to LiteLLM.
- **Model mapping** is handled in Pydantic validators for `MessagesRequest` and `TokenCountRequest` (maps `haiku`/`sonnet` to provider-specific models; keeps `original_model` for responses).
- **Request conversion** in `convert_anthropic_to_litellm` builds OpenAI-style messages, normalizes tool use/tool results, and caps `max_tokens` for OpenAI/Gemini.
- **Provider routing** in `create_message` injects keys and provider-specific parameters for OpenAI, Gemini (API key or Vertex ADC), Azure, or Anthropic.
- **Response conversion** in `convert_litellm_to_anthropic` converts LiteLLM responses back to Anthropic message format, including tool blocks for Claude models.
- **Streaming bridge**: `handle_streaming` converts LiteLLM streaming chunks to Anthropic SSE. For Azure Responses models, it performs a non-streaming request and **synthesizes** SSE via `handle_synth_stream`.

### Files to know
- `server.py`: single-file application that defines all request/response handling and streaming logic.
- `tests.py`: end-to-end tests that compare proxy responses against Anthropic API (streaming and non-streaming).
- `Dockerfile`: runtime container setup using `uv` and `uvicorn`.

## Configuration surface (from README/.env.example)
- Core API keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`.
- Provider selection: `PREFERRED_PROVIDER` with `BIG_MODEL` / `SMALL_MODEL` mappings.
- Vertex AI: `USE_VERTEX_AUTH`, `VERTEX_PROJECT`, `VERTEX_LOCATION`.
- OpenAI base override: `OPENAI_BASE_URL`.
- Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_API_VERSION`.
- Inbound auth (optional): `ANTHROPIC_AUTH_TOKEN`.
- Logging (optional): `LOG_LEVEL`, `LOG_FILE`, `LOG_FILE_BACKUP_COUNT`, `LOG_FILE_PATTERN`.
