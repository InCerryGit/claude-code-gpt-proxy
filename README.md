# Anthropic API Proxy for OpenAI, Gemini, Azure

This is an Azure-focused fork/variant of the Anthropic-compatible proxy, primarily to ensure better Azure OpenAI compatibility.

Use Anthropic-compatible clients (including Claude Code) with OpenAI, Gemini, Azure OpenAI, or Anthropic backends through a single FastAPI proxy powered by LiteLLM.

## Features

- Anthropic-compatible `/v1/messages` and `/v1/messages/count_tokens` endpoints
- Model mapping for `haiku`/`sonnet` to provider-specific models
- OpenAI, Gemini (AI Studio or Vertex ADC), Azure OpenAI, or direct Anthropic routing
- Streaming and non-streaming support
- Optional OpenAI base URL override

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- API keys for the providers you intend to use

### Run from source

```bash
git clone https://github.com/InCerryGit/claude-code-gpt-proxy
cd claude-code-gpt-proxy
cp .env.example .env
```

Edit `.env` with your keys and preferences, then start the server:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

### Run with Docker

Build the image locally (the upstream image is no longer usable):

```bash
docker build -t claude-code-gpt-proxy:latest .
```

Run with port mapping:

```bash
docker run -d --name claude-code-gpt-proxy --env-file .env -p 8082:8082 claude-code-gpt-proxy:latest
```

## Using Claude Code  

```bash
npm install -g @anthropic-ai/claude-code
```

Edit the configuration file to use a proxy:  
```json
# Edit or create the `settings.json` file  
# On MacOS & Linux: `~/.claude/settings.json`  
# On Windows: `User Directory/.claude/settings.json`  
# Add or modify the env field  
# Make sure to replace `your_anthropic_auth_token` with your configured `ANTHROPIC_AUTH_TOKEN`  
{  
  "env": {  
    "ANTHROPIC_AUTH_TOKEN": "your_anthropic_auth_token",  
    "ANTHROPIC_BASE_URL": "http://localhost:8082",  
    "API_TIMEOUT_MS": "3000000",  
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1  
  }  
}
# Then edit or create the `.claude.json` file  
# On MacOS & Linux: `~/.claude.json`  
# On Windows: `User Directory/.claude.json`  
# Add the `hasCompletedOnboarding` parameter  
{  
  "hasCompletedOnboarding": true  
}
```

Then run Claude Code:  
```bash
claude
```

## Configuration

Configure via `.env` or environment variables.

### Core keys

- `OPENAI_API_KEY`
- `GEMINI_API_KEY` (for Google AI Studio)
- `ANTHROPIC_API_KEY` (only required for direct Anthropic routing)

### Inbound auth (optional)

Set `ANTHROPIC_AUTH_TOKEN` to require clients to send `Authorization: Bearer <token>` for all API requests. If unset, the proxy allows anonymous access.

Example request:

```bash
curl http://localhost:8082/v1/messages \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}'
```

### Provider routing

- `PREFERRED_PROVIDER`: `openai` (default), `google`, `azure`, or `anthropic`
- `BIG_MODEL`: model for `sonnet`/`opus` (default `gpt-4.1`)
- `SMALL_MODEL`: model for `haiku` (default `gpt-4.1-mini`)

### Gemini Vertex AI (ADC)

- `USE_VERTEX_AUTH=true`
- `VERTEX_PROJECT`
- `VERTEX_LOCATION`

### OpenAI base URL

- `OPENAI_BASE_URL` (optional)

### Azure OpenAI

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_API_VERSION`

### Logging (optional)

The proxy writes logs to `LOG_FILE` and rotates daily at midnight when enabled.

- `LOG_LEVEL` (default: `INFO`)
- `LOG_FILE` (if set, enable file logging; supports `strftime` tokens, e.g. `./logs/proxy-%Y%m%d.log`)
- `LOG_FILE_BACKUP_COUNT` (default: `7`)
- `LOG_FILE_PATTERN` (strftime pattern for rotated filenames, e.g. `proxy-%Y%m%d.log`)

**Notes**
- `LOG_FILE` is rendered with `strftime` when the process starts, so `%Y%m%d` expands to today’s date.
- For a stable base filename with date-stamped rotations, set `LOG_FILE=./logs/proxy.log` and `LOG_FILE_PATTERN=proxy-%Y%m%d.log`.

## Model Mapping

The proxy maps Anthropic-style model names to provider models:

- `haiku` → `SMALL_MODEL`
- `sonnet` / `opus` → `BIG_MODEL`

Mapping depends on `PREFERRED_PROVIDER`:

- `openai`: prefixes with `openai/`
- `google`: prefixes with `gemini/` if the model is in the known Gemini list; otherwise falls back to OpenAI
- `azure`: prefixes with `azure/`
- `anthropic`: passes through with `anthropic/` and does not remap

### Supported model name prefixes

The proxy automatically adds prefixes when missing:

- OpenAI: `openai/`
- Gemini: `gemini/`
- Azure: `azure/`
- Anthropic: `anthropic/`

## Examples

### Prefer OpenAI (default)

```dotenv
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-gemini-key" # optional fallback
# PREFERRED_PROVIDER="openai"
# BIG_MODEL="gpt-4.1"
# SMALL_MODEL="gpt-4.1-mini"
```

### Prefer Gemini (AI Studio)

```dotenv
GEMINI_API_KEY="your-gemini-key"
OPENAI_API_KEY="your-openai-key" # fallback
PREFERRED_PROVIDER="google"
# BIG_MODEL="gemini-2.5-pro"
# SMALL_MODEL="gemini-2.5-flash"
```

### Prefer Gemini (Vertex ADC)

```dotenv
OPENAI_API_KEY="your-openai-key" # fallback
PREFERRED_PROVIDER="google"
USE_VERTEX_AUTH=true
VERTEX_PROJECT="your-gcp-project-id"
VERTEX_LOCATION="us-central1"
```

### Direct Anthropic

```dotenv
ANTHROPIC_API_KEY="sk-ant-..."
PREFERRED_PROVIDER="anthropic"
```

### Azure OpenAI

```dotenv
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-azure-key"
AZURE_API_VERSION="2024-10-01"
PREFERRED_PROVIDER="azure"
BIG_MODEL="gpt-4o"   # deployment name
SMALL_MODEL="gpt-4o-mini" # deployment name
```

## Tests

```bash
python tests.py
```

Single-scope tests:

```bash
python tests.py --no-streaming
python tests.py --streaming-only
python tests.py --simple
python tests.py --tools
```

## Contributing

Pull requests are welcome.