import os
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Optional inbound auth token for this proxy
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")


def normalize_azure_openai_config(
    endpoint: Optional[str], api_version: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize Azure OpenAI endpoint and infer api-version from Responses URL."""
    if not endpoint:
        return endpoint, api_version

    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint, api_version

    normalized_endpoint = f"{parsed.scheme}://{parsed.netloc}"
    query_version = parse_qs(parsed.query).get("api-version", [None])[0]
    effective_version = api_version or query_version
    return normalized_endpoint, effective_version


AZURE_OPENAI_ENDPOINT, AZURE_API_VERSION = normalize_azure_openai_config(
    AZURE_OPENAI_ENDPOINT, AZURE_API_VERSION
)

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview",
    "gpt-4.1",  # Added default big model
    "gpt-4.1-mini",  # Added default small model
    "gpt-5.1"
    "gpt-5.1-chat"
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.2-chat",
    "gpt-5.2-codex"
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# List of Azure OpenAI models (deployment names)
AZURE_MODELS = []  # These are deployment-specific, so we'll validate at runtime
