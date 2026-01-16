from typing import Any, Dict

from config import (
    ANTHROPIC_API_KEY,
    AZURE_API_VERSION,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    USE_VERTEX_AUTH,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
)


def apply_provider_routing(litellm_request: Dict[str, Any], model: str, logger) -> None:
    if model.startswith("openai/"):
        litellm_request["api_key"] = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            litellm_request["api_base"] = OPENAI_BASE_URL
            logger.debug(
                "Using OpenAI API key and custom base URL %s for model: %s",
                OPENAI_BASE_URL,
                model,
            )
        else:
            logger.debug("Using OpenAI API key for model: %s", model)
    elif model.startswith("gemini/"):
        if USE_VERTEX_AUTH:
            litellm_request["vertex_project"] = VERTEX_PROJECT
            litellm_request["vertex_location"] = VERTEX_LOCATION
            litellm_request["custom_llm_provider"] = "vertex_ai"
            logger.debug(
                "Using Gemini ADC with project=%s, location=%s and model: %s",
                VERTEX_PROJECT,
                VERTEX_LOCATION,
                model,
            )
        else:
            litellm_request["api_key"] = GEMINI_API_KEY
            logger.debug("Using Gemini API key for model: %s", model)
    elif model.startswith("azure/"):
        litellm_request["api_key"] = AZURE_OPENAI_API_KEY
        litellm_request["api_base"] = AZURE_OPENAI_ENDPOINT
        if AZURE_API_VERSION:
            litellm_request["api_version"] = AZURE_API_VERSION
        logger.debug("Using Azure OpenAI API key for model: %s", model)
    else:
        litellm_request["api_key"] = ANTHROPIC_API_KEY
        logger.debug("Using Anthropic API key for model: %s", model)
