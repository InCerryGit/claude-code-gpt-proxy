from models import map_request_model


def test_model_mapping_prefers_provider():
    model = map_request_model("claude-3-sonnet-20240229", context="MODEL")
    assert model.startswith(("openai/", "gemini/", "azure/", "anthropic/"))
