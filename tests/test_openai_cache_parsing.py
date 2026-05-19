"""
Tests for the OpenAI / sglang / vllm cache-token parsing path:
`usage.prompt_tokens_details.cached_tokens` ends up on
FragmentInfo.cached_input_tokens and ultimately on LatencyProfile.
"""
from lightrace.backends import OpenAIBackend, SGLangBackend
from lightrace.schema import InferencePayload


def _chat_request():
    """Request that takes the chat-completion path (matches the fake chunk shape)."""
    return InferencePayload(messages=[{"role": "user", "content": "hi"}], stream=True)


def test_openai_decode_extracts_cached_tokens():
    b = OpenAIBackend(base_url="http://x/v1", api_key="", model_name="x")
    chunk = {
        "choices": [{"delta": {"content": " world"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }
    frag = b.decode_response_chunk(chunk, _chat_request())
    assert frag.text == " world"
    assert frag.usage_tokens == 5
    assert frag.prompt_usage_tokens == 100
    assert frag.cached_input_tokens == 80


def test_openai_decode_without_cache_details_returns_none_cached():
    b = OpenAIBackend(base_url="http://x/v1", api_key="", model_name="x")
    chunk = {
        "choices": [{"delta": {"content": "hi"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    frag = b.decode_response_chunk(chunk, _chat_request())
    assert frag.cached_input_tokens is None


def test_sglang_inherits_openai_cache_parsing():
    """SGLang's OpenAI-compat path serializes cache info the same way."""
    b = SGLangBackend(base_url="http://x/v1", api_key="", model_name="x")
    chunk = {
        "choices": [{"delta": {"content": "yo"}}],
        "usage": {
            "prompt_tokens": 4000,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 3200},
        },
    }
    frag = b.decode_response_chunk(chunk, _chat_request())
    assert frag.cached_input_tokens == 3200


def test_openai_completions_path_with_text_chunk():
    """Completion-style requests parse `choices[].text` not `delta.content`."""
    b = OpenAIBackend(base_url="http://x/v1", api_key="", model_name="x")
    chunk = {
        "choices": [{"text": " world"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 60},
        },
    }
    frag = b.decode_response_chunk(chunk, InferencePayload(prompt="hi", stream=True))
    assert frag.text == " world"
    assert frag.cached_input_tokens == 60
