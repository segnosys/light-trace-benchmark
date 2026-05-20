"""
Tests for AnthropicBackend — request building, SSE parsing, and the
cache_control marker placement that implements the protocol-correct prefix
caching behavior.
"""
import json

import pytest

from legacy.backends import AnthropicBackend
from legacy.schema import InferencePayload


@pytest.fixture
def backend():
    return AnthropicBackend(
        base_url="https://api.anthropic.com",
        api_key="sk-test",
        model_name="claude-3-5-sonnet-20241022",
        tokenizer_name=None,  # skip tokenizer load
        enable_prompt_caching=True,
    )


# ---------- request body ----------

class TestRequestBody:

    def test_endpoint_url(self, backend):
        req = InferencePayload(prompt="hi", stream=True)
        assert backend.build_endpoint_url(req) == "https://api.anthropic.com/v1/messages"

    def test_headers(self, backend):
        h = backend.build_headers()
        assert h["x-api-key"] == "sk-test"
        assert h["anthropic-version"] == "2023-06-01"
        assert h["Content-Type"] == "application/json"

    def test_prompt_form_single_block_when_no_cacheable_prefix(self, backend):
        """Plain `prompt` field: single user content block, marker on it."""
        req = InferencePayload(prompt="hello world", stream=True, max_tokens=8)
        body = backend.build_request_body(req)

        assert body["model"] == "claude-3-5-sonnet-20241022"
        assert body["max_tokens"] == 8
        assert body["stream"] is True
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

        content = body["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["text"] == "hello world"
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_prompt_with_cacheable_prefix_is_split(self, backend):
        """When prompt starts with cacheable_prefix, content splits in two."""
        prefix = "shared filler " * 100
        suffix = " unique #42"
        req = InferencePayload(
            prompt=prefix + suffix,
            cacheable_prefix=prefix,
            stream=True, max_tokens=8,
        )
        body = backend.build_request_body(req)

        content = body["messages"][0]["content"]
        assert len(content) == 2
        # Marker on prefix block ONLY.
        assert content[0]["text"] == prefix
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert content[1]["text"] == suffix
        assert "cache_control" not in content[1]

    def test_cacheable_prefix_equals_whole_prompt(self, backend):
        """If suffix is empty, only one block is produced."""
        prefix = "all of this is cacheable"
        req = InferencePayload(
            prompt=prefix, cacheable_prefix=prefix, stream=True, max_tokens=8,
        )
        body = backend.build_request_body(req)

        content = body["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_cacheable_prefix_does_not_match_prompt_falls_back(self, backend):
        """If prompt doesn't startwith cacheable_prefix, treat as no-split."""
        req = InferencePayload(
            prompt="totally different prompt",
            cacheable_prefix="this prefix was not actually used",
            stream=True, max_tokens=8,
        )
        body = backend.build_request_body(req)
        content = body["messages"][0]["content"]
        assert len(content) == 1
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_messages_with_system_marker_on_system_block(self, backend):
        """Role=system messages are hoisted into top-level `system`; marker goes there."""
        req = InferencePayload(messages=[
            {"role": "system", "content": "you are an assistant"},
            {"role": "user", "content": "hello"},
        ], stream=True, max_tokens=8)
        body = backend.build_request_body(req)

        # System hoisted, with marker
        assert "system" in body
        sys_blocks = body["system"]
        assert len(sys_blocks) == 1
        assert sys_blocks[0]["text"] == "you are an assistant"
        assert sys_blocks[0]["cache_control"] == {"type": "ephemeral"}

        # User message unmodified by cache marker
        assert len(body["messages"]) == 1
        user = body["messages"][0]
        assert user["role"] == "user"
        # Content normalized to a list of blocks
        assert isinstance(user["content"], list)
        # No marker on the user block when system block is present
        assert all("cache_control" not in b for b in user["content"])

    def test_messages_as_json_string_parsed(self, backend):
        """`messages` field accepts a JSON string for OpenAI-compat shape."""
        msgs = json.dumps([{"role": "user", "content": "hi"}])
        req = InferencePayload(messages=msgs, stream=True, max_tokens=8)
        body = backend.build_request_body(req)
        assert body["messages"][0]["role"] == "user"

    def test_caching_disabled_drops_marker(self):
        b = AnthropicBackend(
            base_url="https://api.anthropic.com",
            api_key="sk", model_name="claude-3-5-sonnet-20241022",
            tokenizer_name=None, enable_prompt_caching=False,
        )
        req = InferencePayload(prompt="hello", stream=True, max_tokens=8)
        body = b.build_request_body(req)
        assert "cache_control" not in body["messages"][0]["content"][0]

    def test_temperature_and_top_p_passthrough(self, backend):
        req = InferencePayload(
            prompt="hi", stream=True, max_tokens=8,
            temperature=0.3, top_p=0.9,
        )
        body = backend.build_request_body(req)
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.9

    def test_no_prompt_no_messages_raises(self, backend):
        with pytest.raises(ValueError):
            backend.build_request_body(InferencePayload(stream=True))


# ---------- SSE response parsing ----------

class TestResponseParse:

    def test_message_start_carries_cache_tokens(self, backend):
        req = InferencePayload(prompt="hi", stream=True)
        frag = backend.decode_response_chunk({
            "type": "message_start",
            "message": {"usage": {
                "input_tokens": 1500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 1200,
                "output_tokens": 0,
            }},
        }, req)
        assert frag.text == ""
        assert frag.prompt_usage_tokens == 1500
        assert frag.cached_input_tokens == 1200
        assert frag.cache_creation_input_tokens == 0

    def test_content_block_delta_text(self, backend):
        frag = backend.decode_response_chunk({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        }, InferencePayload(prompt="x"))
        assert frag.text == "Hello"

    def test_message_delta_updates_output_tokens(self, backend):
        frag = backend.decode_response_chunk(
            {"type": "message_delta", "usage": {"output_tokens": 42}},
            InferencePayload(prompt="x"),
        )
        assert frag.text == ""
        assert frag.usage_tokens == 42

    def test_non_streaming_response_collected(self, backend):
        frag = backend.decode_response_chunk({
            "content": [
                {"type": "text", "text": "Hi"},
                {"type": "text", "text": " there."},
            ],
            "usage": {
                "input_tokens": 50,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 0,
                "output_tokens": 6,
            },
        }, InferencePayload(prompt="x"))
        assert frag.text == "Hi there."
        assert frag.prompt_usage_tokens == 50
        assert frag.usage_tokens == 6
        assert frag.cached_input_tokens == 40

    def test_error_event_returns_none(self, backend):
        assert backend.decode_response_chunk(
            {"error": {"type": "rate_limit"}}, InferencePayload(prompt="x")
        ) is None

    def test_unknown_event_does_not_break_stream(self, backend):
        frag = backend.decode_response_chunk(
            {"type": "ping"}, InferencePayload(prompt="x")
        )
        # Should produce an empty fragment, not raise.
        assert frag.text == ""
        assert frag.usage_tokens is None
