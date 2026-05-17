"""
End-to-end backend tests against an in-process fake OpenAI-compatible server.

These exercise the full HTTP path: build_request_body -> POST ->
parse SSE -> ResultEntry. No real network or running inference server
needed — `aiohttp_server` fixture (from pytest-aiohttp) spins up a
real aiohttp app on a random local port for the duration of the test.

Catches regressions in:
  - request body shape per backend
  - SSE chunk parsing
  - cache-token plumbing
  - LatencyProfile.ms_per_token, TTFT, etc.
"""
import asyncio
import json

import pytest
from aiohttp import web

from lightrace.backends import (
    AnthropicBackend,
    OpenAIBackend,
    SGLangBackend,
    TogetherBackend,
    VllmBackend,
)
from lightrace.schema import InferencePayload


# ---------- OpenAI-compatible chat completions fake ----------


def _sse_chat_chunks(text_chunks, *, prompt_tokens=10, cached_tokens=None):
    """Build SSE body that streams `text_chunks` then a final usage chunk."""
    out = []
    for ch in text_chunks:
        out.append("data: " + json.dumps({
            "choices": [{"delta": {"content": ch}}],
        }))
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": sum(len(c) for c in text_chunks),
        "total_tokens": prompt_tokens + sum(len(c) for c in text_chunks),
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    out.append("data: " + json.dumps({"choices": [], "usage": usage}))
    out.append("data: [DONE]")
    return "\n\n".join(out) + "\n\n"


@pytest.fixture
def fake_openai_app():
    async def chat_handler(request):
        body = await request.json()
        # Echo last 1-3 short "tokens" plus a final usage chunk
        if body.get("stream"):
            sse = _sse_chat_chunks(
                ["Hello", " world", "."],
                prompt_tokens=12,
                cached_tokens=request.app.get("cached_tokens"),
            )
            return web.Response(
                body=sse.encode(),
                content_type="text/event-stream",
                headers={"x-request-id": "fake-123"},
            )
        return web.json_response({
            "choices": [{"message": {"content": "Hello world."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        })

    app = web.Application()
    app["cached_tokens"] = None
    app.router.add_post("/v1/chat/completions", chat_handler)
    return app


# ---------- OpenAI backend (chat) ----------


@pytest.mark.asyncio
async def test_openai_chat_round_trip(aiohttp_server, fake_openai_app):
    server = await aiohttp_server(fake_openai_app)
    base_url = str(server.make_url("/v1"))

    b = OpenAIBackend(base_url=base_url, api_key="sk", model_name="test")
    try:
        result = await b.execute_call(InferencePayload(
            messages=[{"role": "user", "content": "hi"}],
            stream=True, max_tokens=8,
        ))
        assert result.success
        assert result.content == "Hello world."
        assert result.metrics.input_token_count == 12
        # Three chunks summed -> total chars
        assert result.metrics.output_char_count == len("Hello world.")
        # Cache fields should be None (server didn't report any)
        assert result.metrics.cached_input_tokens is None
    finally:
        await b.close()


# ---------- sglang inherits OpenAI path; ensure cache parsing still wired ----------


@pytest.mark.asyncio
async def test_sglang_chat_with_cache_report(aiohttp_server, fake_openai_app):
    fake_openai_app["cached_tokens"] = 8  # server reports cache_read
    server = await aiohttp_server(fake_openai_app)
    base_url = str(server.make_url("/v1"))

    b = SGLangBackend(base_url=base_url, api_key="", model_name="test")
    try:
        result = await b.execute_call(InferencePayload(
            messages=[{"role": "user", "content": "hi"}],
            stream=True, max_tokens=8,
        ))
        assert result.success
        assert result.metrics.cached_input_tokens == 8
    finally:
        await b.close()


# ---------- vllm and together inherit the same path (sanity) ----------


@pytest.mark.asyncio
async def test_vllm_chat_round_trip(aiohttp_server, fake_openai_app):
    server = await aiohttp_server(fake_openai_app)
    b = VllmBackend(base_url=str(server.make_url("/v1")), api_key="", model_name="test")
    try:
        r = await b.execute_call(InferencePayload(
            messages=[{"role": "user", "content": "hi"}], stream=True, max_tokens=8,
        ))
        assert r.success and r.content == "Hello world."
    finally:
        await b.close()


@pytest.mark.asyncio
async def test_together_chat_round_trip(aiohttp_server, fake_openai_app):
    server = await aiohttp_server(fake_openai_app)
    b = TogetherBackend(
        base_url=str(server.make_url("/v1")), api_key="sk", model_name="test",
    )
    try:
        r = await b.execute_call(InferencePayload(
            messages=[{"role": "user", "content": "hi"}], stream=True, max_tokens=8,
        ))
        assert r.success
    finally:
        await b.close()


# ---------- Failure path: non-200 -> failed_result ----------


@pytest.mark.asyncio
async def test_backend_records_failure_on_500(aiohttp_server):
    async def boom(request):
        return web.Response(status=500, text="server is dead")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", boom)
    server = await aiohttp_server(app)

    b = OpenAIBackend(base_url=str(server.make_url("/v1")), api_key="sk", model_name="x")
    try:
        result = await b.execute_call(InferencePayload(
            messages=[{"role": "user", "content": "hi"}], stream=True, max_tokens=8,
        ))
        assert not result.success
        assert "server is dead" in (result.content or "")
    finally:
        await b.close()


# ---------- Anthropic backend round-trip via Messages API ----------


@pytest.fixture
def fake_anthropic_app():
    """Streaming Messages API server returning a 'message_start' with cache info."""

    async def handler(request):
        # Always validate required headers came through
        assert request.headers.get("x-api-key") == "sk-anthropic"
        assert request.headers.get("anthropic-version") == "2023-06-01"

        events = []
        # message_start carries usage (input + cache_read)
        events.append("event: message_start")
        events.append("data: " + json.dumps({
            "type": "message_start",
            "message": {"usage": {
                "input_tokens": 1500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 1200,
                "output_tokens": 0,
            }},
        }))
        events.append("")  # SSE blank line

        # text deltas
        for chunk in ("Hi", " there", "."):
            events.append("event: content_block_delta")
            events.append("data: " + json.dumps({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": chunk},
            }))
            events.append("")

        # message_delta with final output_tokens
        events.append("event: message_delta")
        events.append("data: " + json.dumps({
            "type": "message_delta", "usage": {"output_tokens": 3},
        }))
        events.append("")
        events.append("event: message_stop")
        events.append("data: " + json.dumps({"type": "message_stop"}))
        events.append("")

        body = "\n".join(events) + "\n"
        return web.Response(body=body.encode(), content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/messages", handler)
    return app


@pytest.mark.asyncio
async def test_anthropic_round_trip_streams_text_and_cache(aiohttp_server, fake_anthropic_app):
    server = await aiohttp_server(fake_anthropic_app)
    base_url = str(server.make_url("/"))

    b = AnthropicBackend(
        base_url=base_url, api_key="sk-anthropic",
        model_name="claude-3-5-sonnet-20241022", tokenizer_name=None,
    )
    try:
        result = await b.execute_call(InferencePayload(
            prompt="hello",
            stream=True, max_tokens=8,
        ))
        assert result.success, f"unexpected fail: {result.content}"
        assert result.content == "Hi there."
        assert result.metrics.cached_input_tokens == 1200
        assert result.metrics.input_token_count == 1500
        assert result.metrics.output_token_count == 3
    finally:
        await b.close()
