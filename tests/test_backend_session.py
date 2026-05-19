"""
Tests for the shared `aiohttp.ClientSession` lifecycle on BaseBackend.

The previous implementation created a fresh `ClientSession` for every
`execute_call()`, paying TCP+TLS handshake on every request and producing
noisy "Unclosed client session" warnings on shutdown. The refactored
backend lazily creates ONE session per backend instance and `.close()`s
it explicitly from the runner.
"""
import asyncio

import aiohttp

from legacy.backends import OpenAIBackend, SGLangBackend


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_session_lazily_created():
    """No session exists until the first request fires."""
    b = OpenAIBackend(base_url="http://x/v1", api_key="", model_name="x")
    assert b._session is None


def test_session_is_reused_across_get_session_calls():
    """The same session object should be returned every time."""
    b = SGLangBackend(base_url="http://x/v1", api_key="", model_name="x")

    async def _two_gets():
        s1 = await b._get_session()
        s2 = await b._get_session()
        assert s1 is s2
        assert isinstance(s1, aiohttp.ClientSession)
        await b.close()

    _run(_two_gets())


def test_close_clears_session_and_is_idempotent():
    b = OpenAIBackend(base_url="http://x/v1", api_key="", model_name="x")

    async def _open_close():
        s = await b._get_session()
        assert not s.closed
        await b.close()
        assert b._session is None
        # Idempotent: second close is a no-op
        await b.close()
        assert b._session is None

    _run(_open_close())


def test_close_after_session_already_closed_is_safe():
    b = OpenAIBackend(base_url="http://x/v1", api_key="", model_name="x")

    async def _close_twice():
        s = await b._get_session()
        await s.close()  # close the underlying session directly
        # backend.close() should still tidy up its reference without raising
        await b.close()
        assert b._session is None

    _run(_close_twice())


def test_get_session_after_close_creates_a_new_one():
    """Edge case: another request after close() should reopen, not crash."""
    b = OpenAIBackend(base_url="http://x/v1", api_key="", model_name="x")

    async def _reopen():
        s1 = await b._get_session()
        await b.close()
        s2 = await b._get_session()
        assert s1 is not s2
        assert not s2.closed
        await b.close()

    _run(_reopen())


def test_connector_is_configured_with_pool_limits():
    """TCPConnector cap should match the documented 256 max conns."""
    b = OpenAIBackend(base_url="http://x/v1", api_key="", model_name="x")

    async def _check():
        s = await b._get_session()
        # aiohttp's TCPConnector exposes the limit via the `.limit` attr
        assert s.connector.limit == 256
        await b.close()

    _run(_check())
