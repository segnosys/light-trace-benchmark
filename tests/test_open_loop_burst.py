"""
Tests for OpenLoopBurstDriver.

The contrast with BatchedDriver is load-bearing: BatchedDriver waits for the
slowest request in each batch before firing the next, so a slow tail silently
slows the whole sweep. OpenLoopBurstDriver fires on a fixed cadence regardless
of completion.
"""
import asyncio
import time
from typing import List

import pytest

from legacy.backends import BaseBackend
from legacy.load_driver import BatchedDriver, OpenLoopBurstDriver
from legacy.schema import InferencePayload, LatencyProfile, ResultEntry


class _SlowFakeBackend(BaseBackend):
    """Backend whose every request takes `delay` seconds."""

    def __init__(self, *, delay: float = 0.5):
        super().__init__(base_url="http://x/v1", api_key="", model_name="x")
        self.delay = delay
        self.calls: List[float] = []  # wall-clock fire time of every call

    def build_endpoint_url(self, request):
        return ""

    def build_request_body(self, request):
        return {}

    def build_headers(self):
        return {}

    def decode_response_chunk(self, data, request):
        return None

    async def execute_call(self, request: InferencePayload) -> ResultEntry:
        self.calls.append(time.perf_counter())
        await asyncio.sleep(self.delay)
        return ResultEntry(
            model="x",
            request=request,
            content="ok",
            metrics=LatencyProfile(output_token_count=1),
            success=True,
        )


def _reqs(n: int) -> List[InferencePayload]:
    return [InferencePayload(prompt=f"p{i}", stream=False, max_tokens=8) for i in range(n)]


@pytest.mark.asyncio
async def test_open_loop_fires_on_cadence_not_completion():
    """
    Each request takes 0.3s; burst_interval=0.1s, concurrency=2, 3 bursts.
    If we were closed-loop we'd take ~0.9s (3 × slowest). Open-loop should
    fire bursts within ~0.2s + drain time, so total ~0.5s.
    """
    backend = _SlowFakeBackend(delay=0.3)
    driver = OpenLoopBurstDriver(
        backend=backend, concurrency=2, max_num_burst=3, burst_interval=0.1,
    )

    t0 = time.perf_counter()
    results, intervals = await driver.run_load(_reqs(6))
    elapsed = time.perf_counter() - t0

    assert len(results) == 6
    assert all(r.success for r in results)
    # 3 bursts at 0.1s cadence → fire times near 0, 0.1, 0.2; last completes
    # at ~0.2 + 0.3 (its own delay) = ~0.5s. Allow wide margin for CI noise.
    assert elapsed < 0.7, (
        f"Open-loop should be ~0.5s; got {elapsed:.3f}s — driver may be "
        "blocking on completion instead of firing on cadence."
    )


@pytest.mark.asyncio
async def test_open_loop_fires_all_burst_requests():
    backend = _SlowFakeBackend(delay=0.05)
    driver = OpenLoopBurstDriver(
        backend=backend, concurrency=2, max_num_burst=3, burst_interval=0.1,
    )
    await driver.run_load(_reqs(6))
    assert len(backend.calls) == 6


@pytest.mark.asyncio
async def test_open_loop_burst_cadence_is_close_to_interval():
    """First → second burst fire should be within ~50% of burst_interval."""
    backend = _SlowFakeBackend(delay=0.05)
    driver = OpenLoopBurstDriver(
        backend=backend, concurrency=1, max_num_burst=3, burst_interval=0.2,
    )
    await driver.run_load(_reqs(3))
    assert len(backend.calls) == 3
    interval_1 = backend.calls[1] - backend.calls[0]
    interval_2 = backend.calls[2] - backend.calls[1]
    assert 0.1 < interval_1 < 0.35, f"interval_1={interval_1:.3f}s, expected ~0.2"
    assert 0.1 < interval_2 < 0.35, f"interval_2={interval_2:.3f}s, expected ~0.2"


@pytest.mark.asyncio
async def test_open_loop_respects_max_num_burst():
    backend = _SlowFakeBackend(delay=0.05)
    driver = OpenLoopBurstDriver(
        backend=backend, concurrency=2, max_num_burst=2, burst_interval=0.05,
    )
    results, _ = await driver.run_load(_reqs(10))
    assert len(results) == 4  # 2 bursts × 2 concurrency
    assert len(backend.calls) == 4


@pytest.mark.asyncio
async def test_open_loop_returns_per_burst_intervals():
    backend = _SlowFakeBackend(delay=0.05)
    driver = OpenLoopBurstDriver(
        backend=backend, concurrency=2, max_num_burst=3, burst_interval=0.15,
    )
    _, intervals = await driver.run_load(_reqs(6))
    assert intervals is not None
    assert len(intervals) == 3
    assert 0.1 < intervals[0] < 0.35
    assert 0.1 < intervals[1] < 0.35


@pytest.mark.asyncio
async def test_closed_loop_burst_blocks_for_stragglers():
    """Regression-shield for BatchedDriver: it DOES block on slow tails."""
    backend = _SlowFakeBackend(delay=0.2)
    driver = BatchedDriver(
        backend=backend, concurrency=2, max_num_burst=2, burst_interval=0.05,
    )
    t0 = time.perf_counter()
    await driver.run_load(_reqs(4))
    elapsed = time.perf_counter() - t0
    # ~0.4s expected (2 × 0.2s, since per-batch sleep < request time so no-op).
    assert 0.35 < elapsed < 0.6, (
        f"Closed-loop burst should block on stragglers; got {elapsed:.3f}s."
    )
