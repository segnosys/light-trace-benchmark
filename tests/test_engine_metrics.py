"""
Tests for the Prometheus /metrics scraper.

Covers the line parser (pure-Python, no network) and the polling loop
against a fake aiohttp server.
"""
import asyncio
from typing import List

import pytest
from aiohttp import web

from lightrace.engine_metrics import (
    EngineMetricsPoller,
    parse_prometheus_text,
)


# ---------- parser ----------


class TestParser:
    def test_picks_up_simple_lines(self):
        body = "\n".join([
            "# HELP some_unknown_metric description",
            "# TYPE sglang:num_running_reqs gauge",
            "sglang:num_running_reqs 16",
            "sglang:num_waiting_reqs 4",
            "irrelevant_metric 999",
        ])
        out = parse_prometheus_text(body)
        assert out["sglang:num_running_reqs"] == 16.0
        assert out["sglang:num_waiting_reqs"] == 4.0
        assert "irrelevant_metric" not in out

    def test_ignores_labels(self):
        body = 'sglang:gen_throughput{model="kimi"} 781.4'
        out = parse_prometheus_text(body)
        assert out["sglang:gen_throughput"] == pytest.approx(781.4)

    def test_handles_float_and_scientific(self):
        body = "\n".join([
            "sglang:gen_throughput 1.5e2",
            "sglang:token_usage 0.85",
        ])
        out = parse_prometheus_text(body)
        assert out["sglang:gen_throughput"] == 150.0
        assert out["sglang:token_usage"] == 0.85

    def test_ignores_garbage_lines(self):
        body = "not_a_metric_line\n\n# comment\nsglang:num_running_reqs 7\n"
        assert parse_prometheus_text(body) == {"sglang:num_running_reqs": 7.0}

    def test_only_keeps_known_names(self):
        body = "unknown_metric 42\nsglang:num_running_reqs 8"
        out = parse_prometheus_text(body)
        assert "unknown_metric" not in out
        assert out["sglang:num_running_reqs"] == 8.0


# ---------- end-to-end via aiohttp test server ----------


def _make_fake_metrics_server(samples: List[str]):
    counter = {"i": 0}

    async def handler(request):
        i = counter["i"]
        counter["i"] = min(i + 1, len(samples) - 1)
        return web.Response(text=samples[i], content_type="text/plain")

    app = web.Application()
    app.router.add_get("/metrics", handler)
    return app


@pytest.mark.asyncio
async def test_poller_collects_samples_over_time(aiohttp_server):
    samples = [
        "sglang:num_running_reqs 16\nsglang:gen_throughput 750",
        "sglang:num_running_reqs 18\nsglang:gen_throughput 800",
        "sglang:num_running_reqs 20\nsglang:gen_throughput 820",
    ]
    server = await aiohttp_server(_make_fake_metrics_server(samples))
    url = str(server.make_url("/metrics"))

    poller = EngineMetricsPoller(url, interval_s=0.05)
    await poller.start()
    await asyncio.sleep(0.25)
    await poller.stop()

    snap = poller.snapshot()
    assert "sglang:num_running_reqs" in snap
    assert "sglang:gen_throughput" in snap

    running = snap["sglang:num_running_reqs"]
    assert running["samples"] >= 2
    assert running["max"] == 20.0
    assert running["latest"] == 20.0


@pytest.mark.asyncio
async def test_poller_recovers_from_failed_fetch(aiohttp_server):
    state = {"hits": 0}

    async def handler(request):
        state["hits"] += 1
        if state["hits"] < 3:
            return web.Response(status=500)
        return web.Response(text="sglang:num_running_reqs 8", content_type="text/plain")

    app = web.Application()
    app.router.add_get("/metrics", handler)
    server = await aiohttp_server(app)
    url = str(server.make_url("/metrics"))

    poller = EngineMetricsPoller(url, interval_s=0.05)
    await poller.start()
    await asyncio.sleep(0.3)
    await poller.stop()

    snap = poller.snapshot()
    assert snap.get("sglang:num_running_reqs", {}).get("samples", 0) >= 1


@pytest.mark.asyncio
async def test_poller_stop_is_idempotent(aiohttp_server):
    app = _make_fake_metrics_server(["sglang:num_running_reqs 1"])
    server = await aiohttp_server(app)
    poller = EngineMetricsPoller(str(server.make_url("/metrics")), interval_s=0.05)
    await poller.start()
    await asyncio.sleep(0.1)
    await poller.stop()
    await poller.stop()


@pytest.mark.asyncio
async def test_snapshot_omits_metrics_with_no_samples(aiohttp_server):
    app = _make_fake_metrics_server(["sglang:num_running_reqs 5"])
    server = await aiohttp_server(app)
    poller = EngineMetricsPoller(str(server.make_url("/metrics")), interval_s=0.05)
    await poller.start()
    await asyncio.sleep(0.15)
    await poller.stop()
    snap = poller.snapshot()
    assert "sglang:num_running_reqs" in snap
    assert "sglang:num_waiting_reqs" not in snap
    assert "vllm:num_requests_running" not in snap
