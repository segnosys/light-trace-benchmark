"""
Polls Prometheus /metrics from the inference server alongside the client run.

When the user passes `--engine_metrics_url http://server:port/metrics`,
agent-bench launches a background poller that samples sglang/vllm's exported
metrics every second for the duration of a run. The collected samples are
summarized into the BenchmarkReport so users can cross-check client-observed
TPOT against server-reported running/waiting queues, batch sizes, and
KV-cache usage.

This is intentionally lightweight — we don't pull in `prometheus_client` for
a parser; the exposition format is line-based and trivial to handle with
regexes for the small set of names we care about.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp


# Subset of metric names agent-bench surfaces. Both sglang and vllm expose
# OpenAI-compat /metrics; sglang names below match v0.5.x.
# Add new keys here when you want them in the summary.
METRIC_NAMES = (
    "sglang:num_running_reqs",
    "sglang:num_waiting_reqs",
    "sglang:gen_throughput",
    "sglang:token_usage",  # KV cache fraction
    "sglang:cache_hit_rate",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:generation_tokens_total",
)


_LINE_RE = re.compile(
    # Match "name{labels} value" or "name value" — labels optional.
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[0-9.eE+\-]+)\s*$"
)


@dataclass
class MetricSeries:
    """All samples collected for one metric name over the run."""
    name: str
    values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    @property
    def latest(self) -> Optional[float]:
        return self.values[-1] if self.values else None

    @property
    def mean(self) -> Optional[float]:
        return sum(self.values) / len(self.values) if self.values else None

    @property
    def max(self) -> Optional[float]:
        return max(self.values) if self.values else None


def parse_prometheus_text(body: str) -> Dict[str, float]:
    """
    Parse Prometheus exposition text into name -> latest value.

    Ignores HELP/TYPE comment lines, drops labels (we want the aggregate),
    and only keeps the names in METRIC_NAMES. Multiple labelled series
    under the same name will collapse to the last one seen.
    """
    out: Dict[str, float] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name not in METRIC_NAMES:
            continue
        try:
            out[name] = float(m.group("value"))
        except ValueError:
            continue
    return out


class EngineMetricsPoller:
    """
    Polls a Prometheus /metrics endpoint at a fixed cadence.

    Lifecycle:
        poller = EngineMetricsPoller(url, interval_s=1.0)
        await poller.start()
        ... run benchmark ...
        await poller.stop()
        snapshot = poller.snapshot()
    """

    def __init__(
        self,
        url: str,
        *,
        interval_s: float = 1.0,
        timeout_s: float = 2.0,
    ):
        self.url = url
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.series: Dict[str, MetricSeries] = {n: MetricSeries(n) for n in METRIC_NAMES}
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._session: Optional[aiohttp.ClientSession] = None
        self.fetch_errors = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_s)
        )
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=self.interval_s + 1.0)
        except asyncio.TimeoutError:
            self._task.cancel()
        finally:
            self._task = None
            if self._session is not None:
                await self._session.close()
                self._session = None

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._poll_once()
            except Exception as exc:  # noqa: BLE001 — log + keep going
                self.fetch_errors += 1
                logging.debug(f"engine metrics poll error: {exc}")
            # Wait either interval_s OR until stop fires, whichever first.
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass

    async def _poll_once(self) -> None:
        assert self._session is not None
        async with self._session.get(self.url) as resp:
            if resp.status != 200:
                self.fetch_errors += 1
                return
            body = await resp.text()
        ts = time.time()
        parsed = parse_prometheus_text(body)
        for name, value in parsed.items():
            s = self.series.setdefault(name, MetricSeries(name))
            s.values.append(value)
            s.timestamps.append(ts)

    def snapshot(self) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Return a flat dict: metric_name -> {mean, max, latest, samples}.
        Only includes series that saw at least one sample.
        """
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for name, series in self.series.items():
            if not series.values:
                continue
            out[name] = {
                "mean": series.mean,
                "max": series.max,
                "latest": series.latest,
                "samples": float(len(series.values)),
            }
        return out
