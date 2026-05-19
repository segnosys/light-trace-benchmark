import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Iterator, List, Optional, Tuple

from legacy.backends import BaseBackend
from legacy.schema import InferencePayload, ResultEntry


class LoadDriver(ABC):
    def __init__(self, backend: BaseBackend) -> None:
        self.backend = backend

    async def execute_call(self, request: InferencePayload) -> ResultEntry:
        return await self.backend.execute_call(request)

    @abstractmethod
    async def run_load(
        self, requests: List[InferencePayload]
    ) -> Tuple[List[ResultEntry], Optional[List[float]]]:
        pass


def generate_arrival_gaps(
    qps_level: float, duration_s: int, random_seed: int = 42, distribution: str = "uniform"
) -> List[float]:
    gap_times: List[float] = []
    mean_wait_s: float = 1.0 / qps_level

    random.seed(random_seed)

    for _ in range(int(qps_level * duration_s)):
        if distribution == "exponential":
            wait_s = random.expovariate(1.0 / mean_wait_s)
        elif distribution == "uniform":
            wait_s = random.uniform(0, 2 * mean_wait_s)
        elif distribution == "constant":
            wait_s = mean_wait_s
        else:
            raise ValueError(f"Invalid traffic rate distribution {distribution}")

        gap_times.append(wait_s)
    return gap_times


async def dispatch_with_delay(
    *,
    requests: List[InferencePayload],
    intervals: List[float],
) -> AsyncGenerator[InferencePayload, None]:
    for interval, request in zip(intervals, requests):
        yield request

        await asyncio.sleep(interval)


class RateBasedDriver(LoadDriver):
    """
    Generate traffic for x seconds with y requests per second on average.
    The random waiting interval between requests follows poisson distribution.

    Illustration for 10 QPS that runs for 1 second, each dot represents firing
    of one request:

    0s|-.----.---------.------.-.---------.--.------.-----.--.---------|1s
    """

    def __init__(
        self,
        *,
        backend: BaseBackend,
        qps_level: float,
        duration_s: int,
        random_seed: int = 42,
        distribution: str = "uniform",
    ) -> None:
        super().__init__(backend=backend)
        self.qps_level = qps_level
        self.duration_s = duration_s
        self.intervals = generate_arrival_gaps(qps_level, duration_s, random_seed, distribution)

    async def run_load(
        self, requests: List[InferencePayload]
    ) -> Tuple[List[ResultEntry], Optional[List[float]]]:
        assert len(requests) >= int(
            self.qps_level * self.duration_s
        ), f"Number of provided requests {len(requests)} is not enough for traffic at "
        f"{self.qps_level} QPS for {self.duration_s} seconds."

        tasks: List[asyncio.Task] = []

        async for request in dispatch_with_delay(
            requests=requests,
            intervals=self.intervals,
        ):
            tasks.append(asyncio.create_task(self.backend.execute_call(request)))

        return await asyncio.gather(*tasks), None


class ParallelWorkerDriver(LoadDriver):
    def __init__(
        self,
        *,
        backend: BaseBackend,
        concurrency: int,
    ) -> None:
        super().__init__(backend=backend)
        self.concurrency = concurrency

    async def worker_loop(
        self, backend: BaseBackend, requests: Iterator[InferencePayload]
    ) -> List[ResultEntry]:
        responses: List[ResultEntry] = []
        for request in requests:
            responses.append(await backend.execute_call(request))
        return responses

    async def run_load(
        self, requests: List[InferencePayload]
    ) -> Tuple[List[ResultEntry], Optional[List[float]]]:
        backends_pool = [self.backend] * self.concurrency
        requests_iter = iter(requests)

        tasks = [self.worker_loop(backend, requests_iter) for backend in backends_pool]
        results: List[ResultEntry] = []
        for task in asyncio.as_completed(tasks):
            for result in await task:
                results.append(result)
        return results, None


class BatchedDriver(LoadDriver):
    """
    Closed-loop bursts: fire N requests in parallel, wait for all to
    complete, sleep until burst_interval, repeat.

    The "burst" name in the original code was a misnomer — this is closed-loop
    batched traffic, not true bursty arrivals. Per-batch wall time is bounded
    by the SLOWEST request. For an open-loop burst pattern that doesn't wait
    for stragglers, use OpenLoopBurstDriver below.
    """

    def __init__(
        self,
        *,
        backend: BaseBackend,
        concurrency: int,
        max_num_burst: int = 10,
        burst_interval: float = 0.5,
    ) -> None:
        super().__init__(backend=backend)
        self.concurrency = concurrency
        self.burst_interval = burst_interval
        self.max_num_burst = max_num_burst

    async def fire_batch(self, requests: List[InferencePayload]):
        tasks = [self.execute_call(request) for request in requests]
        return await asyncio.gather(*tasks)

    async def run_load(
        self, requests: List[InferencePayload]
    ) -> Tuple[List[ResultEntry], Optional[List[float]]]:
        assert (
            len(requests) >= self.concurrency
        ), f"There's not enough prompts for batch {self.concurrency}"

        all_results: List[ResultEntry] = []
        elapsed_times_per_batch: List[float] = []

        for batch_idx in range(len(requests) // self.concurrency):
            if batch_idx >= self.max_num_burst:
                break
            start_time = time.perf_counter()
            batch_result = await self.fire_batch(
                requests[batch_idx * self.concurrency : (batch_idx + 1) * self.concurrency]
            )
            elapsed_time_this_batch = time.perf_counter() - start_time
            elapsed_times_per_batch.append(elapsed_time_this_batch)
            all_results.extend(batch_result)

            wait_time = max(0, self.burst_interval - elapsed_time_this_batch)

            if max(0, wait_time) > 0:
                await asyncio.sleep(wait_time)

        return all_results, elapsed_times_per_batch


class OpenLoopBurstDriver(LoadDriver):
    """
    True open-loop bursts: fire `concurrency` requests at every
    `burst_interval` regardless of whether prior requests have finished.

    Use this when you want to stress the server with arrivals that don't
    back off when the server is slow — the classic "spike test" shape.

    Difference vs BatchedDriver: BatchedDriver waits for the slowest
    request in each batch before the next batch starts, so a slow tail
    silently slows the whole sweep. OpenLoopBurstDriver guarantees the
    next batch fires on schedule.
    """

    def __init__(
        self,
        *,
        backend: BaseBackend,
        concurrency: int,
        max_num_burst: int = 10,
        burst_interval: float = 0.5,
    ) -> None:
        super().__init__(backend=backend)
        self.concurrency = concurrency
        self.burst_interval = burst_interval
        self.max_num_burst = max_num_burst

    async def run_load(
        self, requests: List[InferencePayload]
    ) -> Tuple[List[ResultEntry], Optional[List[float]]]:
        assert (
            len(requests) >= self.concurrency
        ), f"There's not enough prompts for batch {self.concurrency}"

        in_flight: List[asyncio.Task] = []
        # Per-burst wall-time = (this burst's fire time) - (prior burst's fire time).
        # Useful for verifying burst cadence didn't drift under load.
        fire_times: List[float] = []
        run_start = time.perf_counter()

        for batch_idx in range(len(requests) // self.concurrency):
            if batch_idx >= self.max_num_burst:
                break

            t_fire = time.perf_counter()
            fire_times.append(t_fire - run_start)

            batch = requests[batch_idx * self.concurrency : (batch_idx + 1) * self.concurrency]
            for req in batch:
                in_flight.append(asyncio.create_task(self.execute_call(req)))

            # Wait for the next scheduled fire time, NOT for the batch to finish.
            # This is the open-loop part — bursts run on a fixed cadence.
            if batch_idx + 1 < self.max_num_burst:
                next_fire = run_start + (batch_idx + 1) * self.burst_interval
                delay = next_fire - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)

        # All bursts fired; drain anything still in flight.
        all_results: List[ResultEntry] = list(await asyncio.gather(*in_flight))

        # Convert absolute fire-times into per-burst intervals so analytics
        # can compute Per-GPU throughput from real burst boundaries.
        intervals: List[float] = []
        for i in range(1, len(fire_times)):
            intervals.append(fire_times[i] - fire_times[i - 1])
        if fire_times:
            # Final burst's "interval" extends to the end of the run so the
            # last batch has a meaningful denominator.
            intervals.append(time.perf_counter() - run_start - fire_times[-1])

        return all_results, intervals
