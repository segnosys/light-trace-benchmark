import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Iterator, List, Optional, Tuple

from lightrace.backends import BaseBackend
from lightrace.schema import InferencePayload, ResultEntry


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
