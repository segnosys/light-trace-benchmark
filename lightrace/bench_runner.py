from typing import List, Optional, Tuple

from lightrace import REGISTERED_BACKENDS
from lightrace.load_driver import (
    BatchedDriver,
    LoadDriver,
    OpenLoopBurstDriver,
    ParallelWorkerDriver,
    RateBasedDriver,
)
from lightrace.schema import InferencePayload, ResultEntry


class InferenceBenchRunner:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model_name: str,
        tokenizer_name: str,
        traffic_pattern: str,
        force_recounting_completions: bool = False,
    ):
        if provider not in REGISTERED_BACKENDS:
            raise ValueError(f"Unsupported backend: {provider}.")
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.traffic_pattern = traffic_pattern
        self.force_recounting_completions = force_recounting_completions

    async def run_benchmark(
        self,
        *,
        requests: List[InferencePayload],
        level: float,
        max_num_burst: int,
        burst_interval: float,
        qps_distribution: str,
        qps_duration_s: int,
    ) -> Tuple[List[ResultEntry], Optional[List[float]]]:
        backend_instance = REGISTERED_BACKENDS[self.provider](
            self.base_url,
            self.api_key,
            self.model_name,
            self.tokenizer_name,
            self.force_recounting_completions,
        )

        driver: LoadDriver
        if self.traffic_pattern == "burst":
            driver = BatchedDriver(
                backend=backend_instance,
                concurrency=int(level),
                max_num_burst=max_num_burst,
                burst_interval=burst_interval,
            )
        elif self.traffic_pattern == "open_loop_burst":
            driver = OpenLoopBurstDriver(
                backend=backend_instance,
                concurrency=int(level),
                max_num_burst=max_num_burst,
                burst_interval=burst_interval,
            )
        elif self.traffic_pattern == "qps":
            driver = RateBasedDriver(
                backend=backend_instance,
                qps_level=level,
                duration_s=qps_duration_s,
                distribution=qps_distribution,
            )
        elif self.traffic_pattern == "concurrent":
            driver = ParallelWorkerDriver(
                backend=backend_instance,
                concurrency=int(level),
            )
        else:
            raise ValueError(f"Unsupported traffic pattern: {self.traffic_pattern}.")

        try:
            return await driver.run_load(requests)
        finally:
            # Close the shared aiohttp session opened by the backend.
            # Without this we leak the connector and emit a noisy
            # "Unclosed client session" warning at interpreter shutdown.
            await backend_instance.close()
