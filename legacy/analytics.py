import csv
import os
import re
from dataclasses import fields
from typing import Dict, List, Optional

import numpy as np
from tabulate import SEPARATING_LINE, tabulate

from legacy.schema import (
    BenchmarkReport,
    BackendInfo,
    DeviceThroughput,
    LoadShape,
    OutputDistribution,
    ResultEntry,
    RunOverview,
    StatsSummary,
)


def compute_percentiles(values: List[float], dump_raw: bool = False) -> StatsSummary:

    p05, p50, p80, p95, p99, p999 = np.percentile(values, [5, 50, 80, 95, 99, 99.9])
    return StatsSummary(
        mean=np.mean(values),
        stdev=np.std(values),
        p05=p05,
        p50=p50,
        p80=p80,
        p95=p95,
        p99=p99,
        p999=p999,
        distribution=values if dump_raw else None,
    )


def extract_accept_rate_from_responses(
    results: List[ResultEntry],
    dump_raw: bool,
) -> Optional[StatsSummary]:
    """
    Collect acceptance rate statistics from API responses.
    """
    acceptance_rates = [
        result.metrics.accept_ratio
        for result in results
        if result.success and result.metrics.accept_ratio is not None
    ]

    if len(acceptance_rates) == 0:
        return None

    return compute_percentiles(acceptance_rates, dump_raw=dump_raw)


def extract_accept_rate_from_logs(
    results: List[ResultEntry],
    engine_log_path: str,
    dump_raw: bool,
    accept_rate_pattern: str,
) -> Optional[StatsSummary]:
    if engine_log_path is None or accept_rate_pattern is None:
        return None

    n_requests = len(results)

    with open(engine_log_path, "r") as f:
        content = f.read()
        acceptance_rates = re.findall(accept_rate_pattern, content)
    acceptance_rates = [float(rate) for rate in acceptance_rates]

    if len(acceptance_rates) == 0:
        return None
    return compute_percentiles(acceptance_rates[-n_requests:], dump_raw=dump_raw)


def compute_distribution(values: List[int]) -> OutputDistribution:
    return OutputDistribution(
        avg_len=np.mean(values),
        stdev_len=np.std(values),
        min_len=np.min(values),
        max_len=np.max(values),
        total_tokens=np.sum(values),
    )


def _compute_per_device(
    *,
    results: List[ResultEntry],
    per_batch_elapsed_times: Optional[List[float]],
    traffic_level: float,
    num_gpus: Optional[int],
    total_elapse_time_s: float,
) -> DeviceThroughput:
    """
    Per-GPU throughput. Uses per-batch data when available (burst pattern);
    otherwise falls back to total_decode_tokens / total_elapsed / num_gpus
    so concurrent/qps modes still report a meaningful number instead of 0.
    """
    if num_gpus is None or num_gpus <= 0:
        return DeviceThroughput(num_gpus=-1, tps_mean=0, tps_stdev=0)

    if per_batch_elapsed_times:
        return compute_device_throughput(
            results, per_batch_elapsed_times, int(traffic_level), num_gpus
        )

    # Fallback: aggregate decode tokens / wall time / GPU count.
    total_output = sum(
        r.metrics.output_token_count
        for r in results
        if r.metrics and r.metrics.output_token_count
    )
    tps = (total_output / total_elapse_time_s / num_gpus) if total_elapse_time_s > 0 else 0.0
    return DeviceThroughput(num_gpus=num_gpus, tps_mean=tps, tps_stdev=0.0)


def compute_device_throughput(
    results: List[ResultEntry],
    per_batch_elapsed_times: List[float],
    batch_size: int,
    num_gpus: int = 1,
) -> DeviceThroughput:
    batch_gpu_tps = []
    for i in range(len(per_batch_elapsed_times)):
        batch_total_output_tokens = sum(
            result.metrics.output_token_count
            for result in results[i * batch_size : (i + 1) * batch_size]
        )
        batch_gpu_tps.append(batch_total_output_tokens / per_batch_elapsed_times[i] / num_gpus)

    return DeviceThroughput(
        num_gpus=num_gpus,
        tps_mean=np.mean(batch_gpu_tps),
        tps_stdev=np.std(batch_gpu_tps),
    )


def estimate_ideal_cache_hit_rate(
    *,
    provider: str,
    dataset_type: str,
    num_examples: int,
    concurrency: Optional[int] = None,
    max_num_burst: Optional[int] = None,
    same_prompts_in_burst: bool = False,
    synthetic_input_length: Optional[int] = None,
    synthetic_cached_input_length: Optional[int] = None,
    gsp_cached_fraction: Optional[float] = None,
    gsp_groups: Optional[int] = None,
    traffic_pattern: Optional[str] = None,
) -> Optional[float]:
    """
    Predicted per-request cache hit ratio from workload params.

    Compare this against the server-reported cache_hit_rate.mean to spot:
      - cache feature disabled / misconfigured server (actual << ideal)
      - over-attribution / double-counting (actual > ideal by a wide margin)
      - workload that simply can't hit cache (returns None)

    Approximations:
      * "First request creates, rest hit" — assumes no eviction within the run.
      * Anthropic Sonnet/Haiku minimum cacheable size is 1024 tokens; if the
        cacheable chunk is smaller, returns 0.0 for anthropic provider.
      * 5-min TTL is ignored — benchmark runs should fit inside one window.

    Anthropic protocol notes:
      Cache hits require the prefix UP TO AND INCLUDING the cache_control
      marker to match across requests. AnthropicBackend places the marker:
        - on the system block when messages-mode payloads have one
        - on the cacheable-prefix block when InferencePayload.cacheable_prefix
          is set (synthetic + synthetic_cached_input_length path)
        - on the entire user block otherwise (same_prompts_in_burst path)
      For dataset shapes where the marker can't be cleanly placed against a
      shared prefix (e.g. generated-shared-prefix today), ideal returns 0.

    Returns None when the shape doesn't admit a clean estimate.
    """
    # Anthropic Sonnet/Haiku won't cache chunks below 1024 tokens; Opus is 2048.
    # The estimator can't peek at model_name from this signature, so default to
    # 1024. Callers wanting the Opus threshold can override via the env var
    # AGENT_BENCH_ANTHROPIC_MIN_CACHEABLE (legacy: LIGHTRACE_ANTHROPIC_MIN_CACHEABLE).
    # Kept as an escape hatch rather than yet another argument; the Opus-aware
    # logic lives on AnthropicBackend itself.
    import os as _os
    _override = (
        _os.environ.get("AGENT_BENCH_ANTHROPIC_MIN_CACHEABLE")
        or _os.environ.get("LIGHTRACE_ANTHROPIC_MIN_CACHEABLE")
    )
    anthropic_min_cacheable = (
        int(_override) if (provider == "anthropic" and _override) else (1024 if provider == "anthropic" else 0)
    )

    # Case 1: identical prompts repeated -> trivially cacheable
    if same_prompts_in_burst:
        bursts = max(1, max_num_burst or 1)
        per_burst = max(1, concurrency or 1)
        total = bursts * per_burst
        if total <= 1:
            return 0.0
        # Each burst: 1 write + (per_burst - 1) reads. Across all bursts the
        # avg hit per request is approximately (per_burst - 1) / per_burst,
        # ignoring cross-burst sharing (which would only help).
        avg_hit = (per_burst - 1) / per_burst
        # Anthropic only caches >= 1024-token chunks.
        if synthetic_input_length and synthetic_input_length < anthropic_min_cacheable:
            return 0.0
        return avg_hit

    # Case 2: synthetic with a fixed cacheable prefix
    if (
        dataset_type == "synthetic"
        and synthetic_cached_input_length
        and synthetic_input_length
        and synthetic_input_length > 0
    ):
        # First request creates cache, rest read. Over num_examples:
        eff = max(0.0, (num_examples - 1) / num_examples) if num_examples else 0.0
        cache_frac = synthetic_cached_input_length / synthetic_input_length
        if synthetic_cached_input_length < anthropic_min_cacheable:
            return 0.0
        return eff * cache_frac

    # Case 3: generated-shared-prefix mode (gsp)
    if dataset_type == "generated-shared-prefix" and gsp_cached_fraction is not None:
        # Anthropic protocol guardrail: gsp doesn't currently expose the
        # shared-prefix boundary via InferencePayload.cacheable_prefix, so
        # AnthropicBackend marks the whole content -> cache key includes the
        # varying suffix -> 0% hit. Servers with implicit prefix caching
        # (sglang, vllm, OpenAI radix) still work.
        if provider == "anthropic":
            return 0.0
        groups = max(1, gsp_groups or 1)
        # Per group: 1 write + (n_per_group - 1) reads
        n_per_group = max(1, num_examples // groups)
        if n_per_group <= 1:
            return 0.0
        eff = (n_per_group - 1) / n_per_group
        # Approximate cacheable size from gsp_cached_fraction * total prompt.
        if synthetic_input_length:
            approx_cached_tokens = synthetic_input_length * gsp_cached_fraction
            if approx_cached_tokens < anthropic_min_cacheable:
                return 0.0
        return eff * gsp_cached_fraction

    # No clean estimate (sharegpt random, hf one-shot, jsonl, etc.)
    return None


def summarize_benchmark(
    traffic_mode: str,
    traffic_level: float,
    backend_name: str,
    model_name: str,
    total_elapse_time_s: float,
    results: List[ResultEntry],
    num_gpus: Optional[int] = None,
    per_batch_elapsed_times: Optional[List[float]] = None,
    histogram_metrics: bool = False,
    hf_dataset_name: Optional[str] = None,
    engine_log_path: Optional[str] = None,
    accept_rate_pattern: Optional[str] = None,
    ideal_cache_hit_rate: Optional[float] = None,
    engine_metrics: Optional[Dict] = None,
) -> BenchmarkReport:
    success_results = [result for result in results if result.success]

    return BenchmarkReport(
        backend=BackendInfo(
            name=backend_name,
            model=model_name,
        ),
        load=LoadShape(
            mode=traffic_mode,
            level=traffic_level,
        ),
        input=compute_distribution(
            [
                result.metrics.input_token_count
                for result in success_results
                if result.metrics.input_token_count
            ]
        ),
        output=compute_distribution(
            [
                result.metrics.output_token_count
                for result in success_results
                if result.metrics.output_token_count
            ]
        ),
        ttft=compute_percentiles(
            [
                result.metrics.first_token_latency
                for result in success_results
                if result.metrics.first_token_latency
            ],
            dump_raw=histogram_metrics,
        ),
        user_tps=compute_percentiles(
            [
                1000.0 / result.metrics.ms_per_token
                for result in success_results
                if result.metrics.ms_per_token
            ],
            dump_raw=histogram_metrics,
        ),
        e2e=compute_percentiles(
            [
                result.metrics.end_to_end_ms
                for result in success_results
                if result.metrics.end_to_end_ms
            ]
        ),
        overview=RunOverview(
            total_num_requests=len(results),
            total_elapsed_time_s=total_elapse_time_s,
            job_level_tps=np.sum(
                [
                    result.metrics.output_token_count
                    for result in success_results
                    if result.metrics.output_token_count
                ]
            )
            / total_elapse_time_s,
            actual_qps=len(results) / total_elapse_time_s,
            num_failed_requests=len(results) - len(success_results),
        ),
        per_device=_compute_per_device(
            results=success_results,
            per_batch_elapsed_times=per_batch_elapsed_times,
            traffic_level=traffic_level,
            num_gpus=num_gpus,
            total_elapse_time_s=total_elapse_time_s,
        ),
        accept_ratio=(
            extract_accept_rate_from_responses(
                results=success_results,
                dump_raw=histogram_metrics,
            )
            or extract_accept_rate_from_logs(
                results=results,
                engine_log_path=engine_log_path,
                dump_raw=histogram_metrics,
                accept_rate_pattern=accept_rate_pattern,
            )
        ),
        hf_dataset_name=hf_dataset_name,
        cache_hit_rate=_extract_cache_hit_rate(success_results, histogram_metrics),
        total_cached_input_tokens=_sum_cached_input_tokens(success_results),
        ideal_cache_hit_rate=ideal_cache_hit_rate,
        engine_metrics=engine_metrics,
    )


def _extract_cache_hit_rate(
    results: List[ResultEntry], dump_raw: bool
) -> Optional[StatsSummary]:
    """Per-request cache hit ratio (cached / total input). None if no backend reported."""
    ratios = []
    for r in results:
        if not r.metrics:
            continue
        cached = r.metrics.cached_input_tokens
        total = r.metrics.input_token_count
        if cached is None or not total or total <= 0:
            continue
        ratios.append(cached / total)
    if not ratios:
        return None
    return compute_percentiles(ratios, dump_raw=dump_raw)


def _sum_cached_input_tokens(results: List[ResultEntry]) -> Optional[int]:
    total = 0
    saw_any = False
    for r in results:
        if r.metrics and r.metrics.cached_input_tokens is not None:
            total += r.metrics.cached_input_tokens
            saw_any = True
    return total if saw_any else None


def render_report(report: BenchmarkReport) -> None:
    title = (
        f"Backend: {report.backend.name}, Model: {report.backend.model}, "
        f"GPUs: {'N/A' if not report.per_device else report.per_device.num_gpus}"
    )

    print(os.linesep)
    print(title)
    print(
        tabulate(
            [
                ["Traffic mode:", f"{report.load.mode}"],
                [
                    f"{'QPS' if report.load.mode == 'qps' else 'Concurrency'} level:",
                    f"{report.load.level:.1f}",
                ],
                ["Total num. of requests:", f"{report.overview.total_num_requests}"],
                ["Num. of failed requests:", f"{report.overview.num_failed_requests}"],
                ["Total elapsed time (s):", f"{report.overview.total_elapsed_time_s:.2f}"],
                SEPARATING_LINE,
                ["Prompt length:", f"{report.input.avg_len:.1f} +/- {report.input.stdev_len:.1f}"],
                ["Prompt length range:", f"[{report.input.min_len}, {report.input.max_len}]"],
                SEPARATING_LINE,
                [
                    "Decode length:",
                    f"{report.output.avg_len:.1f} +/- {report.output.stdev_len:.1f}",
                ],
                ["Decode length range:", f"[{report.output.min_len}, {report.output.max_len}]"],
                SEPARATING_LINE,
                [
                    "Per-request tokens/s:",
                    f"{report.user_tps.mean:.2f} +/- {report.user_tps.stdev:.2f}",
                ],
                ["Per-request TTFT mean (ms):", f"{report.ttft.mean:.2f} +/- {report.ttft.stdev:.2f}"],
                ["Per-request TTFT median (ms):", f"{report.ttft.p50:.2f}"],
                ["Per-request TTFT P99 (ms):", f"{report.ttft.p99:.2f}"],
                ["Per-request P99 round-trip latency (ms):", f"{report.e2e.p99:.0f}"],
                (
                    ["", ""]
                    if report.per_device is None
                    else [
                        "Per-GPU tokens/s:",
                        f"{report.per_device.tps_mean:.2f} +/- {report.per_device.tps_stdev:.2f}",
                    ]
                ),
                (
                    ["", ""]
                    if report.accept_ratio is None
                    else [
                        "Acceptance rate:",
                        f"{report.accept_ratio.mean:.2f} +/- {report.accept_ratio.stdev:.2f}",
                    ]
                ),
                (
                    ["", ""]
                    if report.ideal_cache_hit_rate is None
                    else [
                        "Ideal cache hit rate (workload-predicted):",
                        f"{100 * report.ideal_cache_hit_rate:.1f}%",
                    ]
                ),
                (
                    ["", ""]
                    if report.cache_hit_rate is None
                    else [
                        "Prompt cache hit rate (server-reported):",
                        f"{100 * report.cache_hit_rate.mean:.1f}% +/- {100 * report.cache_hit_rate.stdev:.1f}%",
                    ]
                ),
                (
                    ["", ""]
                    if (report.cache_hit_rate is None
                        or report.ideal_cache_hit_rate is None
                        or report.ideal_cache_hit_rate <= 0)
                    else [
                        "Cache efficiency (server / ideal):",
                        f"{100 * report.cache_hit_rate.mean / report.ideal_cache_hit_rate:.1f}%",
                    ]
                ),
                (
                    ["", ""]
                    if report.total_cached_input_tokens is None
                    else [
                        "Total cached input tokens:",
                        f"{report.total_cached_input_tokens}",
                    ]
                ),
                ["Job-level tokens/s (decode):", f"{report.overview.job_level_tps:.2f}"],
                ["Job-level actual QPS:", f"{report.overview.actual_qps:.2f}"],
            ],
            colalign=("left", "right"),
        )
    )

    # Optional server-side metrics block — only shown when --engine_metrics_url
    # was set AND at least one sample came back from the endpoint.
    if report.engine_metrics:
        rows = []
        for name, s in sorted(report.engine_metrics.items()):
            mean = s.get("mean")
            mx = s.get("max")
            latest = s.get("latest")
            samples = int(s.get("samples") or 0)
            rows.append([
                name,
                f"{mean:.2f}" if mean is not None else "n/a",
                f"{mx:.2f}" if mx is not None else "n/a",
                f"{latest:.2f}" if latest is not None else "n/a",
                samples,
            ])
        if rows:
            print()
            print("Server-side metrics (Prometheus /metrics):")
            print(
                tabulate(
                    rows,
                    headers=["metric", "mean", "max", "latest", "samples"],
                    colalign=("left", "right", "right", "right", "right"),
                )
            )


def export_report_csv(
    report: BenchmarkReport,
    output_file: str,
    extra_eval_metadata: Optional[Dict[str, str]] = None,
) -> None:
    def flatten_dataclass(obj):
        flat_dict = {}
        for field in fields(type(obj)):
            value = getattr(obj, field.name)

            if hasattr(value, "__dataclass_fields__"):
                nested_dict = flatten_dataclass(value)
                flat_dict.update({f"{field.name}_{k}": v for k, v in nested_dict.items()})
            else:
                flat_dict[field.name] = value
        return flat_dict

    row = flatten_dataclass(report)
    extra_fields = extra_eval_metadata or {}
    for key, value in extra_fields.items():
        column_name = key[len("extra_"):] if key.startswith("extra_") else key
        if column_name in row:
            raise ValueError(
                f"Extra metadata key '{column_name}' conflicts with existing CSV column."
            )
        row[column_name] = str(value)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_exists = os.path.isfile(output_file)
    existing_fieldnames: List[str] = []
    existing_rows: List[Dict[str, str]] = []

    if csv_exists:
        with open(output_file, newline="") as existing_file:
            reader = csv.DictReader(existing_file)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = [dict(row) for row in reader]

    fieldnames = list(existing_fieldnames)
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    if not csv_exists or fieldnames != existing_fieldnames:
        rows_to_write = existing_rows + [row]
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for existing_row in rows_to_write:
                normalized_row = {field: existing_row.get(field, "") for field in fieldnames}
                writer.writerow(normalized_row)
    else:
        with open(output_file, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
