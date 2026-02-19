import csv
import os
import re
from dataclasses import fields
from typing import Dict, List, Optional

import numpy as np
from tabulate import SEPARATING_LINE, tabulate

from lightrace.schema import (
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
        per_device=(
            DeviceThroughput(-1 if num_gpus is None else num_gpus, 0, 0)
            if per_batch_elapsed_times is None or num_gpus is None
            else compute_device_throughput(
                success_results, per_batch_elapsed_times, int(traffic_level), num_gpus
            )
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
    )


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
                ["Job-level tokens/s (decode):", f"{report.overview.job_level_tps:.2f}"],
                ["Job-level actual QPS:", f"{report.overview.actual_qps:.2f}"],
            ],
            colalign=("left", "right"),
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
