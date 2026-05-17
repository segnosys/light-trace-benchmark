"""
Pure-Python data layer for the agent viewer.

Holds the dataclasses (`MetricPoint`, `BenchmarkTrace`) and the
filesystem-walking loaders (`scan_benchmarks`, `load_metrics`,
`load_metadata`, `load_log_analysis`).

NO Dash / Plotly imports here. Splitting these out of `agent/viewer.py`
lets loader tests run in environments that don't have the optional
`viewer` extras installed (no more `pytest.importorskip("dash")`),
and lets the Flask-based CSV-export route reuse the same loaders
without dragging in the entire dashboard.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MetricPoint:
    """Single data point from metrics.jsonl"""

    timestamp: float
    elapsed_seconds: float
    prefill_tps: float = 0
    prefill_tps_window: float = 0
    prefill_tpm_per_gpu: float = 0
    generation_tps: float = 0
    cache_hit_rate: float = 0
    ideal_cache_hit_rate: float = 0
    requests_completed: int = 0
    requests_sent: int = 0
    errors: int = 0
    in_flight: int = 0
    num_sessions_active: int = 0
    num_sessions_retired: int = 0
    num_sessions_abandoned: int = 0
    num_sessions_total: int = 0
    sessions_created_by_rate: int = 0
    sessions_abandoned_by_rate: int = 0
    gpus: int = 1
    window_size: float = 15.0
    new_session_times: List[float] = None
    forced_session_times: List[float] = None
    existing_session_requests: List = None
    new_planned_prompt_lengths: List[int] = None
    new_planned_ideal_cache_hit_rates: List[float] = None
    new_prompt_lengths: List[int] = None
    new_generation_lengths: List[int] = None
    new_cache_hit_rates: List[float] = None
    new_ideal_cache_hit_rates: List[float] = None
    new_inter_arrival_times: List[float] = None
    new_ttfts: List[float] = None
    new_acceptance_lengths: List[float] = None
    new_acceptance_rates: List[float] = None


@dataclass
class BenchmarkTrace:
    """Per-run state held in memory by the dashboard."""

    label: str               # e.g., "my-test"
    benchmark_name: str      # e.g., "my-test/2025-01-19-14-23-45"
    metrics: List[MetricPoint]
    file_path: Path
    last_position: int = 0   # For incremental reading
    all_new_session_times: List[float] = None
    all_forced_session_times: List[float] = None
    all_existing_session_requests: List = None
    all_planned_prompt_lengths: List[int] = None
    all_planned_ideal_cache_hit_rates: List[float] = None
    all_prompt_lengths: List[int] = None
    all_generation_lengths: List[int] = None
    all_cache_hit_rates: List[float] = None
    all_ideal_cache_hit_rates: List[float] = None
    all_inter_arrival_times: List[float] = None
    metadata: Dict = None    # Config from metadata.json
    log_analysis: Dict = None  # Log analysis data from log_analysis.json


def load_metadata(benchmark_dir: Path) -> Dict:
    """Load metadata.json from a benchmark directory; {} on any error."""
    metadata_file = benchmark_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def load_log_analysis(benchmark_dir: Path) -> Optional[Dict]:
    """Load log_analysis.json if it exists; None otherwise."""
    analysis_file = benchmark_dir / "log_analysis.json"
    if analysis_file.exists():
        try:
            with open(analysis_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def load_metrics(metrics_file: Path, start_position: int = 0):
    """Parse metrics.jsonl incrementally.

    Returns (List[MetricPoint], new_position). Skips malformed lines and
    survives missing fields (everything defaults). The `start_position`
    cursor lets callers tail a file that's still being written.
    """
    metrics: List[MetricPoint] = []
    if not metrics_file.exists():
        return metrics, start_position

    with open(metrics_file, "r") as f:
        f.seek(start_position)
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                metric = MetricPoint(
                    timestamp=data.get("timestamp", 0),
                    elapsed_seconds=data.get("elapsed_seconds", 0),
                    prefill_tps=data.get("prefill_tps", 0),
                    prefill_tps_window=data.get("prefill_tps_window", 0),
                    prefill_tpm_per_gpu=data.get("prefill_tpm_per_gpu", 0),
                    generation_tps=data.get("generation_tps", 0),
                    cache_hit_rate=data.get("cache_hit_rate", 0),
                    ideal_cache_hit_rate=data.get("ideal_cache_hit_rate", 0),
                    requests_completed=data.get("requests_completed", 0),
                    requests_sent=data.get("requests_sent", 0),
                    errors=data.get("errors", 0),
                    in_flight=data.get("in_flight", 0),
                    num_sessions_active=data.get("num_sessions_active", 0),
                    num_sessions_retired=data.get("num_sessions_retired", 0),
                    num_sessions_abandoned=data.get("num_sessions_abandoned", 0),
                    num_sessions_total=data.get("num_sessions_total", 0),
                    sessions_created_by_rate=data.get("sessions_created_by_rate", 0),
                    sessions_abandoned_by_rate=data.get("sessions_abandoned_by_rate", 0),
                    gpus=data.get("gpus", 1),
                    window_size=data.get("window_size", 15.0),
                    new_session_times=data.get("new_session_times", []),
                    forced_session_times=data.get("forced_session_times", []),
                    existing_session_requests=data.get("existing_session_requests", []),
                    new_planned_prompt_lengths=data.get("new_planned_prompt_lengths", []),
                    new_planned_ideal_cache_hit_rates=data.get(
                        "new_planned_ideal_cache_hit_rates", []
                    ),
                    new_prompt_lengths=data.get("new_prompt_lengths", []),
                    new_generation_lengths=data.get("new_generation_lengths", []),
                    new_cache_hit_rates=data.get("new_cache_hit_rates", []),
                    new_ideal_cache_hit_rates=data.get("new_ideal_cache_hit_rates", []),
                    new_inter_arrival_times=data.get("new_inter_arrival_times", []),
                    new_ttfts=data.get("new_ttfts", []),
                    new_acceptance_lengths=data.get("new_acceptance_lengths", []),
                    new_acceptance_rates=data.get("new_acceptance_rates", []),
                )
                metrics.append(metric)
            except json.JSONDecodeError:
                pass
        new_position = f.tell()

    return metrics, new_position


def scan_benchmarks(root_dir: Path) -> Dict[str, BenchmarkTrace]:
    """Discover all `<name>/<timestamp>/metrics.jsonl` runs under root_dir.

    Returns an empty dict when root_dir doesn't exist (silent) — the
    dashboard refresh loop swallows the result and renders an empty
    state. The CSV-export Flask route surfaces a 404 instead.
    """
    traces: Dict[str, BenchmarkTrace] = {}
    if not root_dir.exists():
        return traces

    for name_dir in root_dir.iterdir():
        if not name_dir.is_dir():
            continue
        for timestamp_dir in name_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue
            metrics_file = timestamp_dir / "metrics.jsonl"
            if not metrics_file.exists():
                continue
            benchmark_name = f"{name_dir.name}/{timestamp_dir.name}"
            traces[benchmark_name] = BenchmarkTrace(
                label=name_dir.name,
                benchmark_name=benchmark_name,
                metrics=[],
                file_path=metrics_file,
                all_new_session_times=[],
                all_forced_session_times=[],
                all_existing_session_requests=[],
                all_planned_prompt_lengths=[],
                all_planned_ideal_cache_hit_rates=[],
                all_prompt_lengths=[],
                all_generation_lengths=[],
                all_cache_hit_rates=[],
                all_ideal_cache_hit_rates=[],
                all_inter_arrival_times=[],
                metadata=load_metadata(timestamp_dir),
                log_analysis=load_log_analysis(timestamp_dir),
            )
    return traces
