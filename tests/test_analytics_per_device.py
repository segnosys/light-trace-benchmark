"""
Tests for the Per-GPU throughput fix.

Pre-fix: outside burst mode the table rendered `Per-GPU tokens/s: 0.00`
because `per_batch_elapsed_times` was None. Post-fix: fall back to
total_output / total_elapsed / num_gpus when per-batch data isn't there.
"""
from legacy.analytics import _compute_per_device
from legacy.schema import LatencyProfile, ResultEntry


def _r(out_tokens):
    return ResultEntry(metrics=LatencyProfile(output_token_count=out_tokens), success=True)


def test_burst_path_uses_per_batch_data():
    """When per_batch_elapsed_times is populated, the burst path runs."""
    results = [_r(100) for _ in range(8)]
    elapsed = [1.0, 1.0]  # 2 batches of 4
    out = _compute_per_device(
        results=results, per_batch_elapsed_times=elapsed,
        traffic_level=4, num_gpus=2, total_elapse_time_s=2.0,
    )
    # Each batch: 400 tokens / 1 s / 2 gpus = 200 -> mean 200, stdev 0
    assert out.num_gpus == 2
    assert abs(out.tps_mean - 200.0) < 1e-6


def test_concurrent_fallback_uses_total_tokens_over_wall_clock():
    """When per_batch_elapsed_times is None, fallback formula kicks in."""
    results = [_r(100), _r(100), _r(100), _r(100)]
    out = _compute_per_device(
        results=results, per_batch_elapsed_times=None,
        traffic_level=4, num_gpus=2, total_elapse_time_s=2.0,
    )
    # 400 total tokens / 2 s / 2 gpus = 100
    assert out.num_gpus == 2
    assert abs(out.tps_mean - 100.0) < 1e-6


def test_num_gpus_missing_returns_placeholder():
    """Without num_gpus we can't compute per-device; return a clear marker."""
    out = _compute_per_device(
        results=[_r(10)], per_batch_elapsed_times=None,
        traffic_level=1, num_gpus=None, total_elapse_time_s=1.0,
    )
    assert out.num_gpus == -1
    assert out.tps_mean == 0


def test_zero_elapsed_does_not_divide_by_zero():
    out = _compute_per_device(
        results=[_r(10)], per_batch_elapsed_times=None,
        traffic_level=1, num_gpus=1, total_elapse_time_s=0.0,
    )
    assert out.tps_mean == 0.0
