"""
Tests for the cache-rate aggregation helpers in analytics.
"""
from legacy.analytics import _extract_cache_hit_rate, _sum_cached_input_tokens
from legacy.schema import LatencyProfile, ResultEntry


def _r(*, prompt, cached):
    return ResultEntry(metrics=LatencyProfile(
        input_token_count=prompt,
        cached_input_tokens=cached,
    ), success=True)


def test_hit_rate_mean_over_requests_that_have_cache_info():
    results = [
        _r(prompt=4000, cached=3200),
        _r(prompt=4000, cached=3200),
        _r(prompt=4000, cached=3200),
        _r(prompt=4000, cached=None),  # excluded — backend didn't report cache
    ]
    stats = _extract_cache_hit_rate(results, dump_raw=False)
    assert stats is not None
    # 3 of 4 entries contribute 3200/4000 = 0.8
    assert abs(stats.mean - 0.8) < 1e-6
    # All ratios identical -> stdev should be 0
    assert abs(stats.stdev) < 1e-6


def test_hit_rate_returns_none_when_no_backend_reported():
    results = [_r(prompt=4000, cached=None) for _ in range(3)]
    assert _extract_cache_hit_rate(results, dump_raw=False) is None


def test_sum_cached_input_tokens_aggregates_when_present():
    results = [
        _r(prompt=4000, cached=3200),
        _r(prompt=4000, cached=3200),
        _r(prompt=4000, cached=None),
    ]
    assert _sum_cached_input_tokens(results) == 6400


def test_sum_cached_returns_none_when_no_data():
    results = [_r(prompt=4000, cached=None) for _ in range(3)]
    assert _sum_cached_input_tokens(results) is None


def test_hit_rate_ignores_zero_prompt_length():
    """Guards against divide-by-zero when a backend can't tokenize the prompt."""
    results = [
        _r(prompt=0, cached=0),  # excluded
        _r(prompt=4000, cached=3200),
    ]
    stats = _extract_cache_hit_rate(results, dump_raw=False)
    assert stats is not None
    assert abs(stats.mean - 0.8) < 1e-6
