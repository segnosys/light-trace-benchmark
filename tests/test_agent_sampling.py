"""
Tests for the agent/sampling.py module — distribution-sampling helpers
extracted out of agent_throughput.py.
"""
import math

import numpy as np

from agent.sampling import (
    MAX_INTER_ARRIVAL_TIME,
    MAX_RETRIES,
    MAX_THINK_TIME,
    MIN_GENERATION_TIME,
    MIN_THINK_TIME,
    draw_lognormal,
    draw_session_lifetime,
    draw_think_time,
)


def test_constants_have_expected_values():
    """Lock in the values — they're load-bearing knobs in the live driver."""
    assert MIN_GENERATION_TIME == 0.05
    assert MIN_THINK_TIME == 5.0
    assert MAX_THINK_TIME == 900.0
    assert MAX_INTER_ARRIVAL_TIME == 300.0
    assert MAX_RETRIES == 3


class TestDrawLognormal:

    def test_returns_max_1_int_mean_for_invalid_params(self):
        """When mean or median is <=0, returns `max(1, int(mean))`."""
        assert draw_lognormal(0, 100) == 1
        assert draw_lognormal(100, 0) == 100   # mean=100 -> returns 100
        assert draw_lognormal(-5, 10) == 1      # mean<0, max(1,-5) = 1

    def test_returns_int(self):
        np.random.seed(0)
        v = draw_lognormal(100, 80)
        assert isinstance(v, int)
        assert v >= 1

    def test_mean_above_median_widens_distribution(self):
        """When mean > median, sigma scales with the ratio."""
        np.random.seed(0)
        # Mean 2x median => sigma = sqrt(2*ln(2)) ≈ 1.18
        samples = [draw_lognormal(200, 100) for _ in range(500)]
        # With a wide sigma, samples should span at least an order of magnitude.
        assert max(samples) > 10 * min(samples)

    def test_mean_near_median_keeps_narrow(self):
        """When mean ~= median, sigma should be tight (~1% CV)."""
        np.random.seed(0)
        samples = [draw_lognormal(100, 100) for _ in range(500)]
        mean_s = sum(samples) / len(samples)
        # Stdev should be small relative to mean (within 2%).
        var = sum((s - mean_s) ** 2 for s in samples) / len(samples)
        stdev = math.sqrt(var)
        assert stdev / mean_s < 0.05


class TestDrawSessionLifetime:

    def test_returns_float_above_60s_floor(self):
        np.random.seed(0)
        v = draw_session_lifetime(300, 200)
        assert isinstance(v, float)
        assert v >= 60.0

    def test_invalid_params_fall_through(self):
        # mean<=0 returns max(60, mean) -> 60.0
        assert draw_session_lifetime(0, 0) == 60.0
        assert draw_session_lifetime(-1, 100) == 60.0

    def test_very_small_mean_clamped_to_60(self):
        np.random.seed(0)
        v = draw_session_lifetime(5, 4)
        assert v >= 60.0


class TestDrawThinkTime:

    def test_zero_mean_returns_floor(self):
        assert draw_think_time(0) == MIN_THINK_TIME
        assert draw_think_time(-5) == MIN_THINK_TIME

    def test_result_is_clamped_to_min_max(self):
        np.random.seed(0)
        for _ in range(50):
            v = draw_think_time(60.0, shape=1.0)
            assert MIN_THINK_TIME <= v <= MAX_THINK_TIME

    def test_higher_shape_tightens_distribution(self):
        """shape=10 should produce tighter samples than shape=1."""
        np.random.seed(0)
        loose = [draw_think_time(60.0, shape=1.0) for _ in range(1000)]
        np.random.seed(0)
        tight = [draw_think_time(60.0, shape=10.0) for _ in range(1000)]
        # Both means should be near 60, but tight should have lower variance
        var_loose = np.var(loose)
        var_tight = np.var(tight)
        assert var_tight < var_loose
