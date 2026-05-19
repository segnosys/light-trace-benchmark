"""
Tests for the workload-predicted ideal cache hit rate.

Covers all branches of estimate_ideal_cache_hit_rate, with explicit focus on
Anthropic's protocol semantics (1024-token min, marker-position constraints).
"""

import pytest

from lightrace.analytics import estimate_ideal_cache_hit_rate


def _approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------- same_prompts_in_burst ----------

class TestSamePromptsInBurst:

    def test_per_request_avg_is_one_minus_inverse_concurrency(self):
        # 4 per burst -> 3/4 of requests in each burst are hits
        rate = estimate_ideal_cache_hit_rate(
            provider="sglang", dataset_type="synthetic",
            num_examples=8, concurrency=4, max_num_burst=2,
            same_prompts_in_burst=True,
            synthetic_input_length=4000,
        )
        assert _approx(rate, 0.75)

    def test_concurrency_one_yields_zero(self):
        # No hits possible if only one request per burst
        rate = estimate_ideal_cache_hit_rate(
            provider="sglang", dataset_type="synthetic",
            num_examples=4, concurrency=1, max_num_burst=4,
            same_prompts_in_burst=True,
            synthetic_input_length=4000,
        )
        assert rate == 0.0

    def test_anthropic_below_1024_min_returns_zero(self):
        rate = estimate_ideal_cache_hit_rate(
            provider="anthropic", dataset_type="synthetic",
            num_examples=4, concurrency=4,
            same_prompts_in_burst=True,
            synthetic_input_length=500,  # below 1024
        )
        assert rate == 0.0

    def test_anthropic_above_1024_min_allowed(self):
        rate = estimate_ideal_cache_hit_rate(
            provider="anthropic", dataset_type="synthetic",
            num_examples=8, concurrency=4, max_num_burst=2,
            same_prompts_in_burst=True,
            synthetic_input_length=2000,
        )
        assert _approx(rate, 0.75)


# ---------- synthetic + cached_input_length ----------

class TestSyntheticCachedPrefix:

    def test_formula_is_efficiency_times_cached_fraction(self):
        rate = estimate_ideal_cache_hit_rate(
            provider="sglang", dataset_type="synthetic",
            num_examples=8,
            synthetic_input_length=4000,
            synthetic_cached_input_length=3200,
        )
        # (8-1)/8 * 3200/4000 = 0.875 * 0.8 = 0.7
        assert _approx(rate, 0.7)

    def test_anthropic_below_min_returns_zero(self):
        # Cached prefix size 500 < 1024-token Sonnet minimum
        rate = estimate_ideal_cache_hit_rate(
            provider="anthropic", dataset_type="synthetic",
            num_examples=8,
            synthetic_input_length=4000,
            synthetic_cached_input_length=500,
        )
        assert rate == 0.0

    def test_anthropic_at_or_above_min_allowed(self):
        rate = estimate_ideal_cache_hit_rate(
            provider="anthropic", dataset_type="synthetic",
            num_examples=8,
            synthetic_input_length=4000,
            synthetic_cached_input_length=3200,
        )
        # Same formula as non-anthropic — the split lets the protocol match.
        assert _approx(rate, 0.7)


# ---------- generated-shared-prefix ----------

class TestGeneratedSharedPrefix:

    def test_sglang_gets_efficiency_times_cache_fraction(self):
        rate = estimate_ideal_cache_hit_rate(
            provider="sglang", dataset_type="generated-shared-prefix",
            num_examples=16, concurrency=8, max_num_burst=2,
            gsp_cached_fraction=0.8, gsp_groups=2,
            synthetic_input_length=4000,
        )
        # n_per_group = 16/2 = 8; eff = 7/8; ideal = 7/8 * 0.8 = 0.7
        assert _approx(rate, 0.7)

    def test_anthropic_gsp_is_zero_today(self):
        """gsp doesn't expose a cacheable_prefix yet -> Anthropic can't hit."""
        rate = estimate_ideal_cache_hit_rate(
            provider="anthropic", dataset_type="generated-shared-prefix",
            num_examples=16, concurrency=8,
            gsp_cached_fraction=0.8, gsp_groups=2,
            synthetic_input_length=4000,
        )
        assert rate == 0.0

    def test_single_request_per_group_no_hits(self):
        rate = estimate_ideal_cache_hit_rate(
            provider="sglang", dataset_type="generated-shared-prefix",
            num_examples=2, gsp_cached_fraction=0.8, gsp_groups=2,
            synthetic_input_length=4000,
        )
        # n_per_group = 1 -> no hits
        assert rate == 0.0


# ---------- shapes with no clean estimate ----------

class TestUnknownShapes:

    @pytest.mark.parametrize("dataset_type", ["hf", "jsonl", "sharegpt"])
    def test_returns_none(self, dataset_type):
        rate = estimate_ideal_cache_hit_rate(
            provider="sglang", dataset_type=dataset_type,
            num_examples=8, concurrency=4,
        )
        assert rate is None

    def test_synthetic_without_cache_setup_returns_none(self):
        # No same_prompts_in_burst and no cached_input_length
        rate = estimate_ideal_cache_hit_rate(
            provider="sglang", dataset_type="synthetic",
            num_examples=8, concurrency=4,
            synthetic_input_length=4000,
            synthetic_cached_input_length=None,
        )
        assert rate is None
