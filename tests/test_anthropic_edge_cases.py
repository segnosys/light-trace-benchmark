"""
Edge-case tests for AnthropicBackend that weren't covered in the basic
backend smoke (test_anthropic_backend.py):
  - Opus 2048-token minimum vs Sonnet/Haiku 1024
  - Extended TTL (1-hour cache) beta header
  - LIGHTRACE_ANTHROPIC_MIN_CACHEABLE override for the ideal-rate estimator
"""
import os
from unittest.mock import patch

import pytest

from lightrace.analytics import estimate_ideal_cache_hit_rate
from lightrace.backends import AnthropicBackend


# ---------- Opus-aware minimum cacheable size ----------


def _backend(model: str, **kw) -> AnthropicBackend:
    return AnthropicBackend(
        base_url="https://api.anthropic.com",
        api_key="sk",
        model_name=model,
        tokenizer_name=None,
        **kw,
    )


def test_sonnet_min_cacheable_is_1024():
    assert _backend("claude-3-5-sonnet-20241022").min_cacheable_tokens == 1024


def test_haiku_min_cacheable_is_1024():
    assert _backend("claude-3-5-haiku-20241022").min_cacheable_tokens == 1024


def test_opus_min_cacheable_is_2048():
    """Opus is special: its minimum cacheable block is 2048, not 1024."""
    assert _backend("claude-3-opus-20240229").min_cacheable_tokens == 2048


def test_opus_detection_is_case_insensitive():
    assert _backend("Claude-3-OPUS-LATEST").min_cacheable_tokens == 2048


def test_unknown_model_defaults_to_1024():
    """Don't false-positive into Opus for unrecognized models."""
    assert _backend("claude-future-2099").min_cacheable_tokens == 1024


# ---------- Extended-TTL (1-hour cache) beta header ----------


def test_extended_ttl_off_by_default():
    b = _backend("claude-3-5-sonnet-20241022")
    headers = b.build_headers()
    assert "anthropic-beta" not in headers


def test_extended_ttl_on_emits_beta_header():
    b = _backend("claude-3-5-sonnet-20241022", enable_extended_cache_ttl=True)
    headers = b.build_headers()
    assert headers.get("anthropic-beta") == "prompt-caching-2024-07-31"


def test_extended_ttl_keeps_required_headers():
    """Adding the beta header must not drop x-api-key / anthropic-version."""
    b = _backend("claude-3-5-sonnet-20241022", enable_extended_cache_ttl=True)
    headers = b.build_headers()
    assert headers["x-api-key"] == "sk"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["Content-Type"] == "application/json"


# ---------- LIGHTRACE_ANTHROPIC_MIN_CACHEABLE override ----------


class TestEstimatorMinCacheableOverride:
    """The ideal-rate estimator defaults to 1024 but can be lifted via env."""

    def test_default_min_is_1024(self, monkeypatch):
        monkeypatch.delenv("LIGHTRACE_ANTHROPIC_MIN_CACHEABLE", raising=False)
        # A 1024-token prompt with same_prompts_in_burst should produce a positive rate.
        rate = estimate_ideal_cache_hit_rate(
            provider="anthropic", dataset_type="synthetic",
            num_examples=4, concurrency=4,
            same_prompts_in_burst=True, synthetic_input_length=1024,
        )
        assert rate is not None and rate > 0

    def test_opus_override_via_env(self, monkeypatch):
        """Set the env var to 2048 — a 1024-token prompt should now return 0."""
        monkeypatch.setenv("LIGHTRACE_ANTHROPIC_MIN_CACHEABLE", "2048")
        rate = estimate_ideal_cache_hit_rate(
            provider="anthropic", dataset_type="synthetic",
            num_examples=4, concurrency=4,
            same_prompts_in_burst=True, synthetic_input_length=1024,
        )
        assert rate == 0.0

    def test_env_override_still_lets_above_threshold_cache(self, monkeypatch):
        """Above the lifted threshold, hits should still register."""
        monkeypatch.setenv("LIGHTRACE_ANTHROPIC_MIN_CACHEABLE", "2048")
        rate = estimate_ideal_cache_hit_rate(
            provider="anthropic", dataset_type="synthetic",
            num_examples=4, concurrency=4,
            same_prompts_in_burst=True, synthetic_input_length=4000,
        )
        assert rate is not None and rate > 0
