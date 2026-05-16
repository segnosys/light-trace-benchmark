"""
Tests for the vendored sglang helpers (lightrace/_sglang_compat.py) — the
ones that replaced the sglang==0.5.2 pin that was downgrading server-side
sglang on every install.
"""
from unittest.mock import patch

import pytest

from lightrace import _sglang_compat as compat


def test_get_tokenizer_uses_local_files_only_for_abs_paths(tmp_path):
    """Absolute paths must skip HF cache lookup to avoid validate_repo_id."""
    # Create a fake directory so isdir() check passes
    fake_dir = tmp_path / "fake_model"
    fake_dir.mkdir()

    with patch("transformers.AutoTokenizer") as mock_at:
        mock_at.from_pretrained.return_value = "tok"
        compat.get_tokenizer(str(fake_dir))
        mock_at.from_pretrained.assert_called_once()
        kwargs = mock_at.from_pretrained.call_args.kwargs
        assert kwargs["trust_remote_code"] is True
        assert kwargs["local_files_only"] is True


def test_get_tokenizer_hf_repo_id_goes_through_cache():
    """HF repo ids fall through to the normal cached path."""
    with patch("transformers.AutoTokenizer") as mock_at:
        mock_at.from_pretrained.return_value = "tok"
        compat.get_tokenizer("HuggingFaceH4/MATH-500")
        mock_at.from_pretrained.assert_called_once()
        kwargs = mock_at.from_pretrained.call_args.kwargs
        # No local_files_only override -> goes through cache
        assert "local_files_only" not in kwargs
        assert kwargs["trust_remote_code"] is True


# --- sample_random_requests ---

class _FakeTokenizer:
    """Cheap stand-in: chars-as-tokens, no vocab needed."""
    vocab_size = 8000

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)

    def encode(self, text, add_special_tokens=False):
        return [42] * len(text)


def test_sample_random_requests_returns_correct_count():
    out = compat.sample_random_requests(
        input_len=10, output_len=5, num_prompts=4,
        tokenizer=_FakeTokenizer(),
    )
    assert len(out) == 4
    for r in out:
        assert hasattr(r, "prompt")
        # Each request honors output_len
        assert r.output_len == 5


def test_sample_random_requests_honors_input_length():
    """Prompts should tokenize to ~input_len tokens."""
    tok = _FakeTokenizer()
    out = compat.sample_random_requests(
        input_len=64, output_len=8, num_prompts=2, tokenizer=tok,
    )
    for r in out:
        # 1 char per token in our fake -> len of prompt == prompt_len ~64
        assert r.prompt_len == 64


def test_sample_random_requests_rejects_unsupported_flags():
    """The shim only mirrors the call shape lightrace actually uses."""
    with pytest.raises(ValueError):
        compat.sample_random_requests(
            input_len=10, output_len=5, num_prompts=1,
            tokenizer=_FakeTokenizer(),
            random_sample=False,
            return_text=True,
        )


def test_sample_random_requests_requires_tokenizer():
    with pytest.raises(ValueError):
        compat.sample_random_requests(
            input_len=10, output_len=5, num_prompts=1,
            tokenizer=None,
        )
