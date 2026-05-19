"""
Vendored helpers replacing what agent-bench previously imported from
`sglang.bench_serving`. Pinning sglang in install_requires was downgrading
the server-side sglang install (0.5.11 -> 0.5.2) whenever the benchmark was
installed into the same Python env. These two functions cover everything
agent-bench actually used.
"""
import os
import random
from dataclasses import dataclass
from typing import List, Optional


def get_tokenizer(name_or_path: str):
    """
    Load a HuggingFace AutoTokenizer.

    Accepts either a HF repo id (e.g. "Qwen/Qwen2.5-7B-Instruct") or an
    absolute filesystem path to a model directory. The local-path case is
    detected up-front to avoid huggingface_hub's `validate_repo_id` check,
    which rejects abs paths under newer hub versions.
    """
    from transformers import AutoTokenizer

    # If it looks like a local path, resolve and skip cache validation.
    if os.path.isdir(name_or_path) or name_or_path.startswith("/"):
        return AutoTokenizer.from_pretrained(
            os.path.abspath(name_or_path),
            trust_remote_code=True,
            local_files_only=True,
        )
    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)


@dataclass
class _RandomRequest:
    """Minimal record returned by sample_random_requests. agent-bench only reads .prompt."""
    prompt: str
    prompt_len: int
    output_len: int


def sample_random_requests(
    *,
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float = 1.0,
    tokenizer=None,
    dataset_path: str = "",
    random_sample: bool = True,
    return_text: bool = True,
    seed: Optional[int] = None,
) -> List[_RandomRequest]:
    """
    Generate `num_prompts` random prompts each tokenizing to ~`input_len` tokens.

    This is a minimal stand-in for sglang.bench_serving.sample_random_requests
    sufficient for agent-bench's _build_sharegpt_inputs() call site. Strategy:

      1. Pick a vocab subset from the tokenizer (skip specials/added).
      2. For each prompt, sample tokens, decode, and pad/trim by re-encoding
         so the final prompt has `input_len` tokens (+/- 0).
      3. Vary length per request when range_ratio < 1.0:
            sampled_len ~ uniform(input_len * range_ratio, input_len)

    Args:
      input_len: target token count per prompt.
      output_len: copied into the result for downstream max_tokens usage.
      num_prompts: how many prompts to generate.
      range_ratio: 1.0 = exact length; <1.0 widens the range.
      tokenizer: HF tokenizer (required).
      dataset_path: ignored (kept for signature compatibility).
      random_sample: must be True (kept for signature compatibility).
      return_text: must be True (kept for signature compatibility).
      seed: optional RNG seed.
    """
    if tokenizer is None:
        raise ValueError("sample_random_requests requires a tokenizer")
    if not random_sample or not return_text:
        raise ValueError("only random_sample=True, return_text=True is supported in the shim")

    rng = random.Random(seed)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    # Avoid id 0 (often pad) and the very tail (often specials/added). Picking
    # mid-range ids keeps decoded text mostly ascii filler.
    lo = 256
    hi = max(lo + 1, min(vocab_size - 1024, 50000))

    out: List[_RandomRequest] = []
    for _ in range(num_prompts):
        if range_ratio < 1.0:
            target = rng.randint(int(input_len * range_ratio), input_len)
        else:
            target = input_len

        # Oversample, decode, then re-encode and pad to hit `target` exactly.
        ids = [rng.randint(lo, hi) for _ in range(int(target * 1.3) + 16)]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        # Re-encode to know actual length after decode->encode round trip.
        retok = tokenizer.encode(text, add_special_tokens=False)
        if len(retok) >= target:
            retok = retok[:target]
            text = tokenizer.decode(retok, skip_special_tokens=True)
        else:
            # Pad with " hi," filler tokens to hit target.
            filler_ids = tokenizer.encode(" hi,", add_special_tokens=False) or [lo]
            while len(retok) < target:
                retok.extend(filler_ids)
            retok = retok[:target]
            text = tokenizer.decode(retok, skip_special_tokens=True)

        out.append(_RandomRequest(prompt=text, prompt_len=len(retok), output_len=output_len))

    return out
