#!/usr/bin/env python3
"""
LLM Throughput Simulator - Variable prompt and generation lengths

Two modes:
1. traffic-replay (default): Deterministic traffic pattern, may have concurrent requests per session
2. realistic: Response-chained sessions with max 1 in-flight request per session, automatic
   session scaling to meet QPS targets, think time between requests

Features:
- Samples prompt lengths from lognormal distribution
- Samples generation lengths from lognormal distribution
- Fixed cacheable prefix + variable random suffix
- Tracks actual length distributions (p50/p90/mean/std)
"""

import asyncio
import aiohttp
import time
import random
import argparse
import json
import re
import string
import uuid
import numpy as np
import math
import sys
from dataclasses import dataclass, field
import yaml
from typing import List, Tuple, Optional
from collections import deque
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer

# Extracted into cohesive leaf modules; re-exported here for back-compat
# with the rest of the agent_throughput.py code (run_replay, run_session_walk,
# etc. still use Colors / KeyTap / MIN_THINK_TIME etc. by bare name).
from agent.console import Colors, KeyTap  # noqa: F401,E402
from agent.sampling import (  # noqa: F401,E402
    MAX_INTER_ARRIVAL_TIME,
    MAX_RETRIES,
    MAX_THINK_TIME,
    MIN_GENERATION_TIME,
    MIN_THINK_TIME,
    draw_lognormal,
    draw_session_lifetime,
    draw_think_time,
)

# Add tracer linegraph to path for preview mode visualization
#sys.path.insert(0, str(Path(__file__).parent.parent / "tracer" / "vision"))
#from linegraph import LineGraph

# Pre-computed ASCII character set as bytes for fast random text generation
# Using numpy for vectorized operations is ~100x faster than Python loops
_ASCII_CHARS = string.ascii_letters + string.digits + " " * 10 + ".,!?-\n" * 2
_ASCII_BYTES = np.array(list(_ASCII_CHARS.encode('ascii')), dtype=np.uint8)


# Optional agent-style corpus replacement.
# If env AGENT_BENCH_CORPUS (legacy: LIGHTRACE_AGENT_CORPUS) is set,
# make_filler_seeded() returns text composed of real code files from the
# corpus instead of random ASCII.
import os as _os  # noqa: E402  -- intentionally placed near the corpus-env block
_AGENT_CORPUS_PATH = (
    _os.environ.get("AGENT_BENCH_CORPUS")
    or _os.environ.get("LIGHTRACE_AGENT_CORPUS")
)
_AGENT_CORPUS = None  # lazy-loaded


def _load_agent_corpus():
    """Load the corpus produced by build_agent_corpus.py. Returns list of dicts
    with keys 'kind' ('code'|'nl'), 'path', 'tokens'."""
    global _AGENT_CORPUS
    if _AGENT_CORPUS is None and _AGENT_CORPUS_PATH:
        with open(_AGENT_CORPUS_PATH) as _f:
            _AGENT_CORPUS = json.load(_f)
        # Pre-compute total token count.
        _AGENT_CORPUS_TOTAL = sum(len(e["tokens"]) for e in _AGENT_CORPUS)
        print(f"[agent-bench] loaded agent corpus from {_AGENT_CORPUS_PATH}: "
              f"{len(_AGENT_CORPUS)} entries, {_AGENT_CORPUS_TOTAL:,} tokens",
              flush=True)
    return _AGENT_CORPUS


def make_agent_filler_seeded(target_tokens: int, tokenizer, seed: int) -> str:
    """Build a chatml-shaped multi-turn conversation of approx target_tokens
    using real code chunks from the corpus, deterministic per seed.

    Output structure:
        <|im_start|>system\n<system blurb>\n<|im_end|>
        <|im_start|>user\n[File: path1]\n```\n<code1>\n```\n... <Question>\n<|im_end|>
        <|im_start|>assistant\n<code/explain>\n<|im_end|>
        <|im_start|>user\n<follow-up + tool output>\n<|im_end|>
        ...

    Decoded text is returned; the caller re-tokenizes (truncating to target).
    """
    if target_tokens <= 0:
        return ""
    corpus = _load_agent_corpus()
    if not corpus:
        # fallback to make_filler_seeded if no corpus
        return make_filler_seeded(target_tokens, tokenizer, seed)

    rng = np.random.default_rng(seed)
    parts = []

    sys_blurb = (
        "You are a senior software engineer assisting with the sglang codebase. "
        "Use the tools available (read_file, grep, edit, run_tests) and reason "
        "step by step. Cite file paths when proposing changes."
    )
    parts.append(f"<|im_start|>system\n{sys_blurb}\n<|im_end|>\n")

    questions = [
        "Walk me through this module step by step.",
        "Find any latent bugs or edge cases this code does not handle.",
        "Refactor the entry point to be more readable; add type hints.",
        "Add unit tests covering the main code paths.",
        "Explain how this file interacts with the scheduler.",
        "What is the time complexity of the inner loops?",
        "Suggest one or two performance optimizations.",
        "Document the public API of this module.",
    ]

    # Approximate per-char token ratio. We slice tokens (not chars) directly,
    # avoiding decode/encode roundtrips for speed.
    accum_tokens = 0

    def _pick_entry():
        return corpus[int(rng.integers(0, len(corpus)))]

    def _decode(token_ids):
        return tokenizer.decode(token_ids)

    # First user turn: pile of files + a question
    parts.append("<|im_start|>user\n")
    first_user_target = int(target_tokens * 0.55)
    first_user_used = 0
    while first_user_used < first_user_target:
        e = _pick_entry()
        if e["kind"] != "code":
            continue
        n = min(len(e["tokens"]), first_user_target - first_user_used - 50)
        if n < 80:
            break
        chunk = e["tokens"][:n]
        parts.append(f"\n[File: {e['path']}]\n```\n")
        parts.append(_decode(chunk))
        parts.append("\n```\n")
        first_user_used += n + 30
    parts.append(f"\nQuestion: {questions[int(rng.integers(0, len(questions)))]}\n")
    parts.append("<|im_end|>\n")

    # Now alternating assistant/user turns until target is exhausted.
    accum_tokens = first_user_used
    while accum_tokens < target_tokens:
        # Assistant turn (use code as filler — we want token distribution similar
        # to a real assistant explaining + showing code snippets)
        a_target = max(60, min(int((target_tokens - accum_tokens) * 0.45), 2500))
        parts.append("<|im_start|>assistant\n")
        e = _pick_entry()
        n = min(len(e["tokens"]), a_target)
        parts.append(_decode(e["tokens"][:n]))
        parts.append("\n<|im_end|>\n")
        accum_tokens += n + 20
        if accum_tokens >= target_tokens:
            break

        # User follow-up: either question or simulated tool output
        u_target = max(40, min(int((target_tokens - accum_tokens) * 0.30), 1200))
        parts.append("<|im_start|>user\n")
        if rng.random() < 0.5:
            e = _pick_entry()
            tool_chunk = e["tokens"][:max(u_target - 30, 30)]
            parts.append(f"\n[Tool: read_file({e['path']})]\n```\n")
            parts.append(_decode(tool_chunk))
            parts.append("\n```\n")
        else:
            q = questions[int(rng.integers(0, len(questions)))]
            parts.append(f"\nFollow-up: {q}\n")
            # add a small code snippet too
            e = _pick_entry()
            parts.append(_decode(e["tokens"][:max(u_target - 80, 20)]))
        parts.append("\n<|im_end|>\n")
        accum_tokens += u_target + 30

    return "".join(parts)

# NOTE: constants moved to agent/sampling.py. Kept as one-line re-exports:
# MIN_GENERATION_TIME, MIN_THINK_TIME, MAX_THINK_TIME, MAX_INTER_ARRIVAL_TIME,
# MAX_RETRIES (already imported at top of file).
QPS_TOLERANCE = 0.95  # Create new sessions when QPS drops below this fraction of target


class CachingTokenizer:
    """Ultra minimal cached tokenizer for fast token count estimation.

    Uses hash-based exact match caching for 100% accuracy.
    Main speedup: avoids re-tokenizing identical strings (common in simulations).
    """

    def __init__(self, base_tokenizer, max_cache_size: int = 10000):
        self.base_tokenizer = base_tokenizer
        self.cache = {}  # hash(text) -> actual token list
        self.max_cache_size = max_cache_size

    def encode(self, text: str, **kwargs) -> list:
        """Encode with exact-match caching. Returns token list."""
        if not text:
            return []

        # Use hash for O(1) lookup (exact match only = 100% accurate)
        text_hash = hash(text)

        if text_hash in self.cache:
            return self.cache[text_hash]

        # Cache miss - full tokenization
        tokens = self.base_tokenizer.encode(text, **kwargs)

        # Cache with simple eviction
        if len(self.cache) >= self.max_cache_size:
            # Remove first item (oldest in Python 3.7+)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[text_hash] = tokens

        return tokens

    def decode(self, tokens, **kwargs) -> str:
        """Pass through to base tokenizer."""
        return self.base_tokenizer.decode(tokens, **kwargs)

    def __getattr__(self, name):
        """Forward any other attributes to base tokenizer."""
        return getattr(self.base_tokenizer, name)


# NOTE: KeyTap and Colors moved to agent/console.py — re-exported at top
# of this file for back-compat with the rest of agent_throughput.py code.
keyboard_listener = KeyTap()


@dataclass
class ChatSession:
    """Represents a user session with pre-tokenized base text for exact token control.

    New approach: Generate one large random text upfront, pre-tokenize it once.
    When building requests, slice into pre-tokenized tokens for exact counts.
    Growing prefixes ensure cache hits as engine sees growing prefixes of same base.
    """
    id: str
    base_tokens: List[int]  # Pre-tokenized full token list (system prompt + max content)
    current_token_len: int  # Current slice position (how many tokens to use)
    max_tokens: int  # Maximum allowed tokens for this session
    tokenizer: any  # Reference to tokenizer for decoding
    created_at: float    # Wall-clock timestamp when session was created
    last_used_at: int    # Request sequence number of last use (for deterministic selection)
    request_count: int = 0  # Number of requests served by this session
    retired: bool = False   # Whether session has reached max size
    # Realistic mode fields
    in_flight: bool = False  # Whether session has a request in flight
    abandoned: bool = False  # Whether session was abandoned due to errors
    lifetime_limit: Optional[float] = None  # ChatSession lifetime limit in seconds
    forced: bool = False  # Whether session was manually created via keyboard

    @property
    def prefix_tokens(self) -> int:
        """Backward compatibility: return current token count."""
        return self.current_token_len

    @property
    def prefix_content(self) -> str:
        """Backward compatibility: decode current token slice."""
        return self.get_prompt()

    def get_prompt(self) -> str:
        """Get the current prompt by decoding the token slice."""
        tokens = self.base_tokens[:self.current_token_len]
        return self.tokenizer.decode(tokens)

    def grow(self, new_tokens: int, request_seq: int):
        """Grow the session by extending the slice length."""
        self.current_token_len += new_tokens
        # Clamp to max
        self.current_token_len = min(self.current_token_len, self.max_tokens)
        self.last_used_at = request_seq
        self.request_count += 1

    def add_content(self, new_content: str, new_tokens: int, request_seq: int):
        """Backward compatibility: grow by new_tokens (ignores new_content)."""
        self.grow(new_tokens, request_seq)

    def should_retire(self, max_prefix_tokens: int) -> bool:
        """Check if session should be retired due to size."""
        return self.current_token_len >= max_prefix_tokens

    def should_retire_lifetime(self) -> bool:
        """Check if session should be retired due to lifetime limit"""
        if self.lifetime_limit is None:
            return False
        return (time.time() - self.created_at) >= self.lifetime_limit

    def is_available(self) -> bool:
        """Check if session is available for a new request (realistic mode)"""
        return not self.retired and not self.abandoned and not self.in_flight


@dataclass
class BenchMetrics:
    """Track prefill throughput metrics over time"""
    prefill_samples: deque = field(default_factory=lambda: deque(maxlen=10000))  # (timestamp, tokens, duration, cached_tokens, generation_tps, generation_tps_mtp, generation_time, prefix_size)
    planned_prompt_lengths: List[int] = field(default_factory=list)  # Track prompt lengths at SEND time (deterministic order)
    planned_ideal_cache_hit_rates: List[float] = field(default_factory=list)  # Track ideal cache hit rates at SEND time (deterministic order)
    actual_prompt_lengths: List[int] = field(default_factory=list)  # Track actual prompt lengths at COMPLETION time
    actual_generation_lengths: List[int] = field(default_factory=list)  # Track actual generation lengths (includes reasoning)
    actual_reasoning_lengths: List[int] = field(default_factory=list)  # Track hidden reasoning tokens (e.g. <think>...) separately
    actual_cache_hit_rates: List[float] = field(default_factory=list)  # Track per-request cache hit rates
    ideal_cache_hit_rates: List[float] = field(default_factory=list)  # Track ideal cache hit rates (assuming no eviction)
    actual_ttfts: List[float] = field(default_factory=list)  # Track TTFT (time to first token) in seconds
    actual_tpots: List[float] = field(default_factory=list)  # Track TPOT (time per output token, excl. first) in seconds
    actual_acceptance_lengths: List[float] = field(default_factory=list)  # Track per-request average acceptance length
    actual_prefix_sizes: List[int] = field(default_factory=list)  # Track prefix sizes used per request
    request_timeline: List[Tuple[float, bool, str, bool]] = field(default_factory=list)  # (timestamp, is_new_session, session_id, is_forced)
    inter_arrival_times: List[float] = field(default_factory=list)  # Track inter-arrival times between requests
    requests_sent: int = 0
    requests_completed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    def add_prefill(self, tokens: int, duration: float, cached_tokens: int = 0, generation_tps: float = 0.0, generation_tps_mtp: float = 0.0, actual_gen_length: int = 0, generation_time: float = 0.0, prefix_size: int = 0, reasoning_tokens: int = 0):
        """Add a prefill measurement with optional cache info"""
        now = time.time()
        self.prefill_samples.append((now, tokens, duration, cached_tokens, generation_tps, generation_tps_mtp, generation_time, prefix_size))
        self.actual_prompt_lengths.append(tokens)
        self.actual_generation_lengths.append(actual_gen_length)
        self.actual_reasoning_lengths.append(reasoning_tokens)
        self.actual_prefix_sizes.append(prefix_size)
        # Calculate per-request cache hit rates
        actual_rate = cached_tokens / tokens if tokens > 0 else 0.0
        ideal_rate = prefix_size / tokens if tokens > 0 else 0.0
        self.actual_cache_hit_rates.append(actual_rate)
        self.ideal_cache_hit_rates.append(ideal_rate)
        self.actual_ttfts.append(duration)
        # TPOT (time per output token, excluding the first one which is TTFT).
        # Only record when we have >=2 output tokens AND a non-trivial generation
        # window — otherwise the value is dominated by jitter / scheduling noise.
        if actual_gen_length > 1 and generation_time >= MIN_GENERATION_TIME:
            self.actual_tpots.append(generation_time / (actual_gen_length - 1))

    def add_acceptance_length(self, avg_acc_len: float):
        """Record average acceptance length for a completed request"""
        self.actual_acceptance_lengths.append(avg_acc_len)
        self.requests_completed += 1

    def get_window_throughput(self, window_secs: float = 5.0) -> Tuple[float, int]:
        """Get prefill tokens/sec over a time window"""
        now = time.time()
        cutoff = now - window_secs

        recent = [(t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize) for t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize in self.prefill_samples if t > cutoff]
        if not recent:
            return 0.0, 0

        total_tokens = sum(tokens for _, tokens, _, _, _, _, _, _ in recent)
        total_duration = now - recent[0][0] if recent else 1

        return total_tokens / max(total_duration, 0.001), len(recent)

    def get_cache_hit_rate(self, window_secs: float = 5.0) -> float:
        """Get cache hit rate over a time window"""
        now = time.time()
        cutoff = now - window_secs

        recent = [(t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize) for t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize in self.prefill_samples if t > cutoff]
        if not recent:
            return 0.0

        total_tokens = sum(tokens for _, tokens, _, _, _, _, _, _ in recent)
        total_cached = sum(cached for _, _, _, cached, _, _, _, _ in recent)

        if total_tokens == 0:
            return 0.0

        return total_cached / total_tokens

    def get_ideal_cache_hit_rate(self, window_secs: float = 5.0) -> float:
        """Get ideal cache hit rate over a time window (assuming no eviction)"""
        now = time.time()
        cutoff = now - window_secs

        recent = [(t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize) for t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize in self.prefill_samples if t > cutoff]
        if not recent:
            return 0.0

        total_tokens = sum(tokens for _, tokens, _, _, _, _, _, _ in recent)
        total_prefix = sum(psize for _, _, _, _, _, _, _, psize in recent)

        if total_tokens == 0:
            return 0.0

        return total_prefix / total_tokens

    def get_generation_tps(self, window_secs: float = 5.0, min_gen_time: float = MIN_GENERATION_TIME) -> float:
        """Get average generation TPS (MTP compensated) over a time window"""
        now = time.time()
        cutoff = now - window_secs

        recent = [(t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize)
                  for t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize in self.prefill_samples
                  if t > cutoff and gen_tps_mtp > 0 and gen_time >= min_gen_time]

        if not recent:
            return 0.0

        total_tps = sum(gen_tps_mtp for _, _, _, _, _, gen_tps_mtp, _, _ in recent)
        return total_tps / len(recent)

    def get_in_flight(self) -> int:
        """Get number of in-flight requests"""
        return self.requests_sent - self.requests_completed - self.errors

    def get_peak_throughput(self, window_secs: float = 5.0) -> Tuple[float, float]:
        """Find peak throughput by scanning all historical samples.

        Returns (peak_tps, timestamp) where timestamp is the end of the peak window.
        """
        if len(self.prefill_samples) < 2:
            return 0.0, 0.0

        # Sort samples by timestamp
        sorted_samples = sorted(self.prefill_samples, key=lambda x: x[0])

        best_tps = 0.0
        best_time = 0.0

        # Slide window through all samples
        for i, (end_time, _, _, _, _, _, _, _) in enumerate(sorted_samples):
            start_time = end_time - window_secs

            # Get all samples in this window
            window_tokens = sum(
                tokens for t, tokens, _, _, _, _, _, _ in sorted_samples
                if start_time < t <= end_time
            )

            # Calculate throughput for this window
            tps = window_tokens / window_secs

            if tps > best_tps:
                best_tps = tps
                best_time = end_time

        return best_tps, best_time


class RateTracker:
    """Monitor actual QPS over a rolling window"""

    def __init__(self, window_secs: float = 30.0):
        self.window_secs = window_secs
        self.request_times: deque = deque()
        self.lock = asyncio.Lock()

    async def record_request(self):
        """Record a request completion"""
        async with self.lock:
            self.request_times.append(time.time())

    async def get_qps(self) -> float:
        """Get current QPS over the rolling window"""
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_secs
            # Remove old entries
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()
            # Calculate QPS
            if not self.request_times:
                return 0.0
            return len(self.request_times) / self.window_secs


class LatencyTracker:
    """Track rolling average of response times for session scaling"""

    def __init__(self, window_size: int = 50, initial_estimate: float = 2.0):
        self.samples: deque = deque(maxlen=window_size)
        self.initial_estimate = initial_estimate

    def record(self, response_time: float):
        """Record a response time sample"""
        self.samples.append(response_time)

    def get_average(self) -> float:
        """Get rolling average response time"""
        if not self.samples:
            return self.initial_estimate
        return sum(self.samples) / len(self.samples)


@dataclass
class TurnPlan:
    """Pre-sampled random values for deterministic request execution.
    
    Only numeric random values are sampled in the main loop (fast).
    Text content is generated in async tasks using per-request seeded RNG (deterministic).
    """
    request_id: int
    new_session_roll: float  # Compared against new_session_rate to decide new vs existing
    session_select_roll: float  # Used for weighted session selection (0-1)
    new_tokens: int  # Pre-sampled token count to add to prompt
    generation_length: int  # Pre-sampled generation length
    initial_prefix_tokens: int  # Pre-sampled initial prefix size (0 if not needed)


def make_filler(target_tokens: int, tokenizer) -> str:
    """Generate random ASCII text with exact token count using tokenizer"""
    # Generate excess characters as buffer (8 chars per token to be safe)
    buffer_chars = target_tokens * 8
    ascii_chars = string.ascii_letters + string.digits + " " * 10 + ".,!?-\n" * 2
    raw_text = "".join(random.choice(ascii_chars) for _ in range(buffer_chars))
    
    # Tokenize, trim to exact count, decode back
    # tokens = tokenizer.encode(raw_text, add_special_tokens=False)[:target_tokens]
    # return tokenizer.decode(tokens)

    # average 4 chars per token
    return raw_text[:(target_tokens * 4)]


def make_filler_seeded(target_tokens: int, tokenizer, seed: int) -> str:
    """Generate random ASCII text with exact token count using a seeded RNG.

    This allows deterministic text generation in async tasks without affecting
    the global RNG state. Each request uses a unique seed based on its ID.

    Optimized: Uses numpy vectorized operations which is ~100x faster than
    Python loops. Generates random indices into pre-computed byte array.

    If AGENT_BENCH_CORPUS (legacy: LIGHTRACE_AGENT_CORPUS) env var is set,
    delegates to make_agent_filler_seeded which produces realistic chatml-shaped
    agent prompts from a code corpus.
    """
    if target_tokens <= 0:
        return ""
    if _AGENT_CORPUS_PATH:
        return make_agent_filler_seeded(target_tokens, tokenizer, seed)

    # average 4 chars per token - generate exactly what we need
    num_chars = target_tokens * 4
    # num_chars = target_tokens * 8

    # Use numpy random with seed for reproducibility
    rng = np.random.default_rng(seed)

    # Generate random indices into our pre-computed character array
    # This is ~100x faster than Python's random.choice() loop
    indices = rng.integers(0, len(_ASCII_BYTES), size=num_chars, dtype=np.int32)

    # Use fancy indexing to get characters, then decode from bytes
    raw_text = _ASCII_BYTES[indices].tobytes().decode('ascii')
    # return raw_text

    tokens = tokenizer.encode(raw_text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(tokens)


def make_shared_content(target_cacheable_tokens: int, system_prompt: str, tokenizer) -> str:
    """
    Generate shared content that will be reused across all requests.
    Includes system prompt + random padding to reach target token count.
    """
    # Get actual token count of system prompt
    system_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    # system_tokens = len(system_prompt) / 4

    if target_cacheable_tokens <= system_tokens:
        # Trim system prompt to target token count
        tokens = tokenizer.encode(system_prompt, add_special_tokens=False)[:target_cacheable_tokens]
        return tokenizer.decode(tokens)
        # return system_prompt[:(target_cacheable_tokens * 4)]
    else:
        # Add random padding after system prompt to reach target
        padding_tokens = target_cacheable_tokens - system_tokens
        padding = make_filler(padding_tokens, tokenizer)
        return system_prompt + "\n\n" + padding


# NOTE: draw_lognormal, draw_session_lifetime, draw_think_time moved to
# agent/sampling.py — re-exported at top of this file.


def draw_turn_plan(
    request_id: int,
    new_tokens_mean: float,
    new_tokens_median: float,
    generation_length_mean: float,
    generation_length_median: float,
    initial_prefix_mean: int,
    initial_prefix_median: int,
    system_prompt_tokens: int,
    max_prompt_tokens: int,
    max_generation_length: int = 20000
) -> TurnPlan:
    """Sample all random NUMERIC values for a request in deterministic order.
    
    This function MUST be called from the main sequential loop, not from async tasks.
    The order of random sampling here determines the reproducible sequence.
    
    Text content is NOT generated here (too slow). Instead, async tasks generate
    text using make_filler_seeded() with a seed derived from request_id.
    """
    # 1. Sample new session decision roll
    new_session_roll = random.random()
    
    # 2. Sample session selection roll (for weighted choice)
    session_select_roll = random.random()
    
    # 3. Sample new tokens count
    new_tokens = draw_lognormal(new_tokens_mean, new_tokens_median)
    
    # 4. Sample generation length
    generation_length = draw_lognormal(generation_length_mean, generation_length_median)
    generation_length = min(generation_length, max_generation_length)
    
    # 5. Sample initial prefix token count (always sample to maintain RNG order)
    # Note: uses max(0, ...) to match original behavior (can be 0)
    if initial_prefix_mean > 0 and initial_prefix_median > 0:
        mu = math.log(initial_prefix_median)
        if initial_prefix_mean > initial_prefix_median:
            sigma = math.sqrt(2 * math.log(initial_prefix_mean / initial_prefix_median))
        else:
            sigma = 0.1
        initial_prefix_tokens = max(0, int(np.random.lognormal(mean=mu, sigma=sigma)))
    else:
        initial_prefix_tokens = 0

    # Clamp to enforce system_prompt_tokens + initial_prefix_tokens <= max_prompt_tokens
    max_initial_prefix = max(0, max_prompt_tokens - system_prompt_tokens)
    initial_prefix_tokens = min(initial_prefix_tokens, max_initial_prefix)

    return TurnPlan(
        request_id=request_id,
        new_session_roll=new_session_roll,
        session_select_roll=session_select_roll,
        new_tokens=new_tokens,
        generation_length=generation_length,
        initial_prefix_tokens=initial_prefix_tokens,
    )



def pick_session_with_decay(sessions: List[ChatSession], current_request_seq: int, selection_roll: float, decay_lambda: float = 0.02, min_weight: float = 0.1) -> ChatSession:
    """
    Select an active (non-retired) session with exponential decay bias towards recent sessions.

    Args:
        sessions: List of ChatSession objects
        current_request_seq: Current request sequence number (for deterministic selection)
        selection_roll: Pre-sampled random value [0, 1) for deterministic selection
        decay_lambda: Decay rate for exponential weighting (higher = stronger recency bias)
                      Default 0.02 gives half-life of ~35 requests
        min_weight: Minimum weight floor to keep all sessions viable (default 0.1)

    Returns:
        Selected ChatSession object, or None if no active sessions
    """
    # Filter to only active (non-retired) sessions
    active_sessions = [s for s in sessions if not s.retired]

    if not active_sessions:
        return None

    if len(active_sessions) == 1:
        return active_sessions[0]

    # Calculate age-based weights with exponential decay (based on request sequence)
    # Use min_weight floor to prevent old sessions from becoming negligible
    weights = []
    for session in active_sessions:
        age = current_request_seq - session.last_used_at
        weight = max(min_weight, np.exp(-decay_lambda * age))
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # Select using pre-sampled roll value (CDF-based selection for determinism)
    cumsum = 0.0
    for i, prob in enumerate(probabilities):
        cumsum += prob
        if selection_roll < cumsum:
            return active_sessions[i]
    # Fallback to last session (handles floating point edge cases)
    return active_sessions[-1]


def spawn_session(
    sessions: List[ChatSession],
    system_prompt: str,
    system_prompt_tokens: int,
    request_seq: int,
    tokenizer,
    max_prompt_tokens: int,
    seed: int,
    initial_prefix_tokens: int = 0,
    max_sessions: int = 100,
    lifetime_limit: Optional[float] = None,
    forced: bool = False,
) -> ChatSession:
    """
    Create a new session with pre-tokenized base text for exact token control.

    New approach: Generate ONE large random text upfront (large enough for max_prompt_tokens),
    pre-tokenize it once, then slice into it for requests. This ensures:
    - Exact token counts (no drift, no truncation needed)
    - Cache hits (engine sees growing prefixes of the same base string)
    - Simple implementation (no complex tracking or snapshots)

    Args:
        sessions: List of existing sessions
        system_prompt: The system prompt text
        system_prompt_tokens: Token count of system prompt
        request_seq: Request sequence number for last_used_at
        tokenizer: Tokenizer for encoding/decoding
        max_prompt_tokens: Maximum tokens allowed for this session
        seed: Seed for deterministic random text generation
        initial_prefix_tokens: Initial prefix size (tokens to start with)
        max_sessions: Maximum sessions to keep
        lifetime_limit: Optional session lifetime limit in seconds
        forced: Whether session was manually created via keyboard
    """
    now = time.time()
    session_id = str(uuid.uuid4())[:8]

    # Generate one large base text that covers the full session capacity
    # System prompt + enough random content to reach max_prompt_tokens
    content_tokens_needed = max_prompt_tokens - system_prompt_tokens

    # Generate random content large enough (with margin) and tokenize
    base_text = system_prompt + "\n\n" + make_filler_seeded(
        content_tokens_needed, tokenizer, seed
    )

    # Pre-tokenize the entire base text once
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)

    # Ensure we have enough tokens (truncate if over)
    if len(base_tokens) > max_prompt_tokens:
        base_tokens = base_tokens[:max_prompt_tokens]

    # Initial slice position: system prompt + initial prefix
    initial_len = system_prompt_tokens + initial_prefix_tokens
    initial_len = min(initial_len, len(base_tokens))

    session = ChatSession(
        id=session_id,
        base_tokens=base_tokens,
        current_token_len=initial_len,
        max_tokens=max_prompt_tokens,
        tokenizer=tokenizer,
        created_at=now,
        last_used_at=request_seq,
        request_count=0,
        retired=False,
        in_flight=False,
        abandoned=False,
        lifetime_limit=lifetime_limit,
        forced=forced,
    )

    sessions.append(session)

    # Remove oldest retired sessions if exceeding capacity
    if len(sessions) > max_sessions:
        retired_sessions = [s for s in sessions if s.retired]
        if retired_sessions:
            oldest_retired = min(retired_sessions, key=lambda s: s.last_used_at)
            sessions.remove(oldest_retired)
        else:
            oldest = min(sessions, key=lambda s: s.last_used_at)
            sessions.remove(oldest)

    return session


async def dispatch_turn(
    http_session: aiohttp.ClientSession,
    server_url: str,
    model: str,
    sessions: List[ChatSession],
    system_prompt: str,
    system_prompt_tokens: int,
    new_session_rate: float,
    metrics: BenchMetrics,
    plan: TurnPlan,
    tokenizer,
    base_seed: int,
    api_key: str = None,
    acc_len: float = 3.0,
    mtp_overhead_factor: float = 1.0,
    max_prompt_tokens: int = 200000,
    session_decay_lambda: float = 0.02,
    force_new_session: bool = False,
    ignore_eos: bool = True
):
    """Send a request using session model with pre-tokenized base text.

    New approach: Sessions have pre-tokenized base text. Just grow the slice
    and decode to get the prompt. Exact token counts, no truncation needed.
    """

    metrics.requests_sent += 1

    # Seed for new session creation (if needed)
    session_seed = base_seed + plan.request_id * 1000

    # Decide whether to create new session or use existing (using pre-sampled roll)
    active_sessions = [s for s in sessions if not s.retired]
    use_new_session = force_new_session or plan.new_session_roll < new_session_rate or not active_sessions

    # Track request submission time and session type (session_id added after session selection)
    request_timestamp = time.time()
    is_new_session = use_new_session

    if use_new_session:
        # Create a new session with pre-tokenized base text
        selected_session = spawn_session(
            sessions, system_prompt, system_prompt_tokens,
            plan.request_id,
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            seed=session_seed,
            initial_prefix_tokens=plan.initial_prefix_tokens,
            max_sessions=100
        )
    else:
        # Select existing session with pre-sampled roll value
        selected_session = pick_session_with_decay(
            sessions, plan.request_id, plan.session_select_roll,
            decay_lambda=session_decay_lambda
        )
        if selected_session is None:
            # All sessions retired, create new one
            selected_session = spawn_session(
                sessions, system_prompt, system_prompt_tokens,
                plan.request_id,
                tokenizer=tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                seed=session_seed,
                initial_prefix_tokens=plan.initial_prefix_tokens,
                max_sessions=100
            )
            is_new_session = True

    # Now append to timeline with session_id
    is_forced = force_new_session and is_new_session
    metrics.request_timeline.append((request_timestamp, is_new_session, selected_session.id, is_forced))

    # Get current prefix tokens before growing
    current_prefix_tokens = selected_session.prefix_tokens

    # Use pre-sampled new tokens count, clamped to available space
    new_tokens = plan.new_tokens
    available_space = max_prompt_tokens - current_prefix_tokens
    if new_tokens > available_space:
        new_tokens = max(1, available_space)

    # Record metrics at SEND time for deterministic ordering
    total_prompt_tokens = current_prefix_tokens + new_tokens
    metrics.planned_prompt_lengths.append(total_prompt_tokens)

    # Record ideal cache hit rate at send time (prefix_tokens / total_tokens)
    planned_ideal_cache_rate = current_prefix_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics.planned_ideal_cache_hit_rates.append(planned_ideal_cache_rate)

    # Grow the session by new_tokens (extends the slice into pre-tokenized base)
    selected_session.grow(new_tokens, plan.request_id)

    # Get the full prompt by decoding the current token slice
    # This is exact - no drift, no truncation needed
    full_content = selected_session.get_prompt()

    # Check if session should be retired
    if selected_session.should_retire(max_prompt_tokens):
        selected_session.retired = True

    # Use pre-sampled generation length
    generation_length = plan.generation_length

    messages = [{"role": "user", "content": full_content}]

    payload = {
        # "request_id": str(uuid.uuid4()),  # comment out for current (12/19) Dynamo PD setup
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": generation_length,
        "temperature": 0.0,
        "user": selected_session.id,
        "ignore_eos": ignore_eos,
    }

    url = f"{server_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start_time = time.time()
    ttft = None
    cached_tokens = 0
    reasoning_tokens = 0
    actual_prompt_tokens = 0
    completion_tokens = 0
    full_response = ""  # Accumulate response for accurate token counting
    chunk_token_counts = []  # Track tokens per chunk for acceptance length calculation

    try:
        async with http_session.post(url, json=payload, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=240)) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"{Colors.RED}Request {plan.request_id} failed: HTTP {resp.status} - {error_text[:200]}{Colors.END}")
                metrics.errors += 1
                return False

            async for line in resp.content:
                if line:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if not data or not isinstance(data, dict):
                                continue

                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                resp_content = delta.get("content", "")
                                resp_reason = delta.get("reasoning_content") or delta.get("reasoning")
                                if (resp_content or resp_reason) and ttft is None:
                                    ttft = time.time() - start_time
                                if resp_reason:
                                    full_response += resp_reason
                                if resp_content:
                                    full_response += resp_content
                                    # Count tokens in this chunk for acceptance length
                                    chunk_tokens = len(tokenizer.encode(resp_content, add_special_tokens=False)) - 1  # Subtract 1 for target model token
                                    if chunk_tokens > 0:
                                        chunk_token_counts.append(chunk_tokens)


                            if "usage" in data and data["usage"]:
                                usage = data["usage"]
                                if isinstance(usage, dict):
                                    if "prompt_tokens" in usage:
                                        actual_prompt_tokens = usage.get("prompt_tokens", 0)
                                    if "cache_read_input_tokens" in usage:
                                        cached_tokens = usage.get("cache_read_input_tokens") or 0
                                    elif "prompt_tokens_details" in usage:
                                        details = usage["prompt_tokens_details"]
                                        if isinstance(details, dict):
                                            cached_tokens = details.get("cached_tokens") or 0
                                    if "reasoning_tokens" in usage:
                                        reasoning_tokens = usage.get("reasoning_tokens") or 0
                        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                            # Catch malformed SSE chunks / missing fields, but let
                            # genuine network or programming errors propagate. The
                            # prior `except (json.JSONDecodeError, Exception)` made
                            # `Exception` swallow real bugs (JSONDecodeError was
                            # already a subclass).
                            pass

            # Tokenize accumulated response for accurate completion token count
            completion_tokens = len(tokenizer.encode(full_response, add_special_tokens=False)) if full_response else 0

            if ttft is None:
                ttft = time.time() - start_time

            if actual_prompt_tokens > 0:
                tokens_to_record = actual_prompt_tokens
            else:
                tokens_to_record = total_prompt_tokens
                if not hasattr(metrics, "_warned_no_prompt_tokens"):
                    metrics._warned_no_prompt_tokens = True
                    print(f"{Colors.YELLOW}INFO: Server not returning prompt_tokens - using planned prompt length for TPM calculation{Colors.END}")
            end_time = time.time()
            total_time = end_time - start_time
            generation_time = total_time - ttft if ttft and total_time > ttft else 0.0
            generation_tps = completion_tokens / generation_time if generation_time > 0 and completion_tokens > 0 else 0.0
            generation_tps_mtp = (completion_tokens * acc_len) / (generation_time * mtp_overhead_factor) if generation_time > 0 and completion_tokens > 0 else 0.0

            metrics.add_prefill(tokens_to_record, ttft, cached_tokens, generation_tps, generation_tps_mtp,
                              completion_tokens, generation_time, current_prefix_tokens,
                              reasoning_tokens=reasoning_tokens)

            # Calculate and record acceptance length
            if chunk_token_counts:
                avg_acceptance_length = sum(chunk_token_counts) / len(chunk_token_counts)
                metrics.add_acceptance_length(avg_acceptance_length)
            else:
                # Request completed but no content returned
                metrics.requests_completed += 1

            # ChatSession state was already updated optimistically before sending
            return True

    except asyncio.TimeoutError:
        metrics.errors += 1
        if not hasattr(metrics, '_warned_timeout'):
            metrics._warned_timeout = True
            print(f"{Colors.YELLOW}WARNING: Request timeout - traffic timing may diverge from seed (non-deterministic){Colors.END}")
        print(f"{Colors.RED}Request {plan.request_id} timed out{Colors.END}")
        return False
    except Exception as e:
        metrics.errors += 1
        print(f"{Colors.RED}Request {plan.request_id} error: {e}{Colors.END}")
        return False


async def ramp_requests(
    server_url: str,
    model: str,
    sessions: List[ChatSession],
    system_prompt: str,
    system_prompt_tokens: int,
    new_tokens_mean: float,
    new_tokens_median: float,
    new_session_rate: float,
    initial_qps: float,
    max_qps: float,
    ramp_duration_secs: float,
    sustain_duration_secs: float,
    metrics: BenchMetrics,
    tokenizer,
    base_seed: int,
    generation_length_mean: float = 1,
    generation_length_median: float = 1,
    acc_len: float = 3.0,
    mtp_overhead_factor: float = 1.0,
    api_key: str = None,
    max_prompt_tokens: int = 200000,
    use_poisson: bool = False,
    poisson_shape: float = 1.0,
    initial_prefix_mean: int = 0,
    initial_prefix_median: int = 0,
    max_inflight: int = None,
    session_decay_lambda: float = 0.02,
    ignore_eos: bool = True
):
    """Ramp up request rate from initial_qps to max_qps over ramp_duration.
    
    All random sampling happens in this main loop before async task dispatch,
    ensuring deterministic RNG consumption order regardless of async execution timing.
    """

    connector = aiohttp.TCPConnector(limit=1000)
    async with aiohttp.ClientSession(connector=connector) as http_session:
        start_time = time.time()
        request_id = 0
        tasks = []

        print(f"{Colors.CYAN}Starting ramp: {initial_qps:.2f} -> {max_qps:.2f} QPS over {ramp_duration_secs:.0f}s{Colors.END}")
        print(f"{Colors.CYAN}Then sustain {max_qps:.2f} QPS for {sustain_duration_secs:.0f}s{Colors.END}")
        if max_inflight:
            print(f"{Colors.CYAN}Backpressure: pause when in-flight > {max_inflight}{Colors.END}")
        print()

        while True:
            elapsed = time.time() - start_time

            if elapsed < ramp_duration_secs:
                progress = elapsed / ramp_duration_secs
                current_qps = initial_qps + (max_qps - initial_qps) * progress
            elif elapsed < ramp_duration_secs + sustain_duration_secs:
                current_qps = max_qps
            else:
                break

            mean_interval = 1.0 / current_qps
            if use_poisson:
                interval = np.random.gamma(shape=poisson_shape, scale=mean_interval/poisson_shape)
            else:
                interval = mean_interval

            # Backpressure: wait if too many requests in flight
            if max_inflight is not None:
                if metrics.get_in_flight() >= max_inflight:
                    if not hasattr(metrics, '_warned_max_inflight'):
                        metrics._warned_max_inflight = True
                        print(f"{Colors.YELLOW}WARNING: Hit max_inflight ({max_inflight}) - traffic timing may diverge from seed (non-deterministic){Colors.END}")
                    while metrics.get_in_flight() >= max_inflight:
                        await asyncio.sleep(0.01)  # Check every 10ms

            # Pre-sample all random values BEFORE creating async task (deterministic order)
            plan = draw_turn_plan(
                request_id=request_id,
                new_tokens_mean=new_tokens_mean,
                new_tokens_median=new_tokens_median,
                generation_length_mean=generation_length_mean,
                generation_length_median=generation_length_median,
                initial_prefix_mean=initial_prefix_mean,
                initial_prefix_median=initial_prefix_median,
                system_prompt_tokens=system_prompt_tokens,
                max_prompt_tokens=max_prompt_tokens,
            )

            task = asyncio.create_task(
                dispatch_turn(
                    http_session, server_url, model, sessions, system_prompt, system_prompt_tokens,
                    new_session_rate, metrics, plan, tokenizer, base_seed, api_key, acc_len,
                    mtp_overhead_factor, max_prompt_tokens, session_decay_lambda,
                    ignore_eos=ignore_eos
                )
            )
            tasks.append(task)
            request_id += 1

            # Check for manually triggered new session requests (via n keypress)
            pending_new_sessions = keyboard_listener.get_pending_count()
            for _ in range(pending_new_sessions):
                # Pre-sample for forced request too
                forced_plan = draw_turn_plan(
                    request_id=request_id,
                    new_tokens_mean=new_tokens_mean,
                    new_tokens_median=new_tokens_median,
                    generation_length_mean=generation_length_mean,
                    generation_length_median=generation_length_median,
                    initial_prefix_mean=initial_prefix_mean,
                    initial_prefix_median=initial_prefix_median,
                    system_prompt_tokens=system_prompt_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                )
                forced_task = asyncio.create_task(
                    dispatch_turn(
                        http_session, server_url, model, sessions, system_prompt, system_prompt_tokens,
                        new_session_rate, metrics, forced_plan, tokenizer, base_seed, api_key, acc_len,
                        mtp_overhead_factor, max_prompt_tokens, session_decay_lambda,
                        force_new_session=True,
                        ignore_eos=ignore_eos
                    )
                )
                tasks.append(forced_task)
                request_id += 1

            if len(tasks) > 100:
                done_tasks = [t for t in tasks if t.done()]
                for t in done_tasks:
                    tasks.remove(t)

            metrics.inter_arrival_times.append(interval)
            await asyncio.sleep(interval)

        print(f"\n{Colors.YELLOW}Waiting for remaining requests to complete...{Colors.END}")
        pending = [t for t in tasks if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=180)


async def display_metrics_live(
    metrics: BenchMetrics, 
    duration_secs: float, 
    window_size: float = 15.0,
    target_cache_rate: float = 0.0
):
    """Display live metrics during benchmark"""
    start_time = time.time()

    while time.time() - start_time < duration_secs:
        elapsed = time.time() - metrics.start_time

        # Get throughput over configured window
        throughput_window, count_window = metrics.get_window_throughput(window_size)

        # Get throughput over 1s window for more responsive display
        throughput_1s, count_1s = metrics.get_window_throughput(1.0)

        # Get cache hit rate
        cache_hit_rate = metrics.get_cache_hit_rate(window_size)

        # Get generation TPS (MTP compensated)
        gen_tps = metrics.get_generation_tps(window_size)

        in_flight = metrics.get_in_flight()

        print(f"\r{Colors.BOLD}[{elapsed:6.1f}s]{Colors.END} "
              f"{Colors.GREEN}Prefill: {throughput_1s:8,.0f} tok/s (1s){Colors.END} | "
              f"{Colors.CYAN}{throughput_window:8,.0f} tok/s ({window_size:.0f}s){Colors.END} | "
              f"{Colors.YELLOW}Cache: {cache_hit_rate*100:5.1f}%{Colors.END} | "
              f"{Colors.GREEN}Gen: {gen_tps:6.1f} tok/s{Colors.END} | "
              f"Reqs: {metrics.requests_completed:5d}/{metrics.requests_sent:5d} | "
              f"In-flight: {in_flight:4d} | "
              f"Errors: {metrics.errors:3d}",
              end="", flush=True)

        await asyncio.sleep(0.5)

    print()  # Newline after live display


def percentiles(values: List[float], percentiles: List[float]) -> List[float]:
    """Calculate percentiles from a list of values"""
    if not values:
        return [0.0] * len(percentiles)
    sorted_values = sorted(values)
    n = len(sorted_values)
    return [sorted_values[int(n * p)] if n > 0 else 0.0 for p in percentiles]


def compute_phase_breakdown(metrics, benchmark_start: float, ramp_duration_secs: float, sustain_duration_secs: float) -> List[dict]:
    """Slice prefill samples into ramp / sustain / drain phases and compute TPM breakdowns.

    Phases are defined by wall-clock offset from benchmark_start:
      ramp     t in [0, ramp_end)
      sustain  t in [ramp_end, sustain_end)
      drain    t in [sustain_end, +inf)  -- requests sent during sustain that finished later
    """
    ramp_end = ramp_duration_secs
    sustain_end = ramp_duration_secs + sustain_duration_secs
    phases = [
        ("ramp",    0.0,        ramp_end,     ramp_duration_secs),
        ("sustain", ramp_end,   sustain_end,  sustain_duration_secs),
        ("drain",   sustain_end, float("inf"), None),  # duration inferred from observed window
    ]
    gen_lengths = list(metrics.actual_generation_lengths)
    reason_lengths = list(metrics.actual_reasoning_lengths)
    samples = list(metrics.prefill_samples)
    out = []
    for name, lo, hi, planned in phases:
        bucket = []
        for idx, s in enumerate(samples):
            t_abs = s[0]
            offset = t_abs - benchmark_start
            if lo <= offset < hi:
                gen_len = gen_lengths[idx] if idx < len(gen_lengths) else 0
                rea_len = reason_lengths[idx] if idx < len(reason_lengths) else 0
                bucket.append((s, gen_len, rea_len, offset))
        if bucket:
            first_off = min(b[3] for b in bucket)
            last_off  = max(b[3] for b in bucket)
            observed = last_off - first_off
        else:
            first_off, last_off, observed = 0.0, 0.0, 0.0
        if planned is not None:
            duration = planned
        else:
            # Drain window: require at least a couple seconds of observations
            # before reporting a rate, otherwise the number is meaningless.
            duration = observed if observed >= 2.0 else 0.0
        total_input    = sum(s[1]  for s, _, _, _ in bucket)
        total_cached   = sum(s[3]  for s, _, _, _ in bucket)
        total_gen      = sum(g     for _, g, _, _ in bucket)
        total_reason   = sum(r     for _, _, r, _ in bucket)
        ttfts          = [s[2]     for s, _, _, _ in bucket]
        # TPOT = generation_time / (gen_len - 1); skip noisy samples (gen<=1 or gen_time too short)
        tpots          = [s[6] / (g - 1) for s, g, _, _ in bucket
                          if g > 1 and s[6] >= MIN_GENERATION_TIME]
        uncached = max(0, total_input - total_cached)
        visible_gen = max(0, total_gen - total_reason)
        def per_min(x):
            return (x * 60.0 / duration) if duration > 0 else 0.0
        out.append({
            "phase": name,
            "duration_s": duration,
            "completed": len(bucket),
            "qps": (len(bucket) / duration) if duration > 0 else 0.0,
            "input_tokens": total_input,
            "cached_tokens": total_cached,
            "uncached_tokens": uncached,
            "gen_tokens": total_gen,
            "reasoning_tokens": total_reason,
            "visible_tokens": visible_gen,
            "input_tpm": per_min(total_input),
            "cached_tpm": per_min(total_cached),
            "uncached_tpm": per_min(uncached),
            "gen_tpm": per_min(total_gen),
            "reasoning_tpm": per_min(total_reason),
            "visible_tpm": per_min(visible_gen),
            "cache_hit_rate": (total_cached / total_input) if total_input > 0 else 0.0,
            "ttft_p50_ms": (percentiles(ttfts, [0.5])[0] * 1000) if ttfts else 0.0,
            "ttft_p90_ms": (percentiles(ttfts, [0.9])[0] * 1000) if ttfts else 0.0,
            "ttft_p99_ms": (percentiles(ttfts, [0.99])[0] * 1000) if ttfts else 0.0,
            "tpot_p50_ms": (percentiles(tpots, [0.5])[0] * 1000) if tpots else 0.0,
            "tpot_p90_ms": (percentiles(tpots, [0.9])[0] * 1000) if tpots else 0.0,
            "tpot_p99_ms": (percentiles(tpots, [0.99])[0] * 1000) if tpots else 0.0,
        })
    return out


def print_phase_breakdown(phases: List[dict], num_gpus: int = 1) -> None:
    """Pretty-print the ramp/sustain/drain TPM table.

    gen TPM splits into `visible` (user-facing completion) and `reason`
    (server-reported reasoning/<think> tokens) when the server populates
    `usage.reasoning_tokens`; otherwise reason stays 0.
    """
    print(f"\n{Colors.BOLD}Phase Throughput Breakdown (input TPM includes cache; uncached = actual prefill work):{Colors.END}")
    header = (f"  {'phase':<8} {'dur(s)':>7} {'reqs':>5} {'qps':>5} "
              f"{'input TPM':>12} {'cached TPM':>12} {'uncached TPM':>14} "
              f"{'visible TPM':>12} {'reason TPM':>11} "
              f"{'cache%':>7} {'TTFT p50':>9} {'TTFT p90':>9} "
              f"{'TPOT p50':>9} {'TPOT p90':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in phases:
        if p['duration_s'] <= 0 or p['completed'] == 0:
            print(f"  {p['phase']:<8} {'n/a':>7} {p['completed']:>5d} "
                  f"{'n/a':>5} {'n/a':>12} {'n/a':>12} {'n/a':>14} "
                  f"{'n/a':>12} {'n/a':>11} "
                  f"{'n/a':>7} {'n/a':>9} {'n/a':>9} "
                  f"{'n/a':>9} {'n/a':>9}")
            continue
        print(f"  {p['phase']:<8} {p['duration_s']:>7.1f} {p['completed']:>5d} "
              f"{p['qps']:>5.2f} {p['input_tpm']:>12,.0f} {p['cached_tpm']:>12,.0f} "
              f"{p['uncached_tpm']:>14,.0f} "
              f"{p['visible_tpm']:>12,.0f} {p['reasoning_tpm']:>11,.0f} "
              f"{p['cache_hit_rate']*100:>6.1f}% {p['ttft_p50_ms']:>8.1f}ms {p['ttft_p90_ms']:>8.1f}ms "
              f"{p['tpot_p50_ms']:>8.1f}ms {p['tpot_p90_ms']:>8.1f}ms")
    if num_gpus > 1:
        print(f"  (per-GPU: divide TPM by {num_gpus})")


def write_run_summary(run_dir, metrics, phases: List[dict], context: dict) -> None:
    """Write a machine-readable summary.json with overall + per-phase stats.

    `context` carries the run-level settings downstream tools want to key on
    (model, server_url, target QPS range, num_gpus, mode, etc.).
    """
    try:
        prompts = list(metrics.actual_prompt_lengths)
        gens    = list(metrics.actual_generation_lengths)
        reas    = list(metrics.actual_reasoning_lengths)
        ttfts   = list(metrics.actual_ttfts)
        tpots   = list(metrics.actual_tpots)
        start   = metrics.start_time
        end     = time.time()
        duration = end - start
        total_prompt  = sum(prompts)
        total_gen     = sum(gens)
        total_reason  = sum(reas)
        total_visible = max(0, total_gen - total_reason)
        total_cached  = sum(s[3] for s in metrics.prefill_samples)
        total_prefix  = sum(s[7] for s in metrics.prefill_samples)
        sent      = int(metrics.requests_sent)
        completed = int(metrics.requests_completed)
        errors    = int(metrics.errors)
        success_rate = (completed / sent) if sent else 0.0
        summary = {
            "context": context,
            "duration_s": duration,
            "requests_sent": sent,
            "requests_completed": completed,
            "errors": errors,
            "success_rate": success_rate,
            "actual_average_qps": (sent / duration) if duration > 0 else 0.0,
            "totals": {
                "input_tokens":     total_prompt,
                "cached_tokens":    total_cached,
                "uncached_tokens":  max(0, total_prompt - total_cached),
                "prefix_tokens":    total_prefix,
                "generation_tokens":  total_gen,
                "reasoning_tokens":   total_reason,
                "visible_tokens":     total_visible,
            },
            "prompt_length":  {
                "mean":  float(np.mean(prompts))  if prompts else 0.0,
                "p50":   float(percentiles(prompts,  [0.5])[0])  if prompts else 0.0,
                "p90":   float(percentiles(prompts,  [0.9])[0])  if prompts else 0.0,
                "p99":   float(percentiles(prompts,  [0.99])[0]) if prompts else 0.0,
            },
            "generation_length": {
                "mean":  float(np.mean(gens))  if gens else 0.0,
                "p50":   float(percentiles(gens,  [0.5])[0])  if gens else 0.0,
                "p90":   float(percentiles(gens,  [0.9])[0])  if gens else 0.0,
                "p99":   float(percentiles(gens,  [0.99])[0]) if gens else 0.0,
            },
            "reasoning_length": {
                "mean":  float(np.mean(reas))  if reas else 0.0,
                "p90":   float(percentiles(reas,  [0.9])[0])  if reas else 0.0,
            },
            "ttft_ms": {
                "mean": float(np.mean(ttfts)) * 1000 if ttfts else 0.0,
                "p50":  float(percentiles(ttfts, [0.5])[0])  * 1000 if ttfts else 0.0,
                "p90":  float(percentiles(ttfts, [0.9])[0])  * 1000 if ttfts else 0.0,
                "p99":  float(percentiles(ttfts, [0.99])[0]) * 1000 if ttfts else 0.0,
            },
            "tpot_ms": {
                "mean": float(np.mean(tpots)) * 1000 if tpots else 0.0,
                "p50":  float(percentiles(tpots, [0.5])[0])  * 1000 if tpots else 0.0,
                "p90":  float(percentiles(tpots, [0.9])[0])  * 1000 if tpots else 0.0,
                "p99":  float(percentiles(tpots, [0.99])[0]) * 1000 if tpots else 0.0,
            },
            "cache": {
                "ideal_hit_rate":  (total_prefix / total_prompt) if total_prompt else 0.0,
                "actual_hit_rate": (total_cached / total_prompt) if total_prompt else 0.0,
                "efficiency":      (total_cached / total_prefix) if total_prefix else 0.0,
                "eviction_rate":   (max(0, total_prefix - total_cached) / total_prefix) if total_prefix else 0.0,
                "server_reported_cached": total_cached > 0,
            },
            "phases": phases,
        }
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as exc:
        print(f"{Colors.YELLOW}Could not write summary.json: {exc}{Colors.END}")


class RunStorage:
    """Minimal file operations for benchmark mode"""

    def __init__(self, root_dir: Path = Path("benchmarks")):
        self.root_dir = root_dir

    def create_run_directory(self, name: str) -> Path:
        """Create timestamped directory for benchmark run"""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = self.root_dir / name / timestamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_metadata(self, path: Path, config: dict):
        """Write metadata.json"""
        metadata = {
            **config,
            "timestamp": path.name,
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


async def save_metrics_loop(
    metrics: BenchMetrics,
    sessions: List[ChatSession],
    metrics_file: Path,
    window_size: float,
    num_gpus: int,
    mtp_draft_tokens: int,
    running_flag: dict,
    session_stats: dict = None,  # {"created_by_rate": 0, "abandoned_by_rate": 0}
    ramp_duration_secs: float = 0.0,
    sustain_duration_secs: float = 0.0,
):
    """Save metrics to JSONL file every second for live dashboard.

    Calculates metrics for dashboard streaming:
    - throughput = total_tokens / window_size (fixed denominator)
    - cache_hit_rate = total_cached / total_tokens (persisted when no data)
    """
    last_timeline_index = 0  # Track which timeline events we've already sent
    last_distributions_index = 0  # Track which distribution data we have already sent (completion order)
    last_acceptance_index = 0  # Track which acceptance length data we have already sent
    last_planned_prompt_index = 0  # Track planned prompt lengths (send order - deterministic)
    last_planned_ideal_cache_index = 0  # Track planned ideal cache hit rates (send order - deterministic)
    last_inter_arrival_index = 0  # Track which inter-arrival times we have already sent
    last_cache_hit_rate = 0.0  # Persist cache hit rate when no data in window
    last_ideal_cache_hit_rate = 0.0  # Persist ideal cache hit rate when no data in window
    last_generation_tps = 0.0  # Persist generation TPS when no data in window

    while running_flag.get("running", True):
        now = time.time()
        cutoff = now - window_size
        start_time = metrics.start_time

        # Get samples in window (trailing window for live data)
        window_samples = [
            (t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize)
            for t, tokens, dur, cached, gen_tps, gen_tps_mtp, gen_time, psize in metrics.prefill_samples
            if t > cutoff
        ]

        # Calculate metrics with fixed window denominator
        if window_samples:
            total_tokens = sum(tokens for _, tokens, _, _, _, _, _, _ in window_samples)
            total_cached = sum(cached for _, _, _, cached, _, _, _, _ in window_samples)
            total_prefix = sum(psize for _, _, _, _, _, _, _, psize in window_samples)

            # Use fixed window_size as denominator (matches plot calculation)
            prefill_tps_window = total_tokens / window_size
            if total_tokens > 0:
                cache_hit_rate = total_cached / total_tokens
                ideal_cache_hit_rate = total_prefix / total_tokens
                last_cache_hit_rate = cache_hit_rate  # Update persisted value
                last_ideal_cache_hit_rate = ideal_cache_hit_rate  # Update persisted value
            else:
                cache_hit_rate = last_cache_hit_rate  # Use last known value
                ideal_cache_hit_rate = last_ideal_cache_hit_rate  # Use last known value

            # Filter generation TPS by minimum generation time
            valid_gen_tps_mtp = [gen_tps_mtp for _, _, _, _, _, gen_tps_mtp, gen_time, _ in window_samples if gen_time >= MIN_GENERATION_TIME]
            if valid_gen_tps_mtp:
                generation_tps = sum(valid_gen_tps_mtp) / len(valid_gen_tps_mtp)
                last_generation_tps = generation_tps  # Update persisted value
            else:
                generation_tps = last_generation_tps  # Use last known value
        else:
            prefill_tps_window = 0.0
            cache_hit_rate = last_cache_hit_rate  # Use last known value instead of 0
            ideal_cache_hit_rate = last_ideal_cache_hit_rate  # Use last known value instead of 0
            generation_tps = last_generation_tps  # Use last known value instead of 0

        in_flight = metrics.get_in_flight()

        # ChatSession statistics
        active_sessions = [s for s in sessions if not s.retired]
        retired_sessions = [s for s in sessions if s.retired]

        # Get new request timeline events since last update (relative timestamps)
        new_session_times = []  # Natural new sessions (yellow stars)
        forced_session_times = []  # Forced new sessions via keypress (red stars)
        existing_session_requests = []  # List of [time, session_id] for coloring by session
        current_timeline_len = len(metrics.request_timeline)
        for i in range(last_timeline_index, current_timeline_len):
            timestamp, is_new_session, session_id, is_forced = metrics.request_timeline[i]
            relative_time = timestamp - start_time
            if is_new_session:
                if is_forced:
                    forced_session_times.append(relative_time)
                else:
                    new_session_times.append(relative_time)
            else:
                existing_session_requests.append([relative_time, session_id])
        last_timeline_index = current_timeline_len

        # Get new planned prompt lengths since last update (SEND order - deterministic)
        current_planned_len = len(metrics.planned_prompt_lengths)
        new_planned_prompt_lengths = metrics.planned_prompt_lengths[last_planned_prompt_index:current_planned_len]
        last_planned_prompt_index = current_planned_len

        # Get new planned ideal cache hit rates since last update (SEND order - deterministic)
        current_planned_cache_len = len(metrics.planned_ideal_cache_hit_rates)
        new_planned_ideal_cache_hit_rates = metrics.planned_ideal_cache_hit_rates[last_planned_ideal_cache_index:current_planned_cache_len]
        last_planned_ideal_cache_index = current_planned_cache_len

        # Get new distribution data since last update (COMPLETION order - may vary)
        current_distributions_len = len(metrics.actual_prompt_lengths)
        new_prompt_lengths = metrics.actual_prompt_lengths[last_distributions_index:current_distributions_len]
        new_generation_lengths = metrics.actual_generation_lengths[last_distributions_index:current_distributions_len]
        new_cache_hit_rates = metrics.actual_cache_hit_rates[last_distributions_index:current_distributions_len]
        new_ideal_cache_hit_rates = metrics.ideal_cache_hit_rates[last_distributions_index:current_distributions_len]
        new_ttfts = metrics.actual_ttfts[last_distributions_index:current_distributions_len]
        last_distributions_index = current_distributions_len

        # Get new acceptance lengths since last update
        current_acceptance_len = len(metrics.actual_acceptance_lengths)
        new_acceptance_lengths = metrics.actual_acceptance_lengths[last_acceptance_index:current_acceptance_len]
        new_acceptance_rates = [acc_len / mtp_draft_tokens for acc_len in new_acceptance_lengths] if mtp_draft_tokens > 0 else []
        last_acceptance_index = current_acceptance_len

        # Get new inter-arrival times since last update
        current_inter_arrival_len = len(metrics.inter_arrival_times)
        new_inter_arrival_times = metrics.inter_arrival_times[last_inter_arrival_index:current_inter_arrival_len]
        last_inter_arrival_index = current_inter_arrival_len

        elapsed = now - start_time
        if elapsed < ramp_duration_secs:
            phase = "ramp"
        elif elapsed < ramp_duration_secs + sustain_duration_secs:
            phase = "sustain"
        else:
            phase = "drain"

        record = {
            "timestamp": now,
            "elapsed_seconds": elapsed,
            "phase": phase,
            "prefill_tps": prefill_tps_window,
            "prefill_tps_window": prefill_tps_window,
            "prefill_tpm_per_gpu": (prefill_tps_window * 60) / num_gpus if num_gpus > 0 else 0,
            "generation_tps": generation_tps,
            "cache_hit_rate": cache_hit_rate,
            "ideal_cache_hit_rate": ideal_cache_hit_rate,
            "requests_completed": metrics.requests_completed,
            "requests_sent": metrics.requests_sent,
            "errors": metrics.errors,
            "in_flight": in_flight,
            "num_sessions_active": len(active_sessions),
            "num_sessions_retired": len(retired_sessions),
            "num_sessions_abandoned": len([s for s in sessions if s.abandoned]),
            "num_sessions_total": len(sessions),
            "sessions_created_by_rate": session_stats.get("created_by_rate", 0) if session_stats else 0,
            "sessions_abandoned_by_rate": session_stats.get("abandoned_by_rate", 0) if session_stats else 0,
            "gpus": num_gpus,
            "window_size": window_size,
            "new_session_times": new_session_times,
            "forced_session_times": forced_session_times,
            "existing_session_requests": existing_session_requests,  # [[time, session_id], ...]
            "new_planned_prompt_lengths": new_planned_prompt_lengths,
            "new_planned_ideal_cache_hit_rates": new_planned_ideal_cache_hit_rates,
            "new_prompt_lengths": new_prompt_lengths,
            "new_generation_lengths": new_generation_lengths,
            "new_cache_hit_rates": new_cache_hit_rates,
            "new_ideal_cache_hit_rates": new_ideal_cache_hit_rates,
            "new_ttfts": new_ttfts,
            "new_acceptance_lengths": new_acceptance_lengths,
            "new_acceptance_rates": new_acceptance_rates,
            "new_inter_arrival_times": new_inter_arrival_times,
        }

        with open(metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        await asyncio.sleep(1)


async def run_replay(
    server_url: str,
    model: str,
    system_prompt_len: int,
    new_tokens_mean: int,
    new_tokens_median: int,
    initial_qps: float,
    max_qps: float,
    ramp_duration_secs: float,
    sustain_duration_secs: float,
    tokenizer,
    api_key: str = None,
    window_size: float = 15.0,
    generation_length_mean: int = 1,
    generation_length_median: int = 1,
    acc_len: float = 3.0,
    mtp_overhead_factor: float = 1.0,
    num_gpus: int = 1,
    max_prompt_tokens: int = 200000,
    use_poisson: bool = False,
    poisson_shape: float = 1.0,
    new_session_rate: float = 0.00,
    num_initial_sessions: int = 1,
    random_seed: int = None,
    initial_prefix_mean: int = 0,
    initial_prefix_median: int = 0,
    max_inflight: int = None,
    session_decay_lambda: float = 0.02,
    dashboard_mode: bool = False,
    benchmark_name: str = None,
    data_dir: Path = None,
    ignore_eos: bool = True,
    mtp_draft_tokens: int = 1
):
    """Run the peak throughput benchmark V2 with growing session prefixes"""

    metrics = BenchMetrics()

    # Start keyboard listener for manual session triggers
    keyboard_listener.start()
    metrics_file = None
    run_dir = None

    # Setup benchmark mode output directory
    if dashboard_mode and benchmark_name and data_dir:
        storage = RunStorage(root_dir=data_dir)
        run_dir = storage.create_run_directory(benchmark_name)
        metrics_file = run_dir / "metrics.jsonl"

        # Save metadata
        config_metadata = {
            "server_url": server_url,
            "model": model,
            "system_prompt_len": system_prompt_len,
            "new_tokens_mean": new_tokens_mean,
            "new_tokens_median": new_tokens_median,
            "generation_length_mean": generation_length_mean,
            "generation_length_median": generation_length_median,
            "initial_qps": initial_qps,
            "max_qps": max_qps,
            "ramp_duration_secs": ramp_duration_secs,
            "sustain_duration_secs": sustain_duration_secs,
            "window_size": window_size,
            "num_gpus": num_gpus,
            "max_prompt_tokens": max_prompt_tokens,
            "use_poisson": use_poisson,
            "poisson_shape": poisson_shape,
            "new_session_rate": new_session_rate,
            "num_initial_sessions": num_initial_sessions,
            "random_seed": random_seed,
            "initial_prefix_mean": initial_prefix_mean,
            "initial_prefix_median": initial_prefix_median,
            "max_inflight": max_inflight,
            "mtp_overhead_factor": mtp_overhead_factor,
            "acc_len": acc_len,
        }
        storage.save_metadata(run_dir, config_metadata)
        print(f"{Colors.GREEN}Benchmark mode enabled. Output: {run_dir}{Colors.END}")

    # Generate synthetic system prompt
    print(f"{Colors.CYAN}Generating synthetic system prompt ({system_prompt_len:,} tokens)...{Colors.END}")
    system_prompt = make_filler(system_prompt_len, tokenizer)
    system_prompt_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    print(f"{Colors.GREEN}System prompt generated: {system_prompt_tokens:,} tokens{Colors.END}")

    # Initialize session list
    sessions: List[ChatSession] = []
    
    # Create initial sessions using spawn_session for pre-tokenized base text
    if num_initial_sessions > 0:
        print(f"{Colors.CYAN}Creating {num_initial_sessions} initial session(s)...{Colors.END}")
        # Use the lognormal initial_prefix distribution so the prompt mix at
        # t=0 already reflects sessions at varied life stages.
        if initial_prefix_mean > 0 and initial_prefix_median > 0:
            _mu = math.log(initial_prefix_median)
            _sigma = math.sqrt(2 * math.log(initial_prefix_mean / initial_prefix_median)) \
                if initial_prefix_mean > initial_prefix_median else 0.1
        else:
            _mu = _sigma = 0.0
        for i in range(num_initial_sessions):
            # Use unique seed for each initial session
            seed = random_seed + i * 1000 if random_seed else i * 1000
            if _sigma > 0:
                _ipt = max(0, int(np.random.lognormal(mean=_mu, sigma=_sigma)))
                _ipt = min(_ipt, max(0, max_prompt_tokens - system_prompt_tokens))
            else:
                _ipt = 0
            spawn_session(
                sessions=sessions,
                system_prompt=system_prompt,
                system_prompt_tokens=system_prompt_tokens,
                request_seq=-(num_initial_sessions - i),  # Negative seq for staggered ages
                tokenizer=tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                seed=seed,
                initial_prefix_tokens=_ipt,
                max_sessions=100
            )
        print(f"{Colors.GREEN}Done. Starting with {len(sessions)} session(s).{Colors.END}")

    print(f"{Colors.BOLD}LLM Throughput Simulator (Growing ChatSession Prefixes){Colors.END}")
    print(f"{Colors.DIM}{'-'*80}{Colors.END}")
    print(f"Server: {server_url}")
    print(f"Model: {model}")
    print(f"System prompt: {system_prompt_tokens:,} tokens")
    print(f"New tokens per request (mean / median): {new_tokens_mean:,} / {new_tokens_median:,}")
    print(f"Max prefix size (retirement): {max_prompt_tokens:,} tokens")
    print(f"Generation length (mean / median): {generation_length_mean} / {generation_length_median}")
    if mtp_overhead_factor != 1.0:
        print(f"MTP overhead factor: {mtp_overhead_factor}")
    print(f"GPUs: {num_gpus}")
    print(f"Ramp: {initial_qps:.2f} -> {max_qps:.2f} QPS over {ramp_duration_secs:.0f}s")
    print(f"Sustain: {max_qps:.2f} QPS for {sustain_duration_secs:.0f}s")
    print(f"Initial sessions: {num_initial_sessions}")
    print(f"New session rate: {new_session_rate*100:.1f}%")
    if max_inflight:
        print(f"Max in-flight (backpressure): {max_inflight}")
    print(f"Window size: {window_size:.0f}s")
    print(f"{Colors.DIM}{'='*80}{Colors.END}")
    print()

    # Start the request ramp
    # Use random_seed as base_seed for deterministic text generation (default to 0 if not set)
    base_seed = random_seed if random_seed is not None else 0
    
    ramp_task = asyncio.create_task(
        ramp_requests(
            server_url, model, sessions, system_prompt, system_prompt_tokens,
            new_tokens_mean, new_tokens_median, new_session_rate,
            initial_qps, max_qps, ramp_duration_secs,
            sustain_duration_secs, metrics, tokenizer, base_seed, generation_length_mean, generation_length_median,
            acc_len, mtp_overhead_factor, api_key, max_prompt_tokens, use_poisson, poisson_shape,
            initial_prefix_mean, initial_prefix_median, max_inflight, session_decay_lambda,
            ignore_eos
        )
    )

    # Display live metrics
    total_duration = ramp_duration_secs + sustain_duration_secs

    # Run display and ramp concurrently
    display_task = asyncio.create_task(
        display_metrics_live(metrics, total_duration, window_size, 0.0)  # No target, emergent
    )

    # Start metrics streaming if benchmark mode is enabled
    metrics_task = None
    running_flag = {"running": True}
    if metrics_file:
        metrics_task = asyncio.create_task(
            save_metrics_loop(metrics, sessions, metrics_file, window_size, num_gpus, mtp_draft_tokens, running_flag,
                              ramp_duration_secs=ramp_duration_secs, sustain_duration_secs=sustain_duration_secs)
        )

    try:
        await asyncio.gather(ramp_task, display_task)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
        ramp_task.cancel()
        display_task.cancel()
        try:
            await asyncio.gather(ramp_task, display_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass
    finally:
        # Stop metrics streaming
        running_flag["running"] = False
        keyboard_listener.stop()
        if metrics_task:
            metrics_task.cancel()
            try:
                await metrics_task
            except asyncio.CancelledError:
                pass

    # Final summary
    print(f"\n{Colors.BOLD}Final Results:{Colors.END}")
    print(f"{Colors.DIM}{'-'*80}{Colors.END}")

    # Calculate actual benchmark duration and average QPS
    actual_duration = time.time() - metrics.start_time
    actual_average_qps = metrics.requests_sent / actual_duration if actual_duration > 0 else 0

    print(f"Total requests sent: {metrics.requests_sent:,}")
    print(f"Completed: {metrics.requests_completed:,}")
    print(f"Errors: {metrics.errors:,}")
    print(f"Success rate: {100 * metrics.requests_completed / max(metrics.requests_sent, 1):.1f}%")
    print(f"Actual benchmark duration: {actual_duration:.1f}s")
    print(f"Actual average QPS: {actual_average_qps:.2f} (target: {initial_qps:.2f} -> {max_qps:.2f})")

    # Initialize average metrics with defaults
    avg_throughput_tps = 0.0
    avg_throughput_tpm = 0.0
    avg_generation_tps = 0.0
    avg_cache_hit_rate = 0.0
    avg_ideal_cache_hit_rate = 0.0
    eviction_rate = 0.0

    # Calculate length distributions
    if metrics.actual_prompt_lengths:
        prompt_p50, prompt_p90, prompt_p99 = percentiles(metrics.actual_prompt_lengths, [0.5, 0.9, 0.99])
        prompt_mean = np.mean(metrics.actual_prompt_lengths)
        prompt_std = np.std(metrics.actual_prompt_lengths)

        print(f"\n{Colors.BOLD}Actual Prompt Length Distribution:{Colors.END}")
        print(f"  Mean: {prompt_mean:.0f} tokens")
        print(f"  Std Dev: {prompt_std:.0f} tokens")
        print(f"  p50: {prompt_p50:.0f} tokens")
        print(f"  p90: {prompt_p90:.0f} tokens")
        print(f"  p99: {prompt_p99:.0f} tokens")

    if metrics.actual_generation_lengths:
        gen_p50, gen_p90, gen_p99 = percentiles(metrics.actual_generation_lengths, [0.5, 0.9, 0.99])
        gen_mean = np.mean(metrics.actual_generation_lengths)
        gen_std = np.std(metrics.actual_generation_lengths)

        print(f"\n{Colors.BOLD}Actual Generation Length Distribution:{Colors.END}")
        print(f"  Mean: {gen_mean:.1f} tokens (target: {generation_length_mean})")
        print(f"  Median (p50): {gen_p50:.0f} tokens (target: {generation_length_median})")
        print(f"  Std Dev: {gen_std:.1f} tokens")
        print(f"  p90: {gen_p90:.0f} tokens")
        print(f"  p99: {gen_p99:.0f} tokens")

    # Calculate percentiles
    if metrics.prefill_samples:
        ttfts = [dur for _, _, dur, _, _, _, _, _ in metrics.prefill_samples]
        ttfts.sort()
        n = len(ttfts)
        p50_ttft = ttfts[n//2] if n > 0 else 0
        p90_ttft = ttfts[int(n*0.9)] if n > 0 else 0
        p99_ttft = ttfts[int(n*0.99)] if n > 0 else 0

        print(f"\n{Colors.BOLD}TTFT (Time to First Token):{Colors.END}")
        print(f"  p50: {p50_ttft*1000:.1f}ms")
        print(f"  p90: {p90_ttft*1000:.1f}ms")
        print(f"  p99: {p99_ttft*1000:.1f}ms")

        # TPOT (Time Per Output Token) — excludes the first token, which is captured by TTFT.
        # We work off `actual_tpots` (already filtered to gen_len>1 and gen_time>=MIN_GENERATION_TIME).
        if metrics.actual_tpots:
            tpot_p50, tpot_p90, tpot_p99 = percentiles(metrics.actual_tpots, [0.5, 0.9, 0.99])
            tpot_mean = float(np.mean(metrics.actual_tpots))
            print(f"\n{Colors.BOLD}TPOT (Time Per Output Token, excl. first):{Colors.END}")
            print(f"  Samples: {len(metrics.actual_tpots)} (filtered: gen_len>1 & gen_time>={MIN_GENERATION_TIME*1000:.0f}ms)")
            print(f"  Mean: {tpot_mean*1000:.1f}ms ({(1.0/tpot_mean):.1f} tok/s/req)" if tpot_mean > 0 else "  Mean: n/a")
            print(f"  p50: {tpot_p50*1000:.1f}ms")
            print(f"  p90: {tpot_p90*1000:.1f}ms")
            print(f"  p99: {tpot_p99*1000:.1f}ms")

        # Calculate average throughput over entire benchmark
        total_prefill_tokens = sum(tokens for _, tokens, _, _, _, _, _, _ in metrics.prefill_samples)
        avg_throughput_tps = total_prefill_tokens / actual_duration if actual_duration > 0 else 0
        avg_throughput_tpm = avg_throughput_tps * 60

        # Calculate average generation TPS (MTP compensated)
        # Filter out samples with short generation_time (likely burst artifacts)
        generation_tps_values = [gen_tps_mtp for _, _, _, _, _, gen_tps_mtp, gen_time, _ in metrics.prefill_samples if gen_tps_mtp > 0 and gen_time >= MIN_GENERATION_TIME]
        filtered_count = sum(1 for _, _, _, _, _, gen_tps_mtp, gen_time, _ in metrics.prefill_samples if gen_tps_mtp > 0 and gen_time < MIN_GENERATION_TIME)
        avg_generation_tps = sum(generation_tps_values) / len(generation_tps_values) if generation_tps_values else 0.0

        # Peak throughput (scan all historical samples)
        best_throughput = 0
        for window in [1, 5, 10]:
            throughput, _ = metrics.get_peak_throughput(window)
            if throughput > best_throughput:
                best_throughput = throughput

        print(f"\n{Colors.BOLD}{Colors.GREEN}Peak Prefill Throughput:{Colors.END}")
        print(f"  Total: {best_throughput:,.0f} tokens/sec ({best_throughput*60:,.0f} tokens/min)")
        if num_gpus > 1:
            per_gpu_tps = best_throughput / num_gpus
            per_gpu_tpm = per_gpu_tps * 60
            print(f"  Per GPU: {per_gpu_tps:,.0f} tokens/sec ({per_gpu_tpm:,.0f} tokens/min)")

        print(f"\n{Colors.BOLD}Average Throughput:{Colors.END}")
        if num_gpus > 1:
            avg_tpm_per_gpu = avg_throughput_tpm / num_gpus
            print(f"  Context: {avg_tpm_per_gpu:,.0f} tokens/min/GPU ({avg_throughput_tps / num_gpus:,.0f} tokens/sec/GPU)")
        else:
            print(f"  Context: {avg_throughput_tpm:,.0f} tokens/min ({avg_throughput_tps:,.0f} tokens/sec)")

        if avg_generation_tps > 0:
            print(f"  Generation: {avg_generation_tps:.1f} tokens/sec (MTP compensated)")
            if filtered_count > 0:
                print(f"    (filtered {filtered_count} samples with generation_time < {MIN_GENERATION_TIME*1000:.0f}ms)")

        # Calculate average cache hit rate and eviction stats
        total_tokens = sum(tokens for _, tokens, _, _, _, _, _, _ in metrics.prefill_samples)
        total_cached = sum(cached for _, _, _, cached, _, _, _, _ in metrics.prefill_samples)
        total_prefix = sum(psize for _, _, _, _, _, _, _, psize in metrics.prefill_samples)
        avg_cache_hit_rate = total_cached / total_tokens if total_tokens > 0 else 0.0
        avg_ideal_cache_hit_rate = total_prefix / total_tokens if total_tokens > 0 else 0.0
        
        # Calculate eviction stats
        total_evicted = max(0, total_prefix - total_cached)
        eviction_rate = total_evicted / total_prefix if total_prefix > 0 else 0.0

        print(f"\n{Colors.BOLD}Cache Statistics:{Colors.END}")
        print(f"  Ideal cache hit rate: {avg_ideal_cache_hit_rate*100:.1f}% (assuming no eviction)")
        print(f"  Actual cache hit rate: {avg_cache_hit_rate*100:.1f}%")
        print(f"  Cache efficiency: {(avg_cache_hit_rate/avg_ideal_cache_hit_rate*100) if avg_ideal_cache_hit_rate > 0 else 0:.1f}% (actual/ideal)")
        print(f"  Eviction rate: {eviction_rate*100:.1f}% of expected cache was evicted")
        print(f"  Total tokens: {total_tokens:,} (prefix: {total_prefix:,}, cached: {total_cached:,}, evicted: {total_evicted:,})")
        if total_cached == 0 and avg_ideal_cache_hit_rate > 0.1:
            print(f"  {Colors.YELLOW}NOTE: server returned 0 cached tokens — its OpenAI-compatible API is not{Colors.END}")
            print(f"  {Colors.YELLOW}      reporting cache hits in usage.cache_read_input_tokens (or prompt_tokens_details.cached_tokens).{Colors.END}")
            print(f"  {Colors.YELLOW}      For sglang add --enable-cache-report to launch_server; for others enable{Colors.END}")
            print(f"  {Colors.YELLOW}      usage-level cache reporting. Meanwhile 'Actual cache hit rate' above is unreliable.{Colors.END}")

    # Ramp / sustain / drain throughput table
    phases = compute_phase_breakdown(metrics, metrics.start_time, ramp_duration_secs, sustain_duration_secs)
    print_phase_breakdown(phases, num_gpus=num_gpus)

    # Machine-readable summary for sweep aggregation / CI
    if 'run_dir' in locals() and run_dir is not None:
        write_run_summary(run_dir, metrics, phases, context={
            "mode": "traffic-replay",
            "server_url": server_url,
            "model": model,
            "num_gpus": num_gpus,
            "ramp_duration_secs": ramp_duration_secs,
            "sustain_duration_secs": sustain_duration_secs,
            "initial_qps": initial_qps,
            "max_qps": max_qps,
            "max_inflight": max_inflight,
        })

    # Calculate actual new session ratio from timeline
    if metrics.request_timeline:
        new_session_count = sum(1 for item in metrics.request_timeline if item[1])  # item[1] is is_new
        actual_new_session_ratio = new_session_count / len(metrics.request_timeline)
    else:
        actual_new_session_ratio = 0.0
        new_session_count = 0

    # Print session statistics
    active_sessions = [s for s in sessions if not s.retired]
    retired_sessions = [s for s in sessions if s.retired]
    
    print(f"\n{Colors.BOLD}ChatSession Statistics:{Colors.END}")
    print(f"  Total sessions: {len(sessions)}")
    print(f"  Active: {len(active_sessions)}, Retired: {len(retired_sessions)}")
    print(f"  Target new session rate: {new_session_rate*100:.1f}%")
    print(f"  Actual new session rate: {actual_new_session_ratio*100:.1f}%")
    if sessions:
        prefix_sizes = [s.prefix_tokens for s in sessions]
        print(f"  Final prefix sizes: min={min(prefix_sizes):,}, max={max(prefix_sizes):,}, mean={sum(prefix_sizes)//len(prefix_sizes):,}")


async def run_session_walk(
    server_url: str,
    model: str,
    system_prompt_len: int,
    new_tokens_mean: int,
    new_tokens_median: int,
    initial_qps: float,
    max_qps: float,
    ramp_duration_secs: float,
    sustain_duration_secs: float,
    tokenizer,
    api_key: str = None,
    window_size: float = 15.0,
    generation_length_mean: int = 1,
    generation_length_median: int = 1,
    acc_len: float = 3.0,
    mtp_overhead_factor: float = 1.0,
    num_gpus: int = 1,
    max_prompt_tokens: int = 200000,
    num_initial_sessions: int = 1,
    random_seed: int = None,
    initial_prefix_mean: int = 0,
    initial_prefix_median: int = 0,
    max_inflight: int = None,
    dashboard_mode: bool = False,
    benchmark_name: str = None,
    data_dir: Path = None,
    ignore_eos: bool = True,
    mtp_draft_tokens: int = 1,
    # Realistic mode specific parameters
    think_time_mean: float = 10.0,
    think_time_shape: float = 1.0,  # Gamma shape parameter (1.0 = exponential)
    session_lifetime_mean: float = 600.0,
    session_lifetime_median: float = 400.0,
    max_sessions: int = 100,
    new_session_rate: float = 0.0,  # Probability per second to create a new session
    session_abandon_rate: float = 0.0,  # Probability per request to abandon session
):
    """Run benchmark in realistic mode with response-chained sessions."""

    metrics = BenchMetrics()
    qps_monitor = RateTracker(window_secs=5.0)
    response_time_tracker = LatencyTracker(initial_estimate=think_time_mean)

    # Start keyboard listener for manual session creation (press 'n')
    keyboard_listener.start()

    metrics_file = None
    run_dir = None

    # Setup benchmark mode output directory
    if dashboard_mode and benchmark_name and data_dir:
        storage = RunStorage(root_dir=data_dir)
        run_dir = storage.create_run_directory(benchmark_name)
        metrics_file = run_dir / "metrics.jsonl"

        # Save metadata
        config_metadata = {
            "mode": "realistic",
            "server_url": server_url,
            "model": model,
            "system_prompt_len": system_prompt_len,
            "new_tokens_mean": new_tokens_mean,
            "new_tokens_median": new_tokens_median,
            "generation_length_mean": generation_length_mean,
            "generation_length_median": generation_length_median,
            "duration_secs": sustain_duration_secs,
            "window_size": window_size,
            "num_gpus": num_gpus,
            "max_prompt_tokens": max_prompt_tokens,
            "num_initial_sessions": num_initial_sessions,
            "initial_prefix_mean": initial_prefix_mean,
            "initial_prefix_median": initial_prefix_median,
            "max_inflight": max_inflight,
            "mtp_overhead_factor": mtp_overhead_factor,
            "acc_len": acc_len,
            "think_time_mean": think_time_mean,
            "think_time_shape": think_time_shape,
            "session_lifetime_mean": session_lifetime_mean,
            "session_lifetime_median": session_lifetime_median,
            "max_sessions": max_sessions,
            "new_session_rate": new_session_rate,
            "session_abandon_rate": session_abandon_rate,
        }
        storage.save_metadata(run_dir, config_metadata)
        print(f"{Colors.GREEN}Benchmark mode enabled. Output: {run_dir}{Colors.END}")

    # Generate synthetic system prompt
    print(f"{Colors.CYAN}Generating synthetic system prompt ({system_prompt_len:,} tokens)...{Colors.END}")
    system_prompt = make_filler(system_prompt_len, tokenizer)
    system_prompt_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    print(f"{Colors.GREEN}System prompt generated: {system_prompt_tokens:,} tokens{Colors.END}")

    # ChatSession management
    sessions: List[ChatSession] = []
    session_tasks: List[asyncio.Task] = []
    request_counter = [0]  # Mutable counter for request IDs
    session_stats = {"created_by_rate": 0, "abandoned_by_rate": 0}  # Shared counters for dashboard

    # Use random_seed as base_seed for deterministic text generation
    base_seed = random_seed if random_seed is not None else 0

    def create_new_session(forced: bool = False) -> ChatSession:
        """Create a new session with pre-tokenized base text and sampled lifetime."""
        # Sample initial prefix if configured
        initial_prefix_tokens_count = 0
        if initial_prefix_mean > 0 and initial_prefix_median > 0:
            initial_prefix_tokens_count = draw_lognormal(initial_prefix_mean, initial_prefix_median)
            # Clamp to enforce system_prompt_tokens + initial_prefix_tokens <= max_prompt_tokens
            max_initial_prefix = max(0, max_prompt_tokens - system_prompt_tokens)
            initial_prefix_tokens_count = min(initial_prefix_tokens_count, max_initial_prefix)

        # Sample session lifetime
        lifetime = draw_session_lifetime(session_lifetime_mean, session_lifetime_median)

        # Use deterministic seed based on session count
        seed = base_seed + len(sessions) * 1000

        # Create session using shared spawn_session function
        return spawn_session(
            sessions=sessions,
            system_prompt=system_prompt,
            system_prompt_tokens=system_prompt_tokens,
            request_seq=request_counter[0],
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            seed=seed,
            initial_prefix_tokens=initial_prefix_tokens_count,
            max_sessions=max_sessions,
            lifetime_limit=lifetime,
            forced=forced,
        )

    async def send_realistic_request(
        http_session: aiohttp.ClientSession,
        session: ChatSession,
    ) -> Tuple[bool, float, str]:
        """Send a single request for a session using pre-tokenized base text."""
        request_id = request_counter[0]
        request_counter[0] += 1

        # Sample new tokens and generation length
        new_tokens = draw_lognormal(new_tokens_mean, new_tokens_median)
        generation_length = draw_lognormal(generation_length_mean, generation_length_median)

        # Get current prefix tokens before growing
        current_prefix_tokens = session.prefix_tokens

        # Check if adding new_tokens would exceed max
        available_space = max_prompt_tokens - current_prefix_tokens
        if new_tokens > available_space:
            new_tokens = max(1, available_space)

        # Calculate total prompt tokens
        total_prompt_tokens = current_prefix_tokens + new_tokens

        # Record metrics at send time
        metrics.requests_sent += 1
        metrics.planned_prompt_lengths.append(total_prompt_tokens)
        planned_ideal_cache_rate = current_prefix_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
        metrics.planned_ideal_cache_hit_rates.append(planned_ideal_cache_rate)

        # Record timeline
        request_timestamp = time.time()
        is_new_session = session.request_count == 0
        is_forced = session.forced and is_new_session  # Only mark forced on first request
        metrics.request_timeline.append((request_timestamp, is_new_session, session.id, is_forced))

        # Grow the session by new_tokens (extends the slice into pre-tokenized base)
        session.grow(new_tokens, request_id)

        # Get the full prompt by decoding the current token slice
        # This is exact - no drift, no truncation needed
        full_content = session.get_prompt()

        messages = [{"role": "user", "content": full_content}]
        payload = {
            # "request_id": str(uuid.uuid4()),
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": generation_length,
            "temperature": 0.0,
            "user": session.id,
            "ignore_eos": ignore_eos,
        }

        url = f"{server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        start_time = time.time()
        ttft = None
        cached_tokens = 0
        reasoning_tokens = 0
        actual_prompt_tokens = 0
        full_response = ""
        chunk_token_counts = []

        try:
            async with http_session.post(url, json=payload, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=240)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"{Colors.RED}Request {request_id} failed: HTTP {resp.status} - {error_text[:200]}{Colors.END}")
                    metrics.errors += 1
                    return False, 0.0, ""

                async for line in resp.content:
                    if line:
                        line_str = line.decode("utf-8").strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if not data or not isinstance(data, dict):
                                    continue

                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    resp_content = delta.get("content", "")
                                    resp_reason = delta.get("reasoning_content") or delta.get("reasoning")
                                    if (resp_content or resp_reason) and ttft is None:
                                        ttft = time.time() - start_time
                                    if resp_reason:
                                        full_response += resp_reason
                                    if resp_content:
                                        full_response += resp_content
                                        chunk_tokens = len(tokenizer.encode(resp_content, add_special_tokens=False)) - 1
                                        if chunk_tokens > 0:
                                            chunk_token_counts.append(chunk_tokens)

                                if "usage" in data and data["usage"]:
                                    usage = data["usage"]
                                    if isinstance(usage, dict):
                                        if "prompt_tokens" in usage:
                                            actual_prompt_tokens = usage.get("prompt_tokens", 0)
                                        if "cache_read_input_tokens" in usage:
                                            cached_tokens = usage.get("cache_read_input_tokens") or 0
                                        elif "prompt_tokens_details" in usage:
                                            details = usage["prompt_tokens_details"]
                                            if isinstance(details, dict):
                                                cached_tokens = details.get("cached_tokens") or 0
                                        if "reasoning_tokens" in usage:
                                            reasoning_tokens = usage.get("reasoning_tokens") or 0
                            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                                # See parallel comment in traffic-replay path: narrow
                                # the catch so real exceptions surface instead of being
                                # masked by a blanket `except Exception`.
                                pass

                completion_tokens = len(tokenizer.encode(full_response, add_special_tokens=False)) if full_response else 0

                if ttft is None:
                    ttft = time.time() - start_time

                end_time = time.time()
                total_time = end_time - start_time
                generation_time = total_time - ttft if ttft and total_time > ttft else 0.0
                generation_tps = completion_tokens / generation_time if generation_time > 0 and completion_tokens > 0 else 0.0
                generation_tps_mtp = (completion_tokens * acc_len) / (generation_time * mtp_overhead_factor) if generation_time > 0 and completion_tokens > 0 else 0.0

                tokens_to_record = actual_prompt_tokens if actual_prompt_tokens > 0 else total_prompt_tokens
                metrics.add_prefill(tokens_to_record, ttft, cached_tokens, generation_tps, generation_tps_mtp,
                                    completion_tokens, generation_time, current_prefix_tokens,
                                    reasoning_tokens=reasoning_tokens)

                if chunk_token_counts:
                    avg_acceptance_length = sum(chunk_token_counts) / len(chunk_token_counts)
                    metrics.add_acceptance_length(avg_acceptance_length)
                else:
                    # Request completed but no content returned
                    metrics.requests_completed += 1

                # Update session with response content (realistic mode includes response).
                # Note: the user-turn tokens were already added at session.grow() above,
                # so here we only grow by the *response* tokens. Prior code used a stale
                # `new_content` name (NameError) and a back-compat shim that double-counted
                # the user tokens.
                response_tokens = len(tokenizer.encode(full_response, add_special_tokens=False)) if full_response else 0
                if response_tokens:
                    session.grow(response_tokens, request_id)

                # Check retirement conditions
                if session.should_retire(max_prompt_tokens) or session.should_retire_lifetime():
                    session.retired = True

                # Record response time and QPS
                response_time_tracker.record(total_time)
                await qps_monitor.record_request()

                return True, total_time, full_response

        except asyncio.TimeoutError:
            metrics.errors += 1
            print(f"{Colors.RED}Request {request_id} timed out{Colors.END}")
            return False, 0.0, ""
        except Exception as e:
            metrics.errors += 1
            print(f"{Colors.RED}Request {request_id} error: {e}{Colors.END}")
            return False, 0.0, ""

    async def session_loop(http_session: aiohttp.ClientSession, session: ChatSession):
        """Run the request loop for a single session."""
        consecutive_failures = 0
        last_request_end_time: Optional[float] = None

        while not session.retired and not session.abandoned:
            # Record actual inter-arrival time (time since last request ended)
            if last_request_end_time is not None:
                actual_inter_arrival = time.time() - last_request_end_time
                metrics.inter_arrival_times.append(actual_inter_arrival)

            session.in_flight = True
            success, response_time, _ = await send_realistic_request(http_session, session)
            session.in_flight = False
            last_request_end_time = time.time()

            if success:
                consecutive_failures = 0
                if session.retired:
                    break
                # Check for random abandonment (user closes session)
                if session_abandon_rate > 0 and random.random() < session_abandon_rate:
                    session.abandoned = True
                    session_stats["abandoned_by_rate"] += 1
                    break
                # Think time before next request (capped so inter-arrival doesn't exceed limit)
                think_time = draw_think_time(think_time_mean, think_time_shape)
                max_think = max(MIN_THINK_TIME, MAX_INTER_ARRIVAL_TIME - response_time)
                think_time = min(think_time, max_think)
                await asyncio.sleep(think_time)
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_RETRIES:
                    session.abandoned = True
                    print(f"{Colors.YELLOW}ChatSession {session.id} abandoned after {MAX_RETRIES} failures{Colors.END}")
                    break
                # Exponential backoff
                await asyncio.sleep(2 ** consecutive_failures)

    def count_active_sessions() -> int:
        """Count sessions that are available for requests."""
        return sum(1 for s in sessions if s.is_available() or s.in_flight)

    print(f"\n{Colors.BOLD}LLM Throughput Simulator (Realistic Mode){Colors.END}")
    print(f"{Colors.DIM}{'-'*80}{Colors.END}")
    print(f"Server: {server_url}")
    print(f"Model: {model}")
    print(f"Initial sessions: {num_initial_sessions}, Max sessions: {max_sessions}")
    print(f"New session rate: {new_session_rate*100:.1f}%/sec, Abandon rate: {session_abandon_rate*100:.1f}%/req")
    print(f"Think time: mean={think_time_mean}s, shape={think_time_shape}")
    print(f"ChatSession lifetime: mean={session_lifetime_mean}s, median={session_lifetime_median}s")
    print(f"Duration: {sustain_duration_secs}s")
    print(f"{Colors.DIM}{'='*80}{Colors.END}")
    print()

    # Start metrics streaming BEFORE session creation so we capture everything
    metrics_task = None
    running_flag = {"running": True}
    if metrics_file:
        metrics_task = asyncio.create_task(
            save_metrics_loop(metrics, sessions, metrics_file, window_size, num_gpus, mtp_draft_tokens, running_flag, session_stats,
                              ramp_duration_secs=ramp_duration_secs, sustain_duration_secs=sustain_duration_secs)
        )

    # Start initial sessions with staggered timing (1 second apart to avoid thundering herd)
    print(f"{Colors.CYAN}Starting {num_initial_sessions} initial sessions (staggered)...{Colors.END}")
    # stagger_interval = max(0.1, think_time_mean / max(num_initial_sessions, 1))
    stagger_interval = 1.0

    connector = aiohttp.TCPConnector(limit=1000)
    async with aiohttp.ClientSession(connector=connector) as http_session:
        # Create and start initial sessions
        for i in range(num_initial_sessions):
            session = create_new_session()
            task = asyncio.create_task(session_loop(http_session, session))
            session_tasks.append(task)
            print(f"{Colors.GREEN}Started session {session.id} ({i+1}/{num_initial_sessions}){Colors.END}")
            if i < num_initial_sessions - 1:
                await asyncio.sleep(stagger_interval)

        print(f"{Colors.GREEN}All {num_initial_sessions} sessions started{Colors.END}\n")

        # Main control loop
        start_time = time.time()
        total_duration = sustain_duration_secs

        try:
            while time.time() - start_time < total_duration:
                elapsed = time.time() - start_time
                current_qps = await qps_monitor.get_qps()
                active_sessions = count_active_sessions()

                # Check for manual session creation (press 'n')
                pending_new_sessions = keyboard_listener.get_pending_count()
                for _ in range(pending_new_sessions):
                    if len(sessions) < max_sessions:
                        session = create_new_session(forced=True)
                        task = asyncio.create_task(session_loop(http_session, session))
                        session_tasks.append(task)
                        print(f"\n{Colors.YELLOW}[Manual] Created session {session.id}{Colors.END}")

                # Create new sessions based on rate (probability per second)
                if new_session_rate > 0 and len(sessions) < max_sessions:
                    if random.random() < new_session_rate:
                        session = create_new_session()
                        task = asyncio.create_task(session_loop(http_session, session))
                        session_tasks.append(task)
                        session_stats["created_by_rate"] += 1
                        print(f"\n{Colors.CYAN}[{elapsed:.0f}s] Created session {session.id} "
                              f"(rate-based, total: {len(sessions)}){Colors.END}")

                # Display status
                in_flight = metrics.get_in_flight()
                abandoned_count = sum(1 for s in sessions if s.abandoned)
                retired_count = sum(1 for s in sessions if s.retired)
                print(f"\r{Colors.BOLD}[{elapsed:6.1f}s]{Colors.END} "
                      f"Sessions: {active_sessions}/{len(sessions)} (+{session_stats['created_by_rate']} -{abandoned_count + retired_count}) | "
                      f"Requests: {metrics.requests_completed}/{metrics.requests_sent} | "
                      f"In-flight: {in_flight} | "
                      f"Errors: {metrics.errors}",
                      end="", flush=True)

                await asyncio.sleep(1.0)

            print(f"\n\n{Colors.YELLOW}Benchmark complete. Waiting for in-flight requests...{Colors.END}")

            # Wait for remaining tasks with timeout
            pending_tasks = [t for t in session_tasks if not t.done()]
            if pending_tasks:
                # Cancel remaining tasks after short wait
                await asyncio.sleep(5)
                for t in pending_tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
            for t in session_tasks:
                t.cancel()
            await asyncio.gather(*session_tasks, return_exceptions=True)
        finally:
            running_flag["running"] = False
            keyboard_listener.stop()
            if metrics_task:
                metrics_task.cancel()
                try:
                    await metrics_task
                except asyncio.CancelledError:
                    pass

    # Final summary
    print(f"\n{Colors.BOLD}Final Results (Realistic Mode):{Colors.END}")
    print(f"{Colors.DIM}{'-'*80}{Colors.END}")

    actual_duration = time.time() - metrics.start_time
    actual_average_qps = metrics.requests_sent / actual_duration if actual_duration > 0 else 0

    print(f"Total requests sent: {metrics.requests_sent:,}")
    print(f"Completed: {metrics.requests_completed:,}")
    print(f"Errors: {metrics.errors:,}")
    print(f"Success rate: {100 * metrics.requests_completed / max(metrics.requests_sent, 1):.1f}%")
    print(f"Actual benchmark duration: {actual_duration:.1f}s")
    print(f"Actual average QPS: {actual_average_qps:.2f} (target: {initial_qps:.2f} -> {max_qps:.2f})")

    # Calculate length distributions
    if metrics.actual_prompt_lengths:
        prompt_p50, prompt_p90, prompt_p99 = percentiles(metrics.actual_prompt_lengths, [0.5, 0.9, 0.99])
        prompt_mean = np.mean(metrics.actual_prompt_lengths)
        prompt_std = np.std(metrics.actual_prompt_lengths)

        print(f"\n{Colors.BOLD}Actual Prompt Length Distribution:{Colors.END}")
        print(f"  Mean: {prompt_mean:.0f} tokens")
        print(f"  Std Dev: {prompt_std:.0f} tokens")
        print(f"  p50: {prompt_p50:.0f} tokens")
        print(f"  p90: {prompt_p90:.0f} tokens")
        print(f"  p99: {prompt_p99:.0f} tokens")

    if metrics.actual_generation_lengths:
        gen_p50, gen_p90, gen_p99 = percentiles(metrics.actual_generation_lengths, [0.5, 0.9, 0.99])
        gen_mean = np.mean(metrics.actual_generation_lengths)
        gen_std = np.std(metrics.actual_generation_lengths)

        print(f"\n{Colors.BOLD}Actual Generation Length Distribution:{Colors.END}")
        print(f"  Mean: {gen_mean:.1f} tokens (target: {generation_length_mean})")
        print(f"  Median (p50): {gen_p50:.0f} tokens (target: {generation_length_median})")
        print(f"  Std Dev: {gen_std:.1f} tokens")
        print(f"  p90: {gen_p90:.0f} tokens")
        print(f"  p99: {gen_p99:.0f} tokens")

    # Calculate percentiles
    if metrics.prefill_samples:
        ttfts = [dur for _, _, dur, _, _, _, _, _ in metrics.prefill_samples]
        ttfts.sort()
        n = len(ttfts)
        p50_ttft = ttfts[n//2] if n > 0 else 0
        p90_ttft = ttfts[int(n*0.9)] if n > 0 else 0
        p99_ttft = ttfts[int(n*0.99)] if n > 0 else 0

        print(f"\n{Colors.BOLD}TTFT (Time to First Token):{Colors.END}")
        print(f"  p50: {p50_ttft*1000:.1f}ms")
        print(f"  p90: {p90_ttft*1000:.1f}ms")
        print(f"  p99: {p99_ttft*1000:.1f}ms")

        # TPOT (Time Per Output Token) — derived from per-request generation_time/(gen_len-1).
        if metrics.actual_tpots:
            tpot_p50, tpot_p90, tpot_p99 = percentiles(metrics.actual_tpots, [0.5, 0.9, 0.99])
            tpot_mean = float(np.mean(metrics.actual_tpots))
            print(f"\n{Colors.BOLD}TPOT (Time Per Output Token, excl. first):{Colors.END}")
            print(f"  Samples: {len(metrics.actual_tpots)} (filtered: gen_len>1 & gen_time>={MIN_GENERATION_TIME*1000:.0f}ms)")
            print(f"  Mean: {tpot_mean*1000:.1f}ms ({(1.0/tpot_mean):.1f} tok/s/req)" if tpot_mean > 0 else "  Mean: n/a")
            print(f"  p50: {tpot_p50*1000:.1f}ms")
            print(f"  p90: {tpot_p90*1000:.1f}ms")
            print(f"  p99: {tpot_p99*1000:.1f}ms")

        # Calculate average throughput over entire benchmark
        total_prefill_tokens = sum(tokens for _, tokens, _, _, _, _, _, _ in metrics.prefill_samples)
        avg_throughput_tps = total_prefill_tokens / actual_duration if actual_duration > 0 else 0
        avg_throughput_tpm = avg_throughput_tps * 60

        # Calculate average generation TPS (MTP compensated)
        generation_tps_values = [gen_tps_mtp for _, _, _, _, _, gen_tps_mtp, gen_time, _ in metrics.prefill_samples if gen_tps_mtp > 0 and gen_time >= MIN_GENERATION_TIME]
        filtered_count = sum(1 for _, _, _, _, _, gen_tps_mtp, gen_time, _ in metrics.prefill_samples if gen_tps_mtp > 0 and gen_time < MIN_GENERATION_TIME)
        avg_generation_tps = sum(generation_tps_values) / len(generation_tps_values) if generation_tps_values else 0.0

        # Peak throughput (scan all historical samples)
        best_throughput = 0
        for window in [1, 5, 10]:
            throughput, _ = metrics.get_peak_throughput(window)
            if throughput > best_throughput:
                best_throughput = throughput

        print(f"\n{Colors.BOLD}{Colors.GREEN}Peak Prefill Throughput:{Colors.END}")
        print(f"  Total: {best_throughput:,.0f} tokens/sec ({best_throughput*60:,.0f} tokens/min)")
        if num_gpus > 1:
            per_gpu_tps = best_throughput / num_gpus
            per_gpu_tpm = per_gpu_tps * 60
            print(f"  Per GPU: {per_gpu_tps:,.0f} tokens/sec ({per_gpu_tpm:,.0f} tokens/min)")

        print(f"\n{Colors.BOLD}Average Throughput:{Colors.END}")
        if num_gpus > 1:
            avg_tpm_per_gpu = avg_throughput_tpm / num_gpus
            print(f"  Context: {avg_tpm_per_gpu:,.0f} tokens/min/GPU ({avg_throughput_tps / num_gpus:,.0f} tokens/sec/GPU)")
        else:
            print(f"  Context: {avg_throughput_tpm:,.0f} tokens/min ({avg_throughput_tps:,.0f} tokens/sec)")

        if avg_generation_tps > 0:
            print(f"  Generation: {avg_generation_tps:.1f} tokens/sec (MTP compensated)")
            if filtered_count > 0:
                print(f"    (filtered {filtered_count} samples with generation_time < {MIN_GENERATION_TIME*1000:.0f}ms)")

        # Calculate average cache hit rate and eviction stats
        total_tokens = sum(tokens for _, tokens, _, _, _, _, _, _ in metrics.prefill_samples)
        total_cached = sum(cached for _, _, _, cached, _, _, _, _ in metrics.prefill_samples)
        total_prefix = sum(psize for _, _, _, _, _, _, _, psize in metrics.prefill_samples)
        avg_cache_hit_rate = total_cached / total_tokens if total_tokens > 0 else 0.0
        avg_ideal_cache_hit_rate = total_prefix / total_tokens if total_tokens > 0 else 0.0

        # Calculate eviction stats
        total_evicted = max(0, total_prefix - total_cached)
        eviction_rate = total_evicted / total_prefix if total_prefix > 0 else 0.0

        print(f"\n{Colors.BOLD}Cache Statistics:{Colors.END}")
        print(f"  Ideal cache hit rate: {avg_ideal_cache_hit_rate*100:.1f}% (assuming no eviction)")
        print(f"  Actual cache hit rate: {avg_cache_hit_rate*100:.1f}%")
        print(f"  Cache efficiency: {(avg_cache_hit_rate/avg_ideal_cache_hit_rate*100) if avg_ideal_cache_hit_rate > 0 else 0:.1f}% (actual/ideal)")
        print(f"  Eviction rate: {eviction_rate*100:.1f}% of expected cache was evicted")
        print(f"  Total tokens: {total_tokens:,} (prefix: {total_prefix:,}, cached: {total_cached:,}, evicted: {total_evicted:,})")
        if total_cached == 0 and avg_ideal_cache_hit_rate > 0.1:
            print(f"  {Colors.YELLOW}NOTE: server returned 0 cached tokens — its OpenAI-compatible API is not{Colors.END}")
            print(f"  {Colors.YELLOW}      reporting cache hits in usage.cache_read_input_tokens (or prompt_tokens_details.cached_tokens).{Colors.END}")
            print(f"  {Colors.YELLOW}      For sglang add --enable-cache-report to launch_server; for others enable{Colors.END}")
            print(f"  {Colors.YELLOW}      usage-level cache reporting. Meanwhile 'Actual cache hit rate' above is unreliable.{Colors.END}")

    # Ramp / sustain / drain throughput table
    phases = compute_phase_breakdown(metrics, metrics.start_time, ramp_duration_secs, sustain_duration_secs)
    print_phase_breakdown(phases, num_gpus=num_gpus)

    # Machine-readable summary for sweep aggregation / CI
    if 'run_dir' in locals() and run_dir is not None:
        write_run_summary(run_dir, metrics, phases, context={
            "mode": "realistic",
            "server_url": server_url,
            "model": model,
            "num_gpus": num_gpus,
            "ramp_duration_secs": ramp_duration_secs,
            "sustain_duration_secs": sustain_duration_secs,
            "max_inflight": max_inflight,
        })

    # Inter-arrival time stats (realistic mode specific)
    if metrics.inter_arrival_times:
        iat_p50, iat_p90, iat_p99 = percentiles(metrics.inter_arrival_times, [0.5, 0.9, 0.99])
        iat_mean = np.mean(metrics.inter_arrival_times)

        print(f"\n{Colors.BOLD}Inter-Arrival Time (response + think time):{Colors.END}")
        print(f"  Mean: {iat_mean:.1f}s")
        print(f"  p50: {iat_p50:.1f}s")
        print(f"  p90: {iat_p90:.1f}s")
        print(f"  p99: {iat_p99:.1f}s")

    # ChatSession statistics
    active_sessions = [s for s in sessions if not s.retired and not s.abandoned]
    retired_sessions = [s for s in sessions if s.retired]
    abandoned_sessions = [s for s in sessions if s.abandoned]

    print(f"\n{Colors.BOLD}ChatSession Statistics:{Colors.END}")
    print(f"  Total sessions: {len(sessions)} (initial: {num_initial_sessions}, +{session_stats['created_by_rate']} rate-based)")
    print(f"  Active: {len(active_sessions)}, Retired: {len(retired_sessions)}, Abandoned: {len(abandoned_sessions)} ({session_stats['abandoned_by_rate']} by rate)")
    if sessions:
        prefix_sizes = [s.prefix_tokens for s in sessions]
        print(f"  Final prefix sizes: min={min(prefix_sizes):,}, max={max(prefix_sizes):,}, mean={sum(prefix_sizes)//len(prefix_sizes):,}")
        lifetimes = [(time.time() - s.created_at) for s in sessions]
        print(f"  ChatSession lifetimes: min={min(lifetimes):.0f}s, max={max(lifetimes):.0f}s, mean={sum(lifetimes)/len(lifetimes):.0f}s")


# ---------------------------------------------------------------------------
# Dataset replay (Sub-mode B): replay real multi-turn agent traces.
#
# Each dataset row -> one trace. The K-th LLM call uses the first 2K-1 turns as
# its prompt and the K-th assistant turn's recorded length as max_tokens, so
# the server sees the real growing-prefix shape of a working coding agent.
# ---------------------------------------------------------------------------


def parse_trace_row(row: dict) -> List[Tuple[str, str]]:
    """Parse one dataset row into alternating (role, content) turns.

    Rows are the codex-traces shape: a `conversations` list of
    {"from": "human"/"gpt", "value": str}, strictly alternating and starting
    with a human turn. human -> user, gpt -> assistant.
    """
    return [
        ("assistant" if item["from"] == "gpt" else "user", item["value"])
        for item in row["conversations"]
    ]


_WALL_TIME_RE = re.compile(r"Wall time:\s*([0-9]+\.?[0-9]*)\s*seconds")
_DURATION_RE = re.compile(r'"duration_seconds"\s*:\s*([0-9]+\.?[0-9]*)')


def parse_tool_wait(text: str) -> Optional[float]:
    """Real tool-execution wall time recorded in a tool-output turn.

    codex traces stamp it as `Wall time: X seconds` (bash) or
    `"duration_seconds": X` (json tool results). Returns None when absent so
    the caller can fall back to the simulated wait.
    """
    m = _WALL_TIME_RE.search(text) or _DURATION_RE.search(text)
    return float(m.group(1)) if m else None


def load_agent_traces(dataset: str, split: str, num_traces: int, tokenizer) -> List[dict]:
    """Load the agent-traces dataset and pre-compute per-trace turn token counts.

    Returns a list of trace dicts with keys: id, turns [(role, content)],
    turn_tokens [int], num_calls, tool_waits [Optional[float]]. `tool_waits[k-1]`
    is the real tool-execution wall time recorded after call k (the latency
    before call k+1), or None when the trace didn't record one.
    `num_traces in (0, None)` loads the whole split.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            f"{Colors.RED}dataset-replay needs the 'datasets' package: "
            f"pip install 'datasets>=4.0'{Colors.END}"
        ) from exc

    print(f"{Colors.CYAN}Loading dataset {dataset} (split={split})...{Colors.END}")
    ds = load_dataset(dataset, split=split)
    n_total = len(ds)
    n = n_total if num_traces in (0, None) else min(num_traces, n_total)

    traces: List[dict] = []
    skipped = 0
    for i in range(n):
        turns = parse_trace_row(ds[i])
        num_calls = len(turns) // 2  # each call = a user turn followed by an assistant turn
        if num_calls < 1:
            skipped += 1
            continue
        turn_tokens = [
            len(tokenizer.encode(content, add_special_tokens=False))
            for _, content in turns
        ]
        # Real tool wall time after call k lives in the tool-output turn 2k.
        tool_waits = [
            parse_tool_wait(turns[2 * k][1]) if 2 * k < len(turns) else None
            for k in range(1, num_calls + 1)
        ]
        traces.append({
            "id": f"trace-{i}",
            "turns": turns,
            "turn_tokens": turn_tokens,
            "num_calls": num_calls,
            "tool_waits": tool_waits,
        })

    if not traces:
        raise SystemExit(f"{Colors.RED}No usable traces parsed from {dataset}{Colors.END}")
    calls = [t["num_calls"] for t in traces]
    inter_call = sum(max(0, t["num_calls"] - 1) for t in traces)
    recorded = sum(1 for t in traces for w in t["tool_waits"][:-1] if w is not None)
    cov = 100.0 * recorded / inter_call if inter_call else 0.0
    print(f"{Colors.GREEN}Loaded {len(traces)} traces ({skipped} skipped); "
          f"LLM calls/trace: min={min(calls)} p50={int(np.median(calls))} "
          f"max={max(calls)}{Colors.END}")
    print(f"{Colors.DIM}Recorded tool wall-times: {recorded}/{inter_call} inter-call gaps "
          f"({cov:.1f}%); the rest fall back to the simulated machine wait.{Colors.END}")
    return traces


def draw_agent_wait(mean: float, jitter: float, floor: float, cap: float = 300.0) -> float:
    """Interactive inter-turn wait, Gamma-distributed.

    `jitter` is the coefficient of variation (CV = std/mean):
      jitter=0   -> deterministic `mean`
      jitter=1.0 -> exponential (classic Poisson inter-arrival)
      jitter>1   -> long-tail
    Sampled as Gamma(shape=1/jitter^2, scale=mean*jitter^2) so E=mean, CV=jitter.
    The result is floored and capped.
    """
    if mean <= 0:
        return 0.0
    if jitter <= 0:
        wait = mean
    else:
        shape = 1.0 / (jitter * jitter)
        scale = mean * jitter * jitter
        wait = float(np.random.gamma(shape=shape, scale=scale))
    return max(floor, min(wait, cap))


async def run_dataset_replay(
    server_url: str,
    model: str,
    tokenizer,
    dataset: str,
    dataset_split: str,
    num_traces: int,
    concurrency: int,
    ramp_duration_secs: float,
    sustain_duration_secs: float,
    wait_machine_secs: float,
    wait_human_secs: float,
    wait_jitter: float,
    wait_scale: float = 1.0,
    api_key: str = None,
    window_size: float = 15.0,
    acc_len: float = 3.0,
    mtp_overhead_factor: float = 1.0,
    num_gpus: int = 1,
    random_seed: int = None,
    dashboard_mode: bool = False,
    benchmark_name: str = None,
    data_dir: Path = None,
    ignore_eos: bool = True,
    mtp_draft_tokens: int = 1,
):
    """Replay real multi-turn agent traces from a ShareGPT-format dataset.

    A fixed pool of `concurrency` walkers each pick a trace (round-robin) and
    replay it call by call: the K-th call sends the first 2K-1 turns as the
    prompt with max_tokens taken from the K-th assistant turn's recorded
    length. Between calls the driver waits the trace's *recorded* tool
    wall-time when available, else a simulated machine wait
    (`wait_machine_secs`/`wait_jitter`); `wait_scale` multiplies that gap. A
    human wait (`wait_human_secs`, default 0) is inserted at trace boundaries.
    """
    traces = load_agent_traces(dataset, dataset_split, num_traces, tokenizer)

    metrics = BenchMetrics()
    sessions: List[ChatSession] = []  # unused here; satisfies save_metrics_loop signature
    session_stats = {"created_by_rate": 0, "abandoned_by_rate": 0}

    metrics_file = None
    run_dir = None
    if dashboard_mode and benchmark_name and data_dir:
        storage = RunStorage(root_dir=data_dir)
        run_dir = storage.create_run_directory(benchmark_name)
        metrics_file = run_dir / "metrics.jsonl"
        storage.save_metadata(run_dir, {
            "mode": "dataset-replay",
            "server_url": server_url,
            "model": model,
            "dataset": dataset,
            "dataset_split": dataset_split,
            "num_traces": len(traces),
            "concurrency": concurrency,
            "ramp_duration_secs": ramp_duration_secs,
            "sustain_duration_secs": sustain_duration_secs,
            "wait_machine_secs": wait_machine_secs,
            "wait_human_secs": wait_human_secs,
            "wait_jitter": wait_jitter,
            "wait_scale": wait_scale,
            "num_gpus": num_gpus,
            "acc_len": acc_len,
            "mtp_overhead_factor": mtp_overhead_factor,
        })
        print(f"{Colors.GREEN}Benchmark mode enabled. Output: {run_dir}{Colors.END}")

    total_duration = ramp_duration_secs + sustain_duration_secs

    # Round-robin trace cursor shared across walkers (single-threaded asyncio).
    trace_cursor = {"i": 0}

    def next_trace() -> dict:
        t = traces[trace_cursor["i"] % len(traces)]
        trace_cursor["i"] += 1
        return t

    async def send_call(http_session, trace: dict, k: int) -> bool:
        """Send the K-th (1-indexed) LLM call of `trace`."""
        metrics.requests_sent += 1
        turns = trace["turns"]
        turn_tokens = trace["turn_tokens"]
        # prompt = first 2K-1 turns (ends on a user turn)
        prompt_turns = turns[: 2 * k - 1]
        messages = [{"role": r, "content": c} for r, c in prompt_turns]

        planned_prompt = sum(turn_tokens[: 2 * k - 1])
        # Ideal cacheable prefix = what the previous call already established:
        # turns[0:2k-2]. Zero for the first call in a trace.
        prefix_tokens = 0 if k <= 1 else sum(turn_tokens[: 2 * k - 2])
        # max_tokens = recorded length of the K-th assistant turn (index 2k-1)
        gen_len = max(1, turn_tokens[2 * k - 1])

        metrics.planned_prompt_lengths.append(planned_prompt)
        metrics.planned_ideal_cache_hit_rates.append(
            prefix_tokens / planned_prompt if planned_prompt > 0 else 0.0
        )
        metrics.request_timeline.append((time.time(), k == 1, trace["id"], False))

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": gen_len,
            "temperature": 0.0,
            "user": trace["id"],
            "ignore_eos": ignore_eos,
        }
        url = f"{server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        start_time = time.time()
        ttft = None
        cached_tokens = 0
        reasoning_tokens = 0
        actual_prompt_tokens = 0
        full_response = ""
        chunk_token_counts = []
        try:
            async with http_session.post(url, json=payload, headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=240)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"{Colors.RED}{trace['id']} call {k} failed: "
                          f"HTTP {resp.status} - {error_text[:200]}{Colors.END}")
                    metrics.errors += 1
                    return False

                async for line in resp.content:
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if not data or not isinstance(data, dict):
                            continue
                        if data.get("choices"):
                            delta = data["choices"][0].get("delta", {})
                            resp_content = delta.get("content", "")
                            resp_reason = delta.get("reasoning_content") or delta.get("reasoning")
                            if (resp_content or resp_reason) and ttft is None:
                                ttft = time.time() - start_time
                            if resp_reason:
                                full_response += resp_reason
                            if resp_content:
                                full_response += resp_content
                                chunk_tokens = len(tokenizer.encode(resp_content, add_special_tokens=False)) - 1
                                if chunk_tokens > 0:
                                    chunk_token_counts.append(chunk_tokens)
                        if data.get("usage"):
                            usage = data["usage"]
                            if isinstance(usage, dict):
                                if "prompt_tokens" in usage:
                                    actual_prompt_tokens = usage.get("prompt_tokens", 0)
                                if "cache_read_input_tokens" in usage:
                                    cached_tokens = usage.get("cache_read_input_tokens") or 0
                                elif "prompt_tokens_details" in usage:
                                    details = usage["prompt_tokens_details"]
                                    if isinstance(details, dict):
                                        cached_tokens = details.get("cached_tokens") or 0
                                if "reasoning_tokens" in usage:
                                    reasoning_tokens = usage.get("reasoning_tokens") or 0
                    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                        pass

                completion_tokens = len(tokenizer.encode(full_response, add_special_tokens=False)) if full_response else 0
                if ttft is None:
                    ttft = time.time() - start_time
                tokens_to_record = actual_prompt_tokens if actual_prompt_tokens > 0 else planned_prompt
                end_time = time.time()
                total_time = end_time - start_time
                generation_time = total_time - ttft if ttft and total_time > ttft else 0.0
                generation_tps = completion_tokens / generation_time if generation_time > 0 and completion_tokens > 0 else 0.0
                generation_tps_mtp = (completion_tokens * acc_len) / (generation_time * mtp_overhead_factor) if generation_time > 0 and completion_tokens > 0 else 0.0
                metrics.add_prefill(tokens_to_record, ttft, cached_tokens, generation_tps, generation_tps_mtp,
                                    completion_tokens, generation_time, prefix_tokens,
                                    reasoning_tokens=reasoning_tokens)
                if chunk_token_counts:
                    metrics.add_acceptance_length(sum(chunk_token_counts) / len(chunk_token_counts))
                else:
                    metrics.requests_completed += 1
                return True

        except asyncio.TimeoutError:
            metrics.errors += 1
            print(f"{Colors.RED}{trace['id']} call {k} timed out{Colors.END}")
            return False
        except Exception as e:
            metrics.errors += 1
            print(f"{Colors.RED}{trace['id']} call {k} error: {e}{Colors.END}")
            return False

    async def walker(http_session, walker_id: int):
        # Stagger walker starts across the ramp window.
        if ramp_duration_secs > 0 and concurrency > 1:
            await asyncio.sleep((walker_id / concurrency) * ramp_duration_secs)
        while time.time() - start_wall < total_duration:
            trace = next_trace()
            for k in range(1, trace["num_calls"] + 1):
                if time.time() - start_wall >= total_duration:
                    return
                await send_call(http_session, trace, k)
                if k < trace["num_calls"]:
                    rec = trace["tool_waits"][k - 1]
                    if rec is not None:
                        mw = min(rec * wait_scale, 300.0)
                    else:
                        mw = draw_agent_wait(wait_machine_secs, wait_jitter, floor=0.05) * wait_scale
                    if mw > 0:
                        await asyncio.sleep(mw)
            hw = draw_agent_wait(wait_human_secs, wait_jitter, floor=1.0)
            if hw > 0:
                await asyncio.sleep(hw)

    print(f"\n{Colors.BOLD}LLM Throughput Simulator (Dataset Replay Mode){Colors.END}")
    print(f"{Colors.DIM}{'-'*80}{Colors.END}")
    print(f"Server: {server_url}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset} (split={dataset_split}), traces: {len(traces)}")
    print(f"Concurrency: {concurrency} walkers")
    print(f"Waits: machine=recorded tool wall-time (fallback {wait_machine_secs}s, jitter={wait_jitter}), "
          f"scale={wait_scale}x, human={wait_human_secs}s")
    print(f"Duration: ramp={ramp_duration_secs}s + sustain={sustain_duration_secs}s")
    print(f"{Colors.DIM}{'='*80}{Colors.END}\n")

    start_wall = time.time()
    metrics.start_time = start_wall
    running_flag = {"running": True}
    metrics_task = None
    if metrics_file:
        metrics_task = asyncio.create_task(
            save_metrics_loop(metrics, sessions, metrics_file, window_size, num_gpus,
                              mtp_draft_tokens, running_flag, session_stats,
                              ramp_duration_secs=ramp_duration_secs,
                              sustain_duration_secs=sustain_duration_secs)
        )

    connector = aiohttp.TCPConnector(limit=1000)
    async with aiohttp.ClientSession(connector=connector) as http_session:
        walker_tasks = [asyncio.create_task(walker(http_session, i)) for i in range(concurrency)]
        try:
            while time.time() - start_wall < total_duration:
                elapsed = time.time() - start_wall
                in_flight = metrics.get_in_flight()
                print(f"\r{Colors.BOLD}[{elapsed:6.1f}s]{Colors.END} "
                      f"Requests: {metrics.requests_completed}/{metrics.requests_sent} | "
                      f"In-flight: {in_flight} | Errors: {metrics.errors}",
                      end="", flush=True)
                await asyncio.sleep(1.0)
            print(f"\n\n{Colors.YELLOW}Benchmark complete. Waiting for in-flight requests...{Colors.END}")
            await asyncio.sleep(5)
            for t in walker_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*walker_tasks, return_exceptions=True)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
            for t in walker_tasks:
                t.cancel()
            await asyncio.gather(*walker_tasks, return_exceptions=True)
        finally:
            running_flag["running"] = False
            if metrics_task:
                metrics_task.cancel()
                try:
                    await metrics_task
                except asyncio.CancelledError:
                    pass

    actual_duration = time.time() - metrics.start_time
    print(f"\n{Colors.BOLD}Final Results (Dataset Replay Mode):{Colors.END}")
    print(f"{Colors.DIM}{'-'*80}{Colors.END}")
    print(f"Total requests sent: {metrics.requests_sent:,}")
    print(f"Completed: {metrics.requests_completed:,}")
    print(f"Errors: {metrics.errors:,}")
    print(f"Success rate: {100 * metrics.requests_completed / max(metrics.requests_sent, 1):.1f}%")
    print(f"Actual benchmark duration: {actual_duration:.1f}s")

    if metrics.actual_prompt_lengths:
        p50, p90, p99 = percentiles(metrics.actual_prompt_lengths, [0.5, 0.9, 0.99])
        print(f"\n{Colors.BOLD}Actual Prompt Length Distribution:{Colors.END}")
        print(f"  Mean: {np.mean(metrics.actual_prompt_lengths):.0f}  p50: {p50:.0f}  p90: {p90:.0f}  p99: {p99:.0f} tokens")
    if metrics.actual_generation_lengths:
        p50, p90, p99 = percentiles(metrics.actual_generation_lengths, [0.5, 0.9, 0.99])
        print(f"\n{Colors.BOLD}Actual Generation Length Distribution:{Colors.END}")
        print(f"  Mean: {np.mean(metrics.actual_generation_lengths):.1f}  p50: {p50:.0f}  p90: {p90:.0f}  p99: {p99:.0f} tokens")

    phases = compute_phase_breakdown(metrics, metrics.start_time, ramp_duration_secs, sustain_duration_secs)
    print_phase_breakdown(phases, num_gpus=num_gpus)

    if run_dir is not None:
        write_run_summary(run_dir, metrics, phases, context={
            "mode": "dataset-replay",
            "server_url": server_url,
            "model": model,
            "dataset": dataset,
            "num_gpus": num_gpus,
            "concurrency": concurrency,
            "ramp_duration_secs": ramp_duration_secs,
            "sustain_duration_secs": sustain_duration_secs,
        })


# Workload config parameters that can be set via YAML config file
WORKLOAD_CONFIG_PARAMS = [
    "mode",
    "system_prompt_len",
    "new_tokens_mean",
    "new_tokens_median",
    "initial_prefix_mean",
    "initial_prefix_median",
    "initial_qps",
    "max_qps",
    "ramp_duration",
    "sustain_duration",
    "window",
    "gpus",
    "generation_length_mean",
    "generation_length_median",
    "acc_len",
    "mtp_overhead_factor",
    "mtp_draft_tokens",
    "min_prompt_tokens",
    "max_prompt_tokens",
    "poisson",
    "poisson_shape",
    "new_session_rate",
    "session_decay_lambda",
    "initial_sessions",
    "max_inflight",
    "random_seed",
    "tokenizer",
    # Realistic mode parameters
    "think_time_mean",
    "think_time_shape",
    "session_lifetime_mean",
    "session_lifetime_median",
    "max_sessions",
    "session_abandon_rate",
    # Dataset replay mode parameters
    "agent_dataset",
    "agent_dataset_split",
    "agent_num_traces",
    "agent_concurrency",
    "agent_wait_machine_secs",
    "agent_wait_human_secs",
    "agent_wait_jitter",
    "agent_wait_scale",
]


def merge_workload_yaml(config_path: str, args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Load workload parameters from a YAML config file.
    CLI arguments override config file values.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    workload = config.get("workload", {})
    if not workload:
        print(f"Warning: No 'workload' section found in {config_path}")
        return args
    
    applied = []
    skipped = []
    
    for param in WORKLOAD_CONFIG_PARAMS:
        if param not in workload:
            continue
        
        config_value = workload[param]
        attr_name = param.replace("-", "_")
        
        # Check if CLI provided a non-default value (CLI overrides config)
        default_value = parser.get_default(attr_name)
        current_value = getattr(args, attr_name)
        
        # Special handling for boolean flags like --poisson
        if isinstance(config_value, bool):
            if current_value != default_value:
                skipped.append(f"{param} (CLI override: {current_value})")
            else:
                setattr(args, attr_name, config_value)
                applied.append(f"{param}={config_value}")
        else:
            if current_value != default_value:
                skipped.append(f"{param} (CLI override: {current_value})")
            else:
                setattr(args, attr_name, config_value)
                applied.append(f"{param}={config_value}")
    
    print(f"Loaded workload config from: {config_path}")
    if applied:
        print(f"  Applied: {len(applied)} parameters")
    if skipped:
        print(f"  Skipped (CLI override): {len(skipped)} parameters")
        for s in skipped:
            print(f"    - {s}")
    
    return args


@dataclass
class PreviewTimings:
    """Timing breakdown for a single request in preview mode"""
    request_id: int
    ideal_fire_time: float  # When request would fire with zero overhead (cumulative sleep time)
    actual_fire_time: float  # When request actually fired (wall clock relative to start)
    text_gen_time: float  # Time spent in make_filler_seeded
    tokenization_time: float  # Time spent in tokenizer operations
    prompt_build_time: float  # Time spent building the prompt
    total_overhead: float  # Total overhead = actual - ideal delta for this request
    is_new_session: bool  # Whether this was a new session


def run_preview(
    initial_qps: float,
    max_qps: float,
    ramp_duration_secs: float,
    sustain_duration_secs: float,
    use_poisson: bool = False,
    poisson_shape: float = 1.0,
    random_seed: int = None,
    graph_width: int = 80,
    graph_height: int = 20,
    bucket_size: float = 1.0,
    # New parameters for full execution simulation
    tokenizer=None,
    system_prompt_len: int = 5000,
    new_tokens_mean: int = 5500,
    new_tokens_median: int = 1275,
    initial_prefix_mean: int = 76000,
    initial_prefix_median: int = 68000,
    new_session_rate: float = 0.03,
    max_prompt_tokens: int = 200000,
    num_initial_sessions: int = 1,
    session_decay_lambda: float = 0.02,
):
    """Preview mode: Run FULL execution loop except HTTP requests.

    This mode runs the actual scheduling loop including:
    - Text generation (make_filler_seeded)
    - Tokenization
    - Prompt building
    - ChatSession management

    It skips ONLY the actual HTTP request, revealing the true cost of
    all setup work that happens before each request is sent.

    Shows two graphs:
    - Ideal QPS: Based purely on sleep intervals (what we'd get with zero overhead)
    - Actual QPS: Based on real timestamps when requests would fire (includes all overhead)

    Args:
        initial_qps: Starting QPS for the ramp
        max_qps: Target QPS at end of ramp
        ramp_duration_secs: Duration of the ramp-up phase
        sustain_duration_secs: Duration of the sustain phase at max_qps
        use_poisson: Whether to use Poisson (gamma) arrival process
        poisson_shape: Shape parameter for gamma distribution
        random_seed: Seed for reproducibility
        graph_width: Width of the terminal graph in characters
        graph_height: Height of the terminal graph in characters
        bucket_size: Time bucket size in seconds for QPS calculation
        tokenizer: Tokenizer for text generation (required)
        system_prompt_len: System prompt length in tokens
        new_tokens_mean: Mean new tokens per request
        new_tokens_median: Median new tokens per request
        initial_prefix_mean: Mean initial prefix for new sessions
        initial_prefix_median: Median initial prefix for new sessions
        new_session_rate: Probability of creating a new session
        max_prompt_tokens: Max prefix size before session retirement
        num_initial_sessions: Number of sessions to start with
        session_decay_lambda: Decay rate for session selection
    """
    print(f"\n{Colors.BOLD}=== PREVIEW MODE (Full Execution Simulation) ==={Colors.END}")
    print(f"{Colors.CYAN}Running full execution loop - only HTTP requests are skipped{Colors.END}\n")

    if tokenizer is None:
        print(f"{Colors.RED}ERROR: Tokenizer is required for preview mode{Colors.END}")
        return [], []

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        print(f"Random seed: {random_seed}")

    base_seed = random_seed if random_seed is not None else 0

    print(f"Initial QPS: {initial_qps:.2f}")
    print(f"Max QPS: {max_qps:.2f}")
    print(f"Ramp duration: {ramp_duration_secs:.0f}s")
    print(f"Sustain duration: {sustain_duration_secs:.0f}s")
    print(f"Poisson arrivals: {use_poisson}" + (f" (shape={poisson_shape})" if use_poisson else ""))
    print(f"New tokens (mean/median): {new_tokens_mean}/{new_tokens_median}")
    print(f"Initial prefix (mean/median): {initial_prefix_mean}/{initial_prefix_median}")
    print(f"New session rate: {new_session_rate*100:.1f}%")
    print()

    # Generate system prompt (this is one-time setup cost, not per-request)
    print(f"{Colors.CYAN}Generating system prompt ({system_prompt_len:,} tokens)...{Colors.END}")
    system_prompt_start = time.time()
    system_prompt = make_filler(system_prompt_len, tokenizer)
    system_prompt_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    system_prompt_time = time.time() - system_prompt_start
    print(f"{Colors.GREEN}System prompt generated: {system_prompt_tokens:,} tokens in {system_prompt_time:.2f}s{Colors.END}")
    print()

    # Initialize sessions using spawn_session for pre-tokenized base text
    sessions: List[ChatSession] = []
    if num_initial_sessions > 0:
        for i in range(num_initial_sessions):
            # Use unique seed for each initial session
            seed = random_seed + i * 1000 if random_seed else i * 1000
            spawn_session(
                sessions=sessions,
                system_prompt=system_prompt,
                system_prompt_tokens=system_prompt_tokens,
                request_seq=-(num_initial_sessions - i),  # Negative seq for staggered ages
                tokenizer=tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                seed=seed,
                initial_prefix_tokens=0,  # Start with just system prompt
                max_sessions=100
            )

    # Timing data collection
    timing_breakdowns: List[PreviewTimings] = []
    ideal_fire_times: List[float] = []  # Cumulative sleep times (ideal schedule)
    actual_fire_times: List[float] = []  # Actual wall-clock times relative to start
    sleep_intervals: List[float] = []  # Individual sleep intervals

    total_duration = ramp_duration_secs + sustain_duration_secs
    ideal_elapsed = 0.0  # Cumulative sleep time (ideal schedule)
    request_id = 0

    print(f"{Colors.BOLD}Running simulation...{Colors.END}")
    print(f"{Colors.DIM}(This will take time as it runs actual text generation and tokenization){Colors.END}")
    print()

    wall_start_time = time.time()
    last_progress_print = wall_start_time
    last_progress_request_id = 0

    while ideal_elapsed < total_duration:
        # Calculate current target QPS (same logic as ramp_requests)
        if ideal_elapsed < ramp_duration_secs:
            progress = ideal_elapsed / ramp_duration_secs
            current_qps = initial_qps + (max_qps - initial_qps) * progress
        else:
            current_qps = max_qps

        # Calculate sleep interval
        mean_interval = 1.0 / current_qps
        if use_poisson:
            interval = np.random.gamma(shape=poisson_shape, scale=mean_interval/poisson_shape)
        else:
            interval = mean_interval

        # Record ideal fire time (before any overhead)
        ideal_fire_times.append(ideal_elapsed)
        sleep_intervals.append(interval)

        # === BEGIN ACTUAL EXECUTION WORK (same as dispatch_turn) ===

        request_work_start = time.time()

        # Pre-sample all random values (same as draw_turn_plan)
        plan = draw_turn_plan(
            request_id=request_id,
            new_tokens_mean=new_tokens_mean,
            new_tokens_median=new_tokens_median,
            generation_length_mean=1,  # Not used in preview
            generation_length_median=1,
            initial_prefix_mean=initial_prefix_mean,
            initial_prefix_median=initial_prefix_median,
            system_prompt_tokens=system_prompt_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )

        # Seed for new session creation (if needed)
        session_seed = base_seed + plan.request_id * 1000

        # Timing: text generation (now just session creation, if needed)
        text_gen_start = time.time()

        # Decide whether to create new session or use existing
        active_sessions = [s for s in sessions if not s.retired]
        use_new_session = plan.new_session_roll < new_session_rate or not active_sessions
        is_new_session = use_new_session

        if use_new_session:
            # Create a new session with pre-tokenized base text
            selected_session = spawn_session(
                sessions, system_prompt, system_prompt_tokens,
                plan.request_id,
                tokenizer=tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                seed=session_seed,
                initial_prefix_tokens=plan.initial_prefix_tokens,
                max_sessions=100
            )
        else:
            # Select existing session with pre-sampled roll value
            selected_session = pick_session_with_decay(
                sessions, plan.request_id, plan.session_select_roll,
                decay_lambda=session_decay_lambda
            )
            if selected_session is None:
                selected_session = spawn_session(
                    sessions, system_prompt, system_prompt_tokens,
                    plan.request_id,
                    tokenizer=tokenizer,
                    max_prompt_tokens=max_prompt_tokens,
                    seed=session_seed,
                    initial_prefix_tokens=plan.initial_prefix_tokens,
                    max_sessions=100
                )
                is_new_session = True

        text_gen_end = time.time()
        text_gen_time = text_gen_end - text_gen_start

        # Timing: prompt building (grow + decode)
        prompt_build_start = time.time()

        # Get current prefix tokens before growing
        current_prefix_tokens = selected_session.prefix_tokens

        # Clamp new_tokens to available space
        new_tokens = plan.new_tokens
        available_space = max_prompt_tokens - current_prefix_tokens
        if new_tokens > available_space:
            new_tokens = max(1, available_space)

        # Grow the session by new_tokens (extends the slice into pre-tokenized base).
        # In preview mode we don't fire the request, so we don't need the decoded
        # full_content / total_prompt_tokens — those were dead locals.
        selected_session.grow(new_tokens, plan.request_id)

        # Check if session should be retired
        if selected_session.should_retire(max_prompt_tokens):
            selected_session.retired = True

        prompt_build_end = time.time()
        prompt_build_time = prompt_build_end - prompt_build_start

        # Timing: tokenization (minimal now - just for metrics consistency)
        tokenization_start = time.time()
        tokenization_end = time.time()
        tokenization_time = tokenization_end - tokenization_start

        # === END ACTUAL EXECUTION WORK ===

        request_work_end = time.time()
        total_overhead = request_work_end - request_work_start

        # Record actual fire time (wall clock relative to start)
        actual_fire_time = time.time() - wall_start_time
        actual_fire_times.append(actual_fire_time)

        # Store timing breakdown
        timing_breakdowns.append(PreviewTimings(
            request_id=request_id,
            ideal_fire_time=ideal_elapsed,
            actual_fire_time=actual_fire_time,
            text_gen_time=text_gen_time,
            tokenization_time=tokenization_time,
            prompt_build_time=prompt_build_time,
            total_overhead=total_overhead,
            is_new_session=is_new_session,
        ))

        # Advance ideal elapsed time (this is what sleep would have been)
        ideal_elapsed += interval
        request_id += 1

        # Progress indicator (every 5 seconds of wall time)
        now = time.time()
        if now - last_progress_print > 5.0:
            pct_complete = (ideal_elapsed / total_duration) * 100
            interval_qps = (request_id - last_progress_request_id) / (now - last_progress_print)
            print(f"  Progress: {pct_complete:.1f}% ({request_id} requests, ideal_elapsed={ideal_elapsed:.1f}s), qps={interval_qps:.2f}")
            last_progress_print = now
            last_progress_request_id = request_id

    wall_end_time = time.time()
    total_wall_time = wall_end_time - wall_start_time

    print()
    print(f"{Colors.GREEN}Simulation complete!{Colors.END}")
    print(f"  Total requests: {len(timing_breakdowns)}")
    print(f"  Ideal duration: {ideal_elapsed:.2f}s")
    print(f"  Actual wall time: {total_wall_time:.2f}s")
    print(f"  Overhead ratio: {total_wall_time / ideal_elapsed:.2f}x")
    print()

    # === GRAPH 1: Ideal QPS (based on sleep intervals only) ===
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}GRAPH 1: Ideal QPS (Zero Overhead){Colors.END}")
    print(f"{Colors.DIM}Based purely on sleep intervals - what we'd get if setup was instant{Colors.END}")
    print()

    num_buckets = int(total_duration / bucket_size) + 1

    # Bucket ideal fire times
    ideal_bucket_counts = [0] * num_buckets
    for t in ideal_fire_times:
        bucket_idx = int(t / bucket_size)
        if bucket_idx < num_buckets:
            ideal_bucket_counts[bucket_idx] += 1

    ideal_qps_points = []
    for i, count in enumerate(ideal_bucket_counts):
        time_point = i * bucket_size
        qps = count / bucket_size
        ideal_qps_points.append((time_point, qps))

    # Target QPS curve
    target_qps_points = []
    for i in range(num_buckets):
        t = i * bucket_size
        if t < ramp_duration_secs:
            progress = t / ramp_duration_secs
            target_qps = initial_qps + (max_qps - initial_qps) * progress
        else:
            target_qps = max_qps
        target_qps_points.append((t, target_qps))

#    graph1 = LineGraph(graph_width, graph_height)
#    graph1.title = f"Ideal QPS (bucket={bucket_size}s)"
#    graph1.x_axis_relative = True
#    graph1.add_line(target_qps_points, color="cyan", label="Target QPS")
#    graph1.add_line(ideal_qps_points, color="green", label="Ideal QPS")
#    graph1.set_y_range(0, max_qps * 1.5)
#    print(graph1.render())

    # Stats for ideal QPS
    ideal_qps_values = [qps for _, qps in ideal_qps_points if qps > 0]
    if ideal_qps_values:
        print(f"  Ideal Avg QPS: {sum(ideal_qps_values)/len(ideal_qps_values):.2f}")
        print(f"  Ideal Peak QPS: {max(ideal_qps_values):.2f}")
        print(f"  Ideal Min QPS: {min(ideal_qps_values):.2f}")
    print()

    # === GRAPH 2: Actual QPS (including overhead) ===
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}GRAPH 2: Actual QPS (With Overhead){Colors.END}")
    print(f"{Colors.DIM}Based on real timestamps - includes text generation, tokenization, etc.{Colors.END}")
    print()

    # Bucket actual fire times
    actual_bucket_counts = [0] * num_buckets
    for t in actual_fire_times:
        bucket_idx = int(t / bucket_size)
        if bucket_idx < num_buckets:
            actual_bucket_counts[bucket_idx] += 1

    actual_qps_points = []
    for i, count in enumerate(actual_bucket_counts):
        time_point = i * bucket_size
        qps = count / bucket_size
        actual_qps_points.append((time_point, qps))

#    graph2 = LineGraph(graph_width, graph_height)
#    graph2.title = f"Actual QPS (bucket={bucket_size}s)"
#    graph2.x_axis_relative = True
#    graph2.add_line(target_qps_points, color="cyan", label="Target QPS")
#    graph2.add_line(actual_qps_points, color="yellow", label="Actual QPS")
#    graph2.set_y_range(0, max_qps * 1.5)
#    print(graph2.render())

    # Stats for actual QPS
    actual_qps_values = [qps for _, qps in actual_qps_points if qps > 0]
    if actual_qps_values:
        print(f"  Actual Avg QPS: {sum(actual_qps_values)/len(actual_qps_values):.2f}")
        print(f"  Actual Peak QPS: {max(actual_qps_values):.2f}")
        print(f"  Actual Min QPS: {min(actual_qps_values):.2f}")
    print()

    # === TIMING BREAKDOWN STATISTICS ===
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}TIMING BREAKDOWN{Colors.END}")
    print()

    # Collect timing data
    text_gen_times = [t.text_gen_time * 1000 for t in timing_breakdowns]  # Convert to ms
    tokenization_times = [t.tokenization_time * 1000 for t in timing_breakdowns]
    prompt_build_times = [t.prompt_build_time * 1000 for t in timing_breakdowns]
    total_overheads = [t.total_overhead * 1000 for t in timing_breakdowns]

    # New session vs existing session breakdown
    new_session_overheads = [t.total_overhead * 1000 for t in timing_breakdowns if t.is_new_session]
    existing_session_overheads = [t.total_overhead * 1000 for t in timing_breakdowns if not t.is_new_session]

    def print_timing_stats(name: str, values: List[float], unit: str = "ms"):
        if not values:
            print(f"  {name}: No data")
            return
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        avg = sum(values) / n
        p50 = sorted_vals[int(n * 0.5)]
        p90 = sorted_vals[int(n * 0.9)]
        p99 = sorted_vals[min(int(n * 0.99), n - 1)]
        print(f"  {name}:")
        print(f"    Avg: {avg:.2f}{unit}  |  P50: {p50:.2f}{unit}  |  P90: {p90:.2f}{unit}  |  P99: {p99:.2f}{unit}")

    print(f"{Colors.CYAN}Per-Request Timing:{Colors.END}")
    print_timing_stats("Text Generation", text_gen_times)
    print_timing_stats("Tokenization", tokenization_times)
    print_timing_stats("Prompt Building", prompt_build_times)
    print_timing_stats("Total Overhead", total_overheads)
    print()

    print(f"{Colors.CYAN}Overhead by ChatSession Type:{Colors.END}")
    print(f"  New sessions: {len(new_session_overheads)} ({100*len(new_session_overheads)/len(timing_breakdowns):.1f}%)")
    print_timing_stats("New ChatSession Overhead", new_session_overheads)
    print(f"  Existing sessions: {len(existing_session_overheads)} ({100*len(existing_session_overheads)/len(timing_breakdowns):.1f}%)")
    print_timing_stats("Existing ChatSession Overhead", existing_session_overheads)
    print()

    # Inter-arrival time analysis
    print(f"{Colors.CYAN}Inter-Arrival Time Analysis:{Colors.END}")
    if len(actual_fire_times) > 1:
        actual_inter_arrivals = [actual_fire_times[i+1] - actual_fire_times[i] for i in range(len(actual_fire_times)-1)]
        actual_inter_arrivals_ms = [t * 1000 for t in actual_inter_arrivals]
        print_timing_stats("Actual Inter-Arrival", actual_inter_arrivals_ms)

        ideal_inter_arrivals_ms = [t * 1000 for t in sleep_intervals[:-1]]
        print_timing_stats("Ideal Inter-Arrival (sleep)", ideal_inter_arrivals_ms)

        # Delta between actual and ideal
        deltas = [a - i for a, i in zip(actual_inter_arrivals_ms, ideal_inter_arrivals_ms)]
        print_timing_stats("Inter-Arrival Delta (actual - ideal)", deltas)
    print()

    # Summary comparison
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print()
    print(f"  Total Requests: {len(timing_breakdowns)}")
    print(f"  Ideal Duration: {ideal_elapsed:.2f}s")
    print(f"  Actual Duration: {total_wall_time:.2f}s")
    print(f"  Slowdown Factor: {total_wall_time / ideal_elapsed:.2f}x")
    print()

    if ideal_qps_values and actual_qps_values:
        ideal_avg = sum(ideal_qps_values) / len(ideal_qps_values)
        actual_avg = sum(actual_qps_values) / len(actual_qps_values)
        print(f"  Ideal Avg QPS: {ideal_avg:.2f}")
        print(f"  Actual Avg QPS: {actual_avg:.2f}")
        print(f"  QPS Efficiency: {100 * actual_avg / ideal_avg:.1f}%")
        print()

    print(f"  Avg Overhead per Request: {sum(total_overheads)/len(total_overheads):.2f}ms")
    target_interval_ms = 1000 / max_qps
    print(f"  Target Interval at Max QPS: {target_interval_ms:.2f}ms")
    print(f"  Overhead as % of Target Interval: {100 * (sum(total_overheads)/len(total_overheads)) / target_interval_ms:.1f}%")
    print()

    return ideal_fire_times, actual_fire_times, timing_breakdowns


def main():
    parser = argparse.ArgumentParser(
        description="LLM Throughput Simulator - Growing session prefixes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--server", type=str, default="http://localhost:19971",
                       help="Server URL (default: http://localhost:19971)")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name")
    parser.add_argument("--new-tokens-mean", type=int, default=8000,
                       help="Mean new tokens added per request (default: 8000)")
    parser.add_argument("--new-tokens-median", type=int, default=None,
                       help="Median new tokens per request (default: 90%% of mean)")
    parser.add_argument("--initial-prefix-mean", type=int, default=0,
                       help="Mean initial prefix size for new sessions, excluding system prompt (default: 0)")
    parser.add_argument("--initial-prefix-median", type=int, default=None,
                       help="Median initial prefix size for new sessions (default: 90%% of mean if > 0)")
    parser.add_argument("--initial-qps", type=float, default=1.0,
                       help="Initial queries per second (default: 1.0)")
    parser.add_argument("--max-qps", type=float, default=20.0,
                       help="Maximum queries per second (default: 20.0)")
    parser.add_argument("--ramp-duration", type=float, default=40.0,
                       help="Ramp duration in seconds (default: 40)")
    parser.add_argument("--sustain-duration", type=float, default=60.0,
                       help="Sustain duration at max QPS in seconds (default: 60)")
    parser.add_argument("--system-prompt-len", type=int, default=20000,
                       help="Length of synthetic system prompt in tokens (default: 20000)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for authentication")
    parser.add_argument("--window", type=float, default=15.0,
                       help="Window size in seconds for throughput smoothing (default: 15.0)")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs used by the server (default: 1)")
    parser.add_argument("--generation-length-mean", type=int, default=1,
                       help="Mean number of tokens to generate per request (default: 1)")
    parser.add_argument("--generation-length-median", type=int, default=1,
                       help="Median number of tokens to generate per request (default: 1)")
    parser.add_argument("--acc-len", type=float, default=3.0,
                       help="Average acceptance length for speculative decoding (for MTP compensated TPS) (default: 3.0)")
    parser.add_argument("--mtp-overhead-factor", type=float, default=1.0,
                       help="MTP overhead factor - generation time multiplier to account for MTP-disabled runs being faster (default: 1.0)")
    parser.add_argument("--mtp-draft-tokens", type=int, default=1,
                       help="Number of draft tokens per MTP step for acceptance rate calculation (default: 1)")
    parser.add_argument("--min-prompt-tokens", type=int, default=100,
                       help="Minimum prompt tokens (default: 100)")
    parser.add_argument("--max-prompt-tokens", type=int, default=200000,
                       help="Maximum prefix size before session retirement (default: 200000)")
    parser.add_argument("--poisson", action="store_true",
                       help="Use Poisson arrival process (exponential inter-arrival times) instead of uniform spacing")
    parser.add_argument("--poisson-shape", type=float, default=1.0,
                       help="Shape parameter for Gamma distribution (1=exponential/bursty, higher=smoother). Only used with --poisson")
    parser.add_argument("--disable-ignore-eos", action="store_true",
                       help="Disable ignore_eos (by default, ignore_eos=true forces generation to max_tokens). Disabling ignore_eos can result in malformed generation length distributions due to synthetic data.")
    parser.add_argument("--new-session-rate", type=float, default=0.00,
                       help="Probability of starting a new session (generating new prefix) instead of reusing existing prefix (default: 0.03 = 3%%)")
    parser.add_argument("--session-decay-lambda", type=float, default=0.02,
                       help="Decay rate for session selection recency bias (higher=stronger bias, default: 0.02 gives ~35s half-life)")
    parser.add_argument("--initial-sessions", type=int, default=1,
                       help="Number of sessions (unique prefixes) to start with (default: 1)")
    parser.add_argument("--max-inflight", type=int, default=None,
                       help="Maximum in-flight requests before pausing (backpressure). None = no limit (default: None)")

    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/DeepSeek-V3",
                       help="Tokenizer for token counting. Can be a HuggingFace model name (e.g., deepseek-ai/DeepSeek-V3) or a local path to tokenizer files (default: deepseek-ai/DeepSeek-V3)")
    parser.add_argument("--random-seed", type=int, default=None,
                       help="Random seed for reproducibility (default: None, no seed set)")

    # Benchmark mode arguments (for live dashboard)
    parser.add_argument("--dashboard-mode", action="store_true",
                       help="Enable dashboard mode with metrics.jsonl output for live dashboard")
    parser.add_argument("--name", type=str, default=None,
                       help="Benchmark name (required for dashboard mode, e.g., 'my-test')")
    parser.add_argument("--data-dir", type=str, default="benchmarks",
                       help="Directory to store benchmark data (default: benchmarks)")
    parser.add_argument("--workload-config", type=str, default=None,
                       help="YAML config file for workload parameters (CLI args override config values)")

    # Mode selection
    parser.add_argument("--mode", type=str, default="traffic-replay",
                       choices=["traffic-replay", "realistic", "preview", "dataset-replay"],
                       help="Benchmark mode: 'traffic-replay' (default), 'realistic', 'preview' (plan-only visualization), or 'dataset-replay' (replay real agent traces)")

    # Dataset replay mode arguments (Sub-mode B)
    parser.add_argument("--agent-dataset", type=str, default="Inferact/codex_swebenchpro_traces",
                       help="[Dataset replay] HuggingFace dataset of ShareGPT-format agent traces (default: Inferact/codex_swebenchpro_traces)")
    parser.add_argument("--agent-dataset-split", type=str, default="train",
                       help="[Dataset replay] Dataset split to load (default: train)")
    parser.add_argument("--agent-num-traces", type=int, default=0,
                       help="[Dataset replay] Number of traces to load (0 = whole split) (default: 0)")
    parser.add_argument("--agent-concurrency", type=int, default=8,
                       help="[Dataset replay] Number of concurrent trace walkers (default: 8)")
    parser.add_argument("--agent-wait-machine-secs", type=float, default=2.0,
                       help="[Dataset replay] Simulated tool-execution wait, used only as a FALLBACK for calls with no recorded wall-time (default: 2.0)")
    parser.add_argument("--agent-wait-human-secs", type=float, default=0.0,
                       help="[Dataset replay] Human-in-the-loop wait at trace boundaries; 0 = autonomous batch (default: 0.0)")
    parser.add_argument("--agent-wait-jitter", type=float, default=0.0,
                       help="[Dataset replay] CV for the simulated/fallback waits; 0=deterministic, 1.0=Poisson, >1=long-tail (default: 0.0)")
    parser.add_argument("--agent-wait-scale", type=float, default=1.0,
                       help="[Dataset replay] Multiplier on the inter-call machine wait (recorded or fallback); 0 disables, <1 speeds up replay (default: 1.0)")

    # Realistic mode arguments
    parser.add_argument("--think-time-mean", type=float, default=10.0,
                       help="[Realistic mode] Mean think time in seconds between response and next request (default: 10.0)")
    parser.add_argument("--think-time-shape", type=float, default=1.0,
                       help="[Realistic mode] Gamma shape parameter for think time (1.0=exponential, higher=less variance) (default: 1.0)")
    parser.add_argument("--session-lifetime-mean", type=float, default=600.0,
                       help="[Realistic mode] Mean session lifetime in seconds before retirement (default: 600.0)")
    parser.add_argument("--session-lifetime-median", type=float, default=400.0,
                       help="[Realistic mode] Median session lifetime in seconds (lognormal distribution) (default: 400.0)")
    parser.add_argument("--max-sessions", type=int, default=100,
                       help="[Realistic mode] Maximum number of concurrent sessions (default: 100)")
    parser.add_argument("--session-abandon-rate", type=float, default=0.0,
                       help="[Realistic mode] Probability per request to abandon session (default: 0.0)")

    args = parser.parse_args()

    # Load workload config if specified (CLI args override config values)
    if args.workload_config:
        args = merge_workload_yaml(args.workload_config, args, parser)

    # Validate benchmark mode arguments
    if args.dashboard_mode and not args.name:
        parser.error("--name is required when using --dashboard-mode")

    # Set default median to 90% of mean if not specified
    if args.new_tokens_median is None:
        args.new_tokens_median = int(args.new_tokens_mean * 0.9) if args.new_tokens_mean > 0 else 1
    
    # Set default initial prefix median
    if args.initial_prefix_median is None:
        args.initial_prefix_median = int(args.initial_prefix_mean * 0.9) if args.initial_prefix_mean > 0 else 0

    # Initialize tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    # Check if it looks like a local path and validate it exists
    if args.tokenizer.startswith("/") or args.tokenizer.startswith("./") or args.tokenizer.startswith("../"):
        tokenizer_path = Path(args.tokenizer)
        if not tokenizer_path.exists():
            print(f"{Colors.RED}ERROR: Tokenizer path does not exist: {args.tokenizer}{Colors.END}")
            sys.exit(1)
        if not (tokenizer_path / "tokenizer.json").exists() and not (tokenizer_path / "tokenizer_config.json").exists():
            print(f"{Colors.RED}ERROR: No tokenizer files found in: {args.tokenizer}{Colors.END}")
            print(f"{Colors.RED}Expected tokenizer.json or tokenizer_config.json{Colors.END}")
            sys.exit(1)
    base_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer = CachingTokenizer(base_tokenizer)
    print(f"Tokenizer loaded successfully (vocab size: {tokenizer.vocab_size})")

    # Set random seeds for reproducibility (only if specified)
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        print(f"Random seed set to: {args.random_seed}")

    # Run benchmark based on mode
    if args.mode == "preview":
        # Preview mode now runs full execution simulation (minus HTTP)
        run_preview(
            initial_qps=args.initial_qps,
            max_qps=args.max_qps,
            ramp_duration_secs=args.ramp_duration,
            sustain_duration_secs=args.sustain_duration,
            use_poisson=args.poisson,
            poisson_shape=args.poisson_shape,
            random_seed=args.random_seed,
            tokenizer=tokenizer,
            system_prompt_len=args.system_prompt_len,
            new_tokens_mean=args.new_tokens_mean,
            new_tokens_median=args.new_tokens_median,
            initial_prefix_mean=args.initial_prefix_mean,
            initial_prefix_median=args.initial_prefix_median,
            new_session_rate=args.new_session_rate,
            max_prompt_tokens=args.max_prompt_tokens,
            num_initial_sessions=args.initial_sessions,
            session_decay_lambda=args.session_decay_lambda,
        )
    elif args.mode == "realistic":
        print(f"{Colors.BOLD}Running in REALISTIC mode{Colors.END}")
        print(f"{Colors.DIM}(Response-chained sessions, max 1 in-flight per session){Colors.END}")
        asyncio.run(run_session_walk(
            server_url=args.server,
            model=args.model,
            system_prompt_len=args.system_prompt_len,
            new_tokens_mean=args.new_tokens_mean,
            new_tokens_median=args.new_tokens_median,
            initial_qps=args.initial_qps,
            max_qps=args.max_qps,
            ramp_duration_secs=args.ramp_duration,
            sustain_duration_secs=args.sustain_duration,
            tokenizer=tokenizer,
            api_key=args.api_key,
            window_size=args.window,
            generation_length_mean=args.generation_length_mean,
            generation_length_median=args.generation_length_median,
            acc_len=args.acc_len,
            mtp_overhead_factor=args.mtp_overhead_factor,
            num_gpus=args.gpus,
            max_prompt_tokens=args.max_prompt_tokens,
            num_initial_sessions=args.initial_sessions,
            random_seed=args.random_seed,
            initial_prefix_mean=args.initial_prefix_mean,
            initial_prefix_median=args.initial_prefix_median,
            max_inflight=args.max_inflight,
            dashboard_mode=args.dashboard_mode,
            benchmark_name=args.name,
            ignore_eos=not args.disable_ignore_eos,
            data_dir=Path(args.data_dir) if args.data_dir else None,
            mtp_draft_tokens=args.mtp_draft_tokens,
            # Realistic mode specific parameters
            think_time_mean=args.think_time_mean,
            think_time_shape=args.think_time_shape,
            session_lifetime_mean=args.session_lifetime_mean,
            session_lifetime_median=args.session_lifetime_median,
            max_sessions=args.max_sessions,
            new_session_rate=args.new_session_rate,
            session_abandon_rate=args.session_abandon_rate,
        ))
    elif args.mode == "dataset-replay":
        print(f"{Colors.BOLD}Running in DATASET REPLAY mode{Colors.END}")
        print(f"{Colors.DIM}(Replaying real multi-turn agent traces; growing-prefix per trace){Colors.END}")
        asyncio.run(run_dataset_replay(
            server_url=args.server,
            model=args.model,
            tokenizer=tokenizer,
            dataset=args.agent_dataset,
            dataset_split=args.agent_dataset_split,
            num_traces=args.agent_num_traces,
            concurrency=args.agent_concurrency,
            ramp_duration_secs=args.ramp_duration,
            sustain_duration_secs=args.sustain_duration,
            wait_machine_secs=args.agent_wait_machine_secs,
            wait_human_secs=args.agent_wait_human_secs,
            wait_jitter=args.agent_wait_jitter,
            wait_scale=args.agent_wait_scale,
            api_key=args.api_key,
            window_size=args.window,
            acc_len=args.acc_len,
            mtp_overhead_factor=args.mtp_overhead_factor,
            num_gpus=args.gpus,
            random_seed=args.random_seed,
            dashboard_mode=args.dashboard_mode,
            benchmark_name=args.name,
            data_dir=Path(args.data_dir) if args.data_dir else None,
            ignore_eos=not args.disable_ignore_eos,
            mtp_draft_tokens=args.mtp_draft_tokens,
        ))
    else:
        print(f"{Colors.BOLD}Running in TRAFFIC REPLAY mode{Colors.END}")
        print(f"{Colors.DIM}(Deterministic traffic, may have concurrent requests per session){Colors.END}")
        asyncio.run(run_replay(
            server_url=args.server,
            model=args.model,
            system_prompt_len=args.system_prompt_len,
            new_tokens_mean=args.new_tokens_mean,
            new_tokens_median=args.new_tokens_median,
            initial_qps=args.initial_qps,
            max_qps=args.max_qps,
            ramp_duration_secs=args.ramp_duration,
            sustain_duration_secs=args.sustain_duration,
            tokenizer=tokenizer,
            api_key=args.api_key,
            window_size=args.window,
            generation_length_mean=args.generation_length_mean,
            generation_length_median=args.generation_length_median,
            acc_len=args.acc_len,
            mtp_overhead_factor=args.mtp_overhead_factor,
            num_gpus=args.gpus,
            max_prompt_tokens=args.max_prompt_tokens,
            use_poisson=args.poisson,
            poisson_shape=args.poisson_shape,
            new_session_rate=args.new_session_rate,
            num_initial_sessions=args.initial_sessions,
            random_seed=args.random_seed,
            initial_prefix_mean=args.initial_prefix_mean,
            initial_prefix_median=args.initial_prefix_median,
            max_inflight=args.max_inflight,
            session_decay_lambda=args.session_decay_lambda,
            dashboard_mode=args.dashboard_mode,
            benchmark_name=args.name,
            ignore_eos=not args.disable_ignore_eos,
            data_dir=Path(args.data_dir) if args.data_dir else None,
            mtp_draft_tokens=args.mtp_draft_tokens,
        ))


if __name__ == "__main__":
    main()
