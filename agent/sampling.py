"""
Distribution-sampling helpers shared across the agent driver.

Originally lived in `agent_throughput.py`. Extracted into a leaf module so
unit tests can exercise them in isolation and so the main file can shrink
toward its load-driving role.

Public API (kept stable for back-compat with the parent module):

  - MIN_GENERATION_TIME, MIN_THINK_TIME, MAX_THINK_TIME,
    MAX_INTER_ARRIVAL_TIME, MAX_RETRIES — module-level constants
  - draw_lognormal(mean, median) -> int
  - draw_session_lifetime(mean, median) -> float
  - draw_think_time(mean, shape=1.0) -> float

The text-content generators (`make_filler*`, `make_shared_content`,
corpus loader) are NOT in here yet — they depend on the tokenizer and
on global corpus state that's harder to lift cleanly. Will follow.
"""
import math

import numpy as np

# Generation-time floor below which TPOT samples are too noisy to record.
MIN_GENERATION_TIME = 0.05  # 50 ms

# Think-time bounds for the realistic-traffic mode.
MIN_THINK_TIME = 5.0       # Minimum floor in seconds
MAX_THINK_TIME = 900.0     # Maximum cap (production p99 ~15 min)

# Inter-arrival cap so an extremely slow response doesn't push think time
# beyond what looks like a reasonable user behavior.
MAX_INTER_ARRIVAL_TIME = 300.0  # seconds

# How many times a session retries before we mark it abandoned.
MAX_RETRIES = 3


def draw_lognormal(mean: float, median: float) -> int:
    """Sample from lognormal distribution given mean and median.

    Returns an int (token count or similar discrete quantity).
    When mean <= median, falls back to a tight sigma so we don't sample
    far below the median.
    """
    if median <= 0 or mean <= 0:
        return max(1, int(mean))
    mu = math.log(median)
    if mean > median:
        sigma = math.sqrt(2 * math.log(mean / median))
    else:
        ratio = abs(mean - median) / mean if mean > 0 else 0
        if ratio < 0.001:  # within 0.1% => ~1% CV
            sigma = 0.01
        else:
            sigma = 0.1
    return max(1, int(np.random.lognormal(mean=mu, sigma=sigma)))


def draw_session_lifetime(mean: float, median: float) -> float:
    """Sample session lifetime from a lognormal distribution (seconds).

    Same shape as draw_lognormal but returns a float with a 60 s minimum
    so we never produce a session that retires before its first request.
    """
    if median <= 0 or mean <= 0:
        return max(60.0, mean)
    mu = math.log(median)
    if mean > median:
        sigma = math.sqrt(2 * math.log(mean / median))
    else:
        ratio = abs(mean - median) / mean if mean > 0 else 0
        if ratio < 0.001:
            sigma = 0.01
        else:
            sigma = 0.1
    return max(60.0, np.random.lognormal(mean=mu, sigma=sigma))


def draw_think_time(mean: float, shape: float = 1.0) -> float:
    """Sample think time from a gamma distribution, clamped to [MIN, MAX].

    `shape` controls variance:
      - 1.0 -> exponential (high variance)
      - higher -> tighter around mean
    `scale = mean / shape` keeps the gamma mean equal to `mean`.
    """
    if mean <= 0:
        return MIN_THINK_TIME
    scale = mean / shape
    sampled = np.random.gamma(shape=shape, scale=scale)
    return min(MAX_THINK_TIME, max(MIN_THINK_TIME, sampled))
