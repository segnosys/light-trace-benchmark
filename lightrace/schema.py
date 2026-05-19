import dataclasses
import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LatencyProfile:
    """
    Holds various latency and throughput measurements for a single inference call.

    Attributes:
        first_token_latency (float): Time to first token in milliseconds.
        token_throughput (float): Tokens generated per second.
        end_to_end_ms (float): Total round-trip latency in milliseconds.
        input_token_count (int): Number of tokens in the input prompt.
        output_token_count (int): Number of tokens in the generated output.
        output_char_count (int): Number of characters in the generated output.
        ms_per_token (float): Milliseconds per output token.
        char_throughput (float): Characters generated per second.
        ms_per_char (float): Milliseconds per output character.
        accept_ratio (float): Speculative decoding acceptance ratio.
    """

    first_token_latency: Optional[float] = None
    token_throughput: Optional[float] = None
    end_to_end_ms: Optional[float] = None
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    output_char_count: Optional[int] = None
    ms_per_token: Optional[float] = None
    char_throughput: Optional[float] = None
    ms_per_char: Optional[float] = None
    accept_ratio: Optional[float] = None
    # Prompt-cache reporting from the backend's usage block.
    # - OpenAI / sglang / vllm: usage.prompt_tokens_details.cached_tokens
    # - Anthropic: usage.cache_read_input_tokens (reads) and
    #              usage.cache_creation_input_tokens (writes — first request
    #              that creates the cache entry).
    cached_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict())


@dataclass
class InferencePayload:
    """
    Encapsulates the parameters for a single inference API call.

    Attributes:
        stream (bool): Whether to use streaming mode.
        temperature (float): Sampling temperature.
        top_p (Optional[float]): Nucleus sampling threshold.
        logprobs (Optional[bool]): Whether to return log probabilities.
        top_logprobs (Optional[int]): Number of top log probabilities to return.
        skip_eos (bool): Whether to ignore end-of-sequence token.
        max_tokens (int): Maximum number of tokens to generate.
        n (int): Number of completions to generate.
        adapter_path (Optional[str]): Path to LoRA adapter (e.g., s3://...).
    """

    prompt: Optional[str] = None
    messages: Optional[List[str]] = None
    stream: bool = True
    temperature: float = 1.0
    top_p: Optional[float] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    skip_eos: bool = False
    max_tokens: int = 4096
    n: Optional[int] = None
    prediction: Optional[str] = None
    adapter_path: Optional[str] = None
    lora_name: Optional[str] = None
    reasoning_effort: Optional[str] = None
    reasoning: Optional[dict] = None
    # Explicit cache breakpoint hint. When set, backends that honor explicit
    # cache markers (Anthropic) split the prompt at the end of this text and
    # mark only the prefix portion with cache_control. Servers with implicit
    # prefix-cache detection (sglang, vllm, OpenAI) ignore it — their radix
    # trees pick up any shared prefix automatically.
    cacheable_prefix: Optional[str] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict())


@dataclass
class ResultEntry:
    model: Optional[str] = None
    request: Optional[InferencePayload] = None
    content: Optional[str] = None
    metrics: Optional[LatencyProfile] = None
    success: bool = False

    def to_dict(self):
        return {
            "model": self.model,
            "request": dataclasses.asdict(self.request) if self.request else None,
            "content": self.content,
            "metrics": dataclasses.asdict(self.metrics) if self.metrics else None,
            "success": self.success,
        }

    def to_json(self):
        return json.dumps(self.to_dict())


@dataclass
class FragmentInfo:
    """
    Holds metadata for a single streamed chunk of the response.
    """

    text: str
    logprob_tokens: Optional[int]
    usage_tokens: Optional[int]
    prompt_usage_tokens: Optional[int]
    accept_ratio: Optional[float] = None
    # Per-chunk cache fields — backends populate these from usage when available.
    cached_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None


@dataclass
class OutputDistribution:
    avg_len: int
    stdev_len: float
    min_len: int
    max_len: int
    total_tokens: int


@dataclass
class BackendInfo:
    name: str
    model: str


@dataclass
class LoadShape:
    mode: str
    level: float


@dataclass
class StatsSummary:
    mean: float
    stdev: float
    p05: float
    p50: float
    p80: float
    p95: float
    p99: float
    p999: float
    distribution: Optional[List[float]] = None


@dataclass
class RunOverview:
    total_num_requests: int
    total_elapsed_time_s: float
    job_level_tps: float
    actual_qps: float
    num_failed_requests: int


@dataclass
class DeviceThroughput:
    num_gpus: int
    tps_mean: float
    tps_stdev: float


@dataclass
class BenchmarkReport:
    backend: BackendInfo
    load: LoadShape
    input: OutputDistribution
    output: OutputDistribution
    ttft: StatsSummary
    user_tps: StatsSummary
    e2e: StatsSummary
    overview: RunOverview
    per_device: Optional[DeviceThroughput] = None
    accept_ratio: Optional[StatsSummary] = None
    hf_dataset_name: Optional[str] = None
    # Aggregated cache stats across the run; None if no backend reported cache info.
    cache_hit_rate: Optional[StatsSummary] = None
    total_cached_input_tokens: Optional[int] = None
    # Workload-predicted ideal cache hit rate (0..1). Filled in by run.py from
    # the workload parameters (same_prompts_in_burst, synthetic_cached_input_length,
    # gsp_cached_fraction, etc.). Compare against cache_hit_rate.mean to spot
    # broken or absent cache reporting on the server side.
    ideal_cache_hit_rate: Optional[float] = None
    # Server-side metrics scraped from the inference server's /metrics endpoint
    # (when --engine_metrics_url is set). Maps metric name -> {mean, max, latest, samples}.
    engine_metrics: Optional[dict] = None
