import abc
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import aiohttp
import orjson
from transformers import AutoTokenizer

from lightrace.schema import (
    FragmentInfo,
    InferencePayload,
    LatencyProfile,
    ResultEntry,
)

HTTP_SESSION_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
HTTP_READ_BUFFER = 256 * 1024  # 256KB, default is 64KB


def extract_trace_id_from_headers(headers) -> str:
    """
    Extract request ID from response headers.

    Args:
        headers: Response headers from aiohttp response

    Returns:
        str: Request ID from x-request-id header, or cf-ray header as fallback, or "N/A"
    """
    return headers.get("x-request-id") or headers.get("cf-ray", "N/A")


class BaseBackend(abc.ABC):
    """
    Abstract base class for inference API backends.

    Attributes:
        base_url (str): The base URL for the API.
        api_key (str): The API key for authentication.
        model_name (str): The name of the model to use.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        force_recounting_completions: bool = False,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.force_recounting_completions = force_recounting_completions

        if tokenizer_name:
            self.tokenizer_name = tokenizer_name
        else:
            self.tokenizer_name = model_name

        # Shared session — lazy-created on first request, reused for every
        # subsequent call. Replaces the prior per-request ClientSession that
        # paid TCP+TLS handshake on every benchmark request. The connector
        # cap allows enough concurrent sockets for our typical concurrency
        # range (up to 256) and `force_close=False` keeps keep-alive on.
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared ClientSession, creating it on first use."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=256,        # max concurrent connections across the pool
                limit_per_host=256,  # all of ours hit one host
                force_close=False,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                headers=self.build_headers(),
                timeout=HTTP_SESSION_TIMEOUT,
                read_bufsize=HTTP_READ_BUFFER,
                connector=connector,
            )
        return self._session

    async def close(self) -> None:
        """Close the shared session. Safe to call multiple times.

        Always clears the reference, even when the underlying session was
        already closed externally — keeps `_get_session()` semantics simple
        (None == no live session, recreate on next request).
        """
        try:
            if self._session is not None and not self._session.closed:
                await self._session.close()
        finally:
            self._session = None

    @abc.abstractmethod
    def build_endpoint_url(self, request: InferencePayload) -> str:
        pass

    @abc.abstractmethod
    def build_request_body(self, request: InferencePayload) -> Dict:
        pass

    @abc.abstractmethod
    def build_headers(self) -> Dict:
        pass

    @abc.abstractmethod
    def decode_response_chunk(self, data: Any, request: InferencePayload) -> Optional[FragmentInfo]:
        pass

    def before_request(self, request: InferencePayload):
        pass

    def after_request(self, request: InferencePayload, metrics: LatencyProfile):
        pass

    def count_prompt_tokens(self, request: InferencePayload) -> int:
        return -1

    def count_text_tokens(self, text: str) -> int:
        return -1

    async def execute_call(self, request: InferencePayload) -> ResultEntry:
        """
        Sends an asynchronous request to the backend's API.

        Args:
            request (InferencePayload): The request object.

        Returns:
            ResultEntry: The result of the inference call.
        """
        self.before_request(request)

        payload = self.build_request_body(request)
        logging.debug(payload)
        session = await self._get_session()
        failed_result = ResultEntry(success=False)
        t_start = time.perf_counter()

        try:
            async with session.post(self.build_endpoint_url(request), json=payload) as response:
                if response.status != 200:
                    error_bytes = b""
                    async for chunk_bytes in response.content:
                        error_bytes += chunk_bytes
                    error_text = error_bytes.decode("utf-8")
                    failed_result.content = error_text

                    trace_id = extract_trace_id_from_headers(response.headers)
                    logging.warning(
                        f"Request failed with status {response.status}, "
                        f"request-id: {trace_id}"
                    )
                    logging.warning(error_text)
                    return failed_result

                accumulated_text = ""
                finished = False
                total_usage_tokens = None
                total_logprob_tokens = None
                t_first_token = None
                accept_ratio = None
                # Prompt-cache accounting from backend usage objects.
                cached_input_tokens = None
                cache_creation_input_tokens = None
                # Sticky across chunks — only set when a fragment carries it.
                # Anthropic reports prompt_tokens in message_start only; text
                # deltas in between have prompt_usage_tokens=None and must
                # not overwrite the real value.
                prompt_usage_tokens = 0
                raw_chunks = []

                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()

                    if not chunk_bytes:
                        continue

                    line = chunk_bytes.decode("utf-8")
                    raw_chunks.append(line)

                    if finished:
                        if "[DONE]" not in line:
                            print(f"WARNING: Received more chunks after [DONE]: {line}")
                        continue
                    now = time.perf_counter()

                    # stream mode
                    if request.stream:
                        if not line.startswith("data:"):
                            continue
                        line = line[len("data:"):]
                        if line.strip() == "[DONE]":
                            finished = True
                            continue

                    data = orjson.loads(line)

                    # parse output
                    fragment = self.decode_response_chunk(data=data, request=request)

                    if fragment is None:
                        return failed_result

                    if fragment.usage_tokens:
                        total_usage_tokens = max(total_usage_tokens or 0, fragment.usage_tokens)

                    if fragment.prompt_usage_tokens:
                        prompt_usage_tokens = fragment.prompt_usage_tokens

                    # capture acceptance rate from chunk metadata if available
                    if fragment.accept_ratio is not None:
                        accept_ratio = fragment.accept_ratio

                    # capture prompt-cache info — usage chunks typically appear
                    # at end of stream, so we take the last non-None value.
                    if fragment.cached_input_tokens is not None:
                        cached_input_tokens = fragment.cached_input_tokens
                    if fragment.cache_creation_input_tokens is not None:
                        cache_creation_input_tokens = fragment.cache_creation_input_tokens

                    if fragment.text is None:
                        continue

                    # update accumulated text
                    accumulated_text += fragment.text

                    # some backends send an empty chunk first skewing the TTFT
                    if accumulated_text and t_first_token is None:
                        t_first_token = now

                    # update logprob tokens
                    if fragment.logprob_tokens:
                        total_logprob_tokens = (total_logprob_tokens or 0) + fragment.logprob_tokens

                # get latency metrics
                now = time.perf_counter()
                dur_total = now - t_start

                if t_first_token is None and not accumulated_text:
                    t_first_token = now

                # get num tokens
                if self.force_recounting_completions:
                    output_tokens = self.count_text_tokens(accumulated_text)
                else:
                    if total_logprob_tokens is not None:
                        output_tokens = total_logprob_tokens
                    else:
                        output_tokens = total_usage_tokens

                    if output_tokens is None:
                        output_tokens = self.count_text_tokens(accumulated_text)

                output_chars = len(accumulated_text)

                if request.stream:
                    dur_generation = now - t_first_token
                else:
                    dur_generation = dur_total

                dur_first_token = t_first_token - t_start

                if output_tokens > 0 and dur_generation > 0:
                    tps = output_tokens / dur_generation
                else:
                    tps = 0.0

                logging.debug(
                    f"Response received: total {(dur_total * 1000):.2f} ms, "
                    f"first token {(dur_first_token * 1000):.2f} ms, "
                    f"{output_chars} chars, {output_tokens} tokens, "
                    f"{tps:.2f} tokens/s"
                )

                # init metrics
                metrics = LatencyProfile()

                # latency per char
                if output_chars and dur_generation > 0:
                    metrics.ms_per_char = dur_generation / output_chars * 1000
                    metrics.char_throughput = 1000 / metrics.ms_per_char
                    metrics.output_char_count = output_chars

                # time to first token
                if request.stream:
                    metrics.first_token_latency = dur_first_token * 1000
                else:
                    metrics.first_token_latency = dur_total * 1000

                # total latency
                metrics.end_to_end_ms = dur_total * 1000

                # ms latency per token and tokens per second
                if output_tokens and output_tokens > 0:
                    metrics.output_token_count = output_tokens
                    if request.stream and dur_generation > 0:
                        metrics.ms_per_token = dur_generation / output_tokens * 1000
                    elif not request.stream and dur_total > 0:
                        metrics.ms_per_token = dur_total / output_tokens * 1000

                    if hasattr(metrics, "ms_per_token") and metrics.ms_per_token > 0:
                        metrics.token_throughput = 1000 / metrics.ms_per_token

                # get prompt tokens
                if not prompt_usage_tokens:
                    metrics.input_token_count = self.count_prompt_tokens(request)
                else:
                    metrics.input_token_count = prompt_usage_tokens

                # set acceptance rate if available
                if accept_ratio is not None:
                    metrics.accept_ratio = accept_ratio

                # set cache stats if backend reported them
                if cached_input_tokens is not None:
                    metrics.cached_input_tokens = cached_input_tokens
                if cache_creation_input_tokens is not None:
                    metrics.cache_creation_input_tokens = cache_creation_input_tokens

                # post-validation
                self.after_request(request, metrics)

                return ResultEntry(
                    model=self.model_name,
                    request=request,
                    content=accumulated_text,
                    metrics=metrics,
                    success=True,
                )

        except Exception as e:
            trace_id = "N/A"
            try:
                if "response" in locals():
                    trace_id = extract_trace_id_from_headers(response.headers)
            except (AttributeError, KeyError, TypeError, NameError):
                # Best-effort trace-id lookup for logging context only;
                # never let it mask the original exception we're handling.
                pass

            logging.warning(
                f"Request failed with exception (request-id: {trace_id}): {e}",
                exc_info=True,
            )
            failed_result.content = str(e)
            return failed_result


class OpenAIBackend(BaseBackend):

    def before_request(self, request: InferencePayload):
        if request.skip_eos:
            raise ValueError("skip_eos is not supported for OpenAI API")

    def build_endpoint_url(self, request: InferencePayload):
        base_url = self.base_url

        if request.messages:
            return os.path.join(base_url, "chat/completions")
        elif request.prompt:
            return os.path.join(base_url, "completions")
        else:
            raise ValueError("Invalid request")

    def build_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def count_prompt_tokens(self, request: InferencePayload) -> int:
        return -1

    def build_request_body(self, request: InferencePayload):
        if request.messages:
            if isinstance(request.messages, str):
                request.messages = json.loads(request.messages)

            assert isinstance(request.messages, list), "Messages should be a list"
        elif request.prompt:
            assert isinstance(request.prompt, str), "Prompt should be a string"
        else:
            raise ValueError("Invalid request")

        data = {
            "model": self.model_name,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "temperature": request.temperature if request.temperature is not None else 0.0,
        }

        if request.prediction is not None:
            data["prediction"] = request.prediction

        if request.n is not None:
            data["n"] = request.n

        if request.top_p is not None:
            data["top_p"] = request.top_p

        if request.reasoning_effort is not None:
            data["reasoning_effort"] = request.reasoning_effort

        if request.reasoning is not None:
            data["reasoning"] = request.reasoning

        if request.messages:
            data["messages"] = request.messages
        elif request.prompt:
            data["prompt"] = request.prompt
        else:
            raise ValueError("Invalid request")

        if request.logprobs is not None:
            data["logprobs"] = request.logprobs

        if request.stream:
            data["stream_options"] = {"include_usage": True}

        return data

    def extract_content(self, choice: Dict, request: InferencePayload) -> str:
        if request.messages:
            if request.stream:
                text = choice["delta"].get("content", "") or choice["delta"].get("reasoning", "")
            else:
                text = choice["message"]["content"] or choice["message"].get("reasoning", "")
        elif request.prompt:
            text = choice["text"]
        else:
            raise ValueError("Invalid request")

        return text

    def decode_response_chunk(self, data: Any, request: InferencePayload) -> Optional[FragmentInfo]:
        usage = data.get("usage", None)

        assert len(data["choices"]) <= 1, f"Too many choices {len(data['choices'])}"

        if len(data["choices"]) == 0:
            text = ""
            logprobs = None
        else:
            choice = data["choices"][0]
            text = self.extract_content(choice, request)

            logprobs = choice.get("logprobs", None)

        if isinstance(logprobs, float):
            logprobs = None

        # OpenAI-style prompt-cache stats live under `usage.prompt_tokens_details`.
        # sglang / vllm OpenAI-compat servers also surface them here when their
        # cache-report flag is on (sglang: --enable-cache-report).
        cached = None
        if usage:
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens")

        return FragmentInfo(
            text=text,
            logprob_tokens=len(logprobs["tokens"]) if logprobs else None,
            usage_tokens=usage["completion_tokens"] if usage else None,
            prompt_usage_tokens=usage.get("prompt_tokens", None) if usage else None,
            cached_input_tokens=cached,
        )


class NvidiaNIMBackend(OpenAIBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        tokenizer_name: str = None,
        force_recounting_completions: bool = False,
    ):
        super().__init__(
            base_url, api_key, model_name, tokenizer_name, force_recounting_completions
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, trust_remote_code=True
            )
        except Exception as e:
            self.tokenizer = None
            logging.warning(f"NvidiaNIMBackend failed to load tokenizer and moving on: {e}")

    def count_prompt_tokens(self, request: InferencePayload) -> int:
        if request.messages:
            assert isinstance(request.messages, list), "Messages should be a list"
            return len(
                self.tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        elif request.prompt:
            assert isinstance(request.prompt, str), "Prompt should be a string"
            return len(self.tokenizer.encode(request.prompt))
        else:
            raise ValueError("Invalid request")

    def count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class VllmBackend(OpenAIBackend):
    def before_request(self, request: InferencePayload):
        pass

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        tokenizer_name: str = None,
        force_recounting_completions: bool = False,
    ):
        super().__init__(
            base_url, api_key, model_name, tokenizer_name, force_recounting_completions
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, trust_remote_code=True
            )
        except Exception as e:
            self.tokenizer = None
            logging.warning(f"VllmBackend failed to load tokenizer and moving on: {e}")

    def build_endpoint_url(self, request: InferencePayload):
        base_url = self.base_url

        if request.messages:
            return os.path.join(base_url, "chat/completions")
        elif request.prompt:
            return os.path.join(base_url, "completions")
        else:
            raise ValueError("Invalid request")

    def after_request(self, request: InferencePayload, metrics: LatencyProfile):
        if request.skip_eos:
            if metrics.output_token_count != request.max_tokens:
                raise ValueError(
                    "output_token_count does not match max_tokens when skip_eos is set."
                    "Please check the backend's support for skip_eos."
                )

    def count_prompt_tokens(self, request: InferencePayload) -> int:
        if request.messages:
            assert isinstance(request.messages, list), "Messages should be a list"
            return len(
                self.tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        elif request.prompt:
            assert isinstance(request.prompt, str), "Prompt should be a string"
            return len(self.tokenizer.encode(request.prompt))
        else:
            raise ValueError("Invalid request")

    def count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def build_request_body(self, request: InferencePayload):
        data = super().build_request_body(request)
        if request.skip_eos:
            data["ignore_eos"] = True
        return data


class SGLangBackend(VllmBackend):
    def extract_content(self, choice: Dict, request: InferencePayload) -> str:
        if request.messages:
            if request.stream:
                text = choice["delta"].get("content", "") or choice["delta"].get(
                    "reasoning_content", ""
                )
            else:
                text = choice["message"].get("content", "") or choice["message"].get(
                    "reasoning_content", ""
                )
        elif request.prompt:
            text = choice["text"]
        else:
            raise ValueError("Invalid request")
        return text

    def build_request_body(self, request: InferencePayload):
        data = super().build_request_body(request)

        if request.adapter_path:
            data["adapter_path"] = request.adapter_path
        elif request.lora_name:
            data["model"] = request.lora_name
            data["lora_path"] = request.lora_name

        return data


class FireworksBackend(VllmBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        tokenizer_name: str = None,
        force_recounting_completions: bool = False,
    ):
        if not api_key:
            api_key = os.environ["FIREWORKS_API_KEY"]

        super().__init__(
            base_url, api_key, model_name, tokenizer_name, force_recounting_completions
        )

    def build_request_body(self, request: InferencePayload):
        data = super().build_request_body(request)
        if request.skip_eos:
            data["ignore_eos"] = True
        return data


class TogetherBackend(VllmBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        tokenizer_name: str = None,
        force_recounting_completions: bool = False,
    ):
        if not api_key:
            api_key = os.environ["TOGETHER_API_KEY"]
        super().__init__(
            base_url, api_key, model_name, tokenizer_name, force_recounting_completions
        )

    def build_endpoint_url(self, request: InferencePayload):
        return super().build_endpoint_url(request)

    def build_request_body(self, request: InferencePayload):
        data = super().build_request_body(request)

        if request.adapter_path:
            data["adapter_path"] = request.adapter_path
        elif request.lora_name:
            data["lora_name"] = request.lora_name

        return data


class AnthropicBackend(BaseBackend):
    """
    Anthropic Messages API (https://docs.anthropic.com/en/api/messages).

    Differences from OpenAI backend:
      - endpoint: POST /v1/messages
      - body uses `system` as a top-level field (string OR list of content blocks)
        rather than a system-role message inside `messages`
      - SSE event names differ; usage info arrives in `message_start` (input,
        cache_*) and `message_delta` (output_tokens)
      - prompt caching is opt-in via `cache_control: {type: "ephemeral"}` on a
        content block. When `prompt_caching=True` on the request, this backend
        wraps the system text (or first user content) with that marker so the
        prefix becomes cacheable across requests.

    Required header set: `x-api-key`, `anthropic-version`. The version pin
    `2023-06-01` is the long-standing GA version that supports cache_control.
    """

    ANTHROPIC_API_VERSION = "2023-06-01"

    # Sonnet/Haiku cacheable minimum is 1024 tokens; Opus needs 2048.
    # The 1h beta TTL header is "anthropic-beta: prompt-caching-2024-07-31".
    MIN_CACHEABLE_TOKENS_DEFAULT = 1024
    MIN_CACHEABLE_TOKENS_OPUS = 2048
    EXTENDED_TTL_BETA_HEADER = "prompt-caching-2024-07-31"
    # Maximum cache_control breakpoints the API accepts per request.
    MAX_CACHE_BREAKPOINTS = 4

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        force_recounting_completions: bool = False,
        enable_prompt_caching: bool = True,
        enable_extended_cache_ttl: bool = False,
    ):
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        super().__init__(
            base_url or "https://api.anthropic.com",
            api_key,
            model_name,
            tokenizer_name,
            force_recounting_completions,
        )
        # When True, mark long content blocks with cache_control so the
        # Messages API populates cache_read_input_tokens on repeat hits.
        self.enable_prompt_caching = enable_prompt_caching
        # Opt-in: 1-hour cache TTL (vs default 5 min). Adds the beta header.
        self.enable_extended_cache_ttl = enable_extended_cache_ttl

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, trust_remote_code=True
            )
        except Exception as e:
            self.tokenizer = None
            logging.warning(f"AnthropicBackend tokenizer load skipped: {e}")

    @property
    def min_cacheable_tokens(self) -> int:
        """Per-block minimum size below which Anthropic ignores cache_control."""
        # Opus models need 2048; everything else 1024. Match on the substring
        # in case the user passed a versioned name like "claude-3-opus-20240229".
        if self.model_name and "opus" in self.model_name.lower():
            return self.MIN_CACHEABLE_TOKENS_OPUS
        return self.MIN_CACHEABLE_TOKENS_DEFAULT

    def build_endpoint_url(self, request: InferencePayload) -> str:
        return os.path.join(self.base_url.rstrip("/"), "v1/messages")

    def build_headers(self) -> Dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }
        if self.enable_extended_cache_ttl:
            # Opt into 1-hour cache TTL (default is 5 minutes).
            headers["anthropic-beta"] = self.EXTENDED_TTL_BETA_HEADER
        return headers

    def count_prompt_tokens(self, request: InferencePayload) -> int:
        if not self.tokenizer:
            return -1
        if request.messages:
            return len(self.tokenizer.apply_chat_template(
                request.messages, tokenize=True, add_generation_prompt=True
            ))
        if request.prompt:
            return len(self.tokenizer.encode(request.prompt))
        return -1

    def count_text_tokens(self, text: str) -> int:
        if not self.tokenizer:
            return -1
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _split_prompt_blocks(self, prompt: Optional[str], cacheable_prefix: Optional[str]):
        """
        Build a list of content blocks for a `prompt` payload. When the prompt
        starts with a known cacheable prefix, split into [prefix, suffix] so
        Anthropic's cache_control can sit on the prefix block.
        """
        prompt = prompt or ""
        if cacheable_prefix and prompt.startswith(cacheable_prefix):
            suffix = prompt[len(cacheable_prefix):]
            if suffix:
                return [
                    {"type": "text", "text": cacheable_prefix},
                    {"type": "text", "text": suffix},
                ]
            # Whole prompt is the cacheable prefix.
            return [{"type": "text", "text": cacheable_prefix}]
        return [{"type": "text", "text": prompt}]

    def _maybe_wrap_cache_control(self, blocks):
        """
        Attach `cache_control: {type: "ephemeral"}` to the LAST text block in
        the list, which Anthropic uses as the cache boundary (the cumulative
        prefix up to and including this block becomes the cacheable chunk).
        No-op when prompt caching is disabled.
        """
        if not self.enable_prompt_caching or not blocks:
            return blocks
        # Find rightmost text block to mark; leave others untouched.
        for block in reversed(blocks):
            if isinstance(block, dict) and block.get("type") == "text":
                block["cache_control"] = {"type": "ephemeral"}
                break
        return blocks

    def build_request_body(self, request: InferencePayload) -> Dict:
        if not request.messages and not request.prompt:
            raise ValueError("AnthropicBackend requires messages or prompt")

        system_blocks = []
        message_list = []

        if request.messages:
            msgs = (
                json.loads(request.messages)
                if isinstance(request.messages, str)
                else list(request.messages)
            )
            # Anthropic puts system messages in a separate `system` field, not
            # inside `messages`. Hoist any role=system entries out.
            for m in msgs:
                if m.get("role") == "system":
                    text = m.get("content")
                    if isinstance(text, str):
                        system_blocks.append({"type": "text", "text": text})
                    elif isinstance(text, list):
                        system_blocks.extend(text)
                else:
                    # Normalize content -> list of blocks so we can mark caches.
                    content = m.get("content")
                    if isinstance(content, str):
                        m = {**m, "content": [{"type": "text", "text": content}]}
                    message_list.append(m)
        else:
            # `prompt` form -> single user message. If a cacheable_prefix is
            # known, split the prompt into [prefix_block, suffix_block] so the
            # cache_control marker can sit on the prefix block alone — only
            # the prefix needs to match across requests for a cache hit. Without
            # the split, Anthropic's cache key includes the (varying) suffix
            # and we get 0% hits.
            content_blocks = self._split_prompt_blocks(
                request.prompt, request.cacheable_prefix
            )
            message_list.append({
                "role": "user",
                "content": content_blocks,
            })

        # Cache-control on the longest cacheable chunk:
        #   - prefer system blocks if present (most stable across requests)
        #   - otherwise: if the user message was already split with an explicit
        #     prefix block, mark that block (not the trailing suffix); else
        #     fall back to marking the rightmost text block (covers the
        #     same-prompts-in-burst case where the whole prompt matches).
        if system_blocks:
            system_blocks = self._maybe_wrap_cache_control(system_blocks)
        elif message_list:
            last = message_list[-1]
            if isinstance(last.get("content"), list):
                if self.enable_prompt_caching and len(last["content"]) >= 2 and request.cacheable_prefix:
                    # Mark the FIRST block (prefix); leave the suffix unmarked.
                    last["content"][0]["cache_control"] = {"type": "ephemeral"}
                else:
                    last["content"] = self._maybe_wrap_cache_control(last["content"])

        body = {
            "model": self.model_name,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "messages": message_list,
            "temperature": request.temperature if request.temperature is not None else 0.0,
        }
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if system_blocks:
            body["system"] = system_blocks
        # Anthropic doesn't accept skip_eos / ignore_eos — model decides EOS.
        return body

    def decode_response_chunk(self, data: Any, request: InferencePayload) -> Optional[FragmentInfo]:
        """
        Anthropic SSE stream event types we care about:
          message_start         -> data.message.usage has input + cache_* tokens
          content_block_delta   -> data.delta.text is the next token chunk
          message_delta         -> data.usage.output_tokens updates running count
          message_stop          -> end of stream
        Non-streaming responses just return the whole message at once.
        """
        if isinstance(data, dict) and "error" in data:
            logging.warning(f"Anthropic error: {data['error']}")
            return None

        ev_type = data.get("type")

        # Streaming forms.
        if ev_type == "message_start":
            usage = (data.get("message") or {}).get("usage", {}) or {}
            return FragmentInfo(
                text="",  # no text yet
                logprob_tokens=None,
                usage_tokens=usage.get("output_tokens"),
                prompt_usage_tokens=usage.get("input_tokens"),
                cached_input_tokens=usage.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            )

        if ev_type == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") in ("text_delta", "input_json_delta"):
                return FragmentInfo(
                    text=delta.get("text") or delta.get("partial_json") or "",
                    logprob_tokens=None,
                    usage_tokens=None,
                    prompt_usage_tokens=None,
                )
            # Non-text deltas (tool use, etc.) — skip without failing.
            return FragmentInfo(text="", logprob_tokens=None,
                                usage_tokens=None, prompt_usage_tokens=None)

        if ev_type == "message_delta":
            usage = data.get("usage", {}) or {}
            return FragmentInfo(
                text="",
                logprob_tokens=None,
                usage_tokens=usage.get("output_tokens"),
                prompt_usage_tokens=None,
            )

        if ev_type in ("content_block_start", "content_block_stop", "ping", "message_stop"):
            return FragmentInfo(text="", logprob_tokens=None,
                                usage_tokens=None, prompt_usage_tokens=None)

        # Non-streaming response: top-level usage + content blocks.
        if "content" in data:
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text") or ""
            usage = data.get("usage", {}) or {}
            return FragmentInfo(
                text=text,
                logprob_tokens=None,
                usage_tokens=usage.get("output_tokens"),
                prompt_usage_tokens=usage.get("input_tokens"),
                cached_input_tokens=usage.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            )

        # Unknown event — emit empty to keep stream loop going.
        return FragmentInfo(text="", logprob_tokens=None,
                            usage_tokens=None, prompt_usage_tokens=None)


class TRTLLMBackend(SGLangBackend):
    """
    Backend for TensorRT-LLM that extracts acceptance rate from API responses.

    TRT-LLM returns avg_decoded_tokens_per_iter in the usage field of the response,
    which indicates the average number of tokens accepted per decoding iteration
    (relevant for speculative decoding).
    """

    def decode_response_chunk(self, data: Any, request: InferencePayload) -> Optional[FragmentInfo]:
        chunk_metadata = super().decode_response_chunk(data, request)

        choices = data.get("choices", [])
        if len(choices) > 0:
            choice = choices[0]
            if "avg_decoded_tokens_per_iter" in choice:
                ratio_value = choice["avg_decoded_tokens_per_iter"]
                logging.debug(f"TRT-LLM found avg_decoded_tokens_per_iter: {ratio_value}")
                chunk_metadata.accept_ratio = ratio_value

        return chunk_metadata


class TgiBackend(BaseBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        tokenizer_name: str = None,
        force_recounting_completions: bool = False,
    ):
        super().__init__(
            base_url, api_key, model_name, tokenizer_name, force_recounting_completions
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, trust_remote_code=True
            )
        except Exception as e:
            self.tokenizer = None
            print(f"Failed to initialize tokenizer: {e}")

    def build_endpoint_url(self, request: InferencePayload):
        if request.stream:
            return os.path.join(self.base_url, "generate_stream")
        else:
            return os.path.join(self.base_url, "generate")

    def build_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def before_request(self, request: InferencePayload):
        pass

    def after_request(self, request: InferencePayload, metrics: LatencyProfile):
        if request.skip_eos:
            if metrics.output_token_count != request.max_tokens:
                raise ValueError(
                    "output_token_count does not match max_tokens when skip_eos is set."
                    "Please check the backend's support for skip_eos."
                )

    def count_prompt_tokens(self, request: InferencePayload) -> int:
        if request.messages:
            assert isinstance(request.messages, list), "Messages should be a list"
            return len(
                self.tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        elif request.prompt:
            assert isinstance(request.prompt, str), "Prompt should be a string"
            return len(self.tokenizer.encode(request.prompt))
        else:
            raise ValueError("Invalid request")

    def count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def build_request_body(self, request: InferencePayload):
        if request.messages:
            if isinstance(request.messages, str):
                request.messages = json.loads(request.messages)

            assert isinstance(request.messages, list), "Messages should be a list"
            if not self.tokenizer:
                raise ValueError("For TGI, tokenizer_name is required for chat mode")

            prompt = self.tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif request.prompt:
            assert isinstance(request.prompt, str), "Prompt should be a string"
            prompt = request.prompt
        else:
            raise ValueError("Invalid request")

        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_n_tokens": request.top_logprobs,
                "details": request.logprobs is not None,
            },
        }

        if request.prediction is not None:
            data["parameters"]["prediction"] = request.prediction

        if request.top_p is not None:
            data["parameters"]["top_p"] = request.top_p
        return data

    def decode_response_chunk(self, data: Any, request: InferencePayload) -> Optional[FragmentInfo]:
        if "error" in data:
            return None

        if "token" in data:
            return FragmentInfo(
                text=data["token"]["text"],
                logprob_tokens=1,
                usage_tokens=None,
                prompt_usage_tokens=None,
            )
        else:
            return FragmentInfo(
                text=data["generated_text"],
                logprob_tokens=(len(data["details"]["tokens"]) if "details" in data else None),
                usage_tokens=(data["details"]["generated_tokens"] if "details" in data else None),
                prompt_usage_tokens=None,
            )


class OpenAIVectorBackend(BaseBackend):
    """
    OpenAI-compatible embeddings backend.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        tokenizer_name: str = None,
        force_recounting_completions: bool = False,
    ):
        super().__init__(
            base_url, api_key, model_name, tokenizer_name, force_recounting_completions
        )
        if not self.base_url.endswith("v1/embeddings"):
            raise ValueError("Base URL must end with /v1/embeddings")

    def build_endpoint_url(self, request: InferencePayload):
        return self.base_url

    def build_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def count_prompt_tokens(self, request: InferencePayload) -> int:
        return self.tokenizer.encode(request.prompt)

    def build_request_body(self, request: InferencePayload):
        data = {
            "input": request.prompt,
            "model": self.model_name,
        }

        return data

    def decode_response_chunk(self, data: Any, request: InferencePayload) -> Optional[FragmentInfo]:
        if "error" in data:
            logging.error(f"API returned error: {data['error']}")
            return None

        if "data" not in data:
            logging.error("Response missing 'data' field")
            return None

        embeddings_data = data["data"]
        if not embeddings_data:
            logging.error("No embeddings in response")
            return None

        usage_tokens = None
        prompt_usage_tokens = None
        if "usage" in data:
            usage_tokens = data["usage"].get("total_tokens", None)
            prompt_usage_tokens = data["usage"].get("prompt_tokens", None)

        return FragmentInfo(
            text="",
            logprob_tokens=None,
            usage_tokens=usage_tokens,
            prompt_usage_tokens=prompt_usage_tokens,
        )

    async def execute_call(self, request: InferencePayload) -> ResultEntry:
        self.before_request(request)

        payload = self.build_request_body(request)
        logging.debug(payload)

        async with aiohttp.ClientSession(
            headers=self.build_headers(),
            timeout=HTTP_SESSION_TIMEOUT,
            read_bufsize=HTTP_READ_BUFFER,
        ) as session:
            failed_result = ResultEntry(success=False)
            t_start = time.perf_counter()

            try:
                async with session.post(self.build_endpoint_url(request), json=payload) as response:
                    if response.status != 200:
                        error_bytes = b""
                        async for chunk_bytes in response.content:
                            error_bytes += chunk_bytes
                        error_text = error_bytes.decode("utf-8")
                        failed_result.content = error_text

                        trace_id = extract_trace_id_from_headers(response.headers)
                        logging.warning(
                            f"Embeddings request failed with status {response.status}, "
                            f"request-id: {trace_id}"
                        )
                        logging.warning(error_text)
                        return failed_result

                    response_bytes = await response.read()
                    data = orjson.loads(response_bytes)

                    chunk_metadata = self.decode_response_chunk(data=data, request=request)
                    if chunk_metadata is None:
                        return failed_result

                    now = time.perf_counter()
                    total_latency = (now - t_start) * 1000

                    metrics = LatencyProfile()
                    metrics.end_to_end_ms = total_latency
                    metrics.first_token_latency = -1
                    metrics.ms_per_token = -1
                    metrics.input_token_count = (
                        chunk_metadata.prompt_usage_tokens or self.count_prompt_tokens(request)
                    )

                    embeddings_data = json.loads(chunk_metadata.text) if chunk_metadata.text else []
                    embedding_count = len(embeddings_data)

                    metrics.output_char_count = -1
                    metrics.output_token_count = -1

                    logging.debug(
                        f"Embeddings response received: total {total_latency:.2f} ms, "
                        f"prompt tokens: {metrics.input_token_count}, "
                        f"embeddings: {embedding_count}, "
                        f"dimensions: {metrics.output_token_count}"
                    )

                    self.after_request(request, metrics)

                    return ResultEntry(
                        model=self.model_name,
                        request=request,
                        content=chunk_metadata.text,
                        metrics=metrics,
                        success=True,
                    )

            except Exception as e:
                trace_id = "N/A"
                try:
                    if "response" in locals():
                        trace_id = extract_trace_id_from_headers(response.headers)
                except (AttributeError, KeyError, TypeError, NameError):
                    # Best-effort trace-id lookup for logging context only;
                    # never let it mask the original exception we're handling.
                    pass

                logging.warning(
                    f"Embeddings request failed with exception (request-id: {trace_id}): {e}",
                    exc_info=True,
                )
                failed_result.content = str(e)
                return failed_result
