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
                        else:
                            prompt_usage_tokens = 0

                        # capture acceptance rate from chunk metadata if available
                        if fragment.accept_ratio is not None:
                            accept_ratio = fragment.accept_ratio

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
                except Exception:
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

        return FragmentInfo(
            text=text,
            logprob_tokens=len(logprobs["tokens"]) if logprobs else None,
            usage_tokens=usage["completion_tokens"] if usage else None,
            prompt_usage_tokens=usage.get("prompt_tokens", None) if usage else None,
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
                except Exception:
                    pass

                logging.warning(
                    f"Embeddings request failed with exception (request-id: {trace_id}): {e}",
                    exc_info=True,
                )
                failed_result.content = str(e)
                return failed_result
