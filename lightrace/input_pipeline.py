import json
import os
import random
from time import time
from typing import List, Optional

import datasets
import numpy as np
from lightrace._sglang_compat import get_tokenizer, sample_random_requests
from together import Together
from tqdm import tqdm

from lightrace.schema import InferencePayload

FILLER_TOKEN_PHRASE: str = "hi,"
FILLER_TOKEN_COUNT: int = 2

# Default cacheable dataset (can be overridden by user args)
PREFIX_CACHE_DATASET = "emozilla/pg19"
PREFIX_CACHE_COLUMN = "text"

# Default non-cacheable dataset for generated-shared-prefix.
# Must be publicly accessible; the previous default was a private togethercomputer
# org dataset that failed on fresh installs with DatasetNotFoundError.
DEFAULT_SUFFIX_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_SUFFIX_DATASET_COLUMN = "problem"
DEFAULT_SUFFIX_DATASET_SPLIT = "test"


def replicate_inputs_per_batch(
    requests: List[InferencePayload], concurrency: int
) -> List[InferencePayload]:
    output_requests = []
    total_examples = len(requests)
    assert total_examples % concurrency == 0, "Total examples must be divisible by concurrency"

    num_batches = total_examples // concurrency
    for i in range(num_batches):
        output_requests += [requests[i]] * concurrency

    return output_requests


class InputPipeline:
    def __init__(
        self,
        *,
        model_name: str,
        dataset_type: str,
        stream: bool,
        max_tokens: int,
        skip_eos: bool,
        temperature: float,
        top_p: float,
        chat: bool,
        reasoning_effort: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
        num_examples: Optional[int] = None,
        tokenizer_name: Optional[str] = None,
        hf_dataset: Optional[str] = None,
        hf_dataset_split: Optional[str] = None,
        hf_dataset_config: Optional[str] = None,
        hf_dataset_column_name: Optional[str] = None,
        jsonl_input_path: Optional[str] = None,
        jsonl_dataset_column_name: Optional[str] = None,
        jsonl_convert_to_chat_request_format: Optional[bool] = None,
        jsonl_r2_file_ids: Optional[str] = None,
        jsonl_r2_download_dir: Optional[str] = None,
        together_api_key: Optional[str] = None,
        synthetic_input_length: Optional[int] = None,
        synthetic_output_length: Optional[int] = None,
        synthetic_input_length_stdev: Optional[int] = None,
        synthetic_output_length_stdev: Optional[int] = None,
        synthetic_cached_input_length: Optional[int] = None,
        same_prompts_in_burst: Optional[bool] = None,
        concurrency: Optional[int] = None,
        use_temperature_top_p_from_jsonl_args: Optional[bool] = None,
        gsp_cached_fraction: Optional[float] = None,
        gsp_hf_dataset: Optional[str] = None,
        gsp_hf_dataset_split: Optional[str] = None,
        gsp_hf_dataset_config: Optional[str] = None,
        gsp_cacheable_dataset: Optional[str] = None,
        gsp_cacheable_dataset_split: Optional[str] = None,
        gsp_cacheable_dataset_config: Optional[str] = None,
        gsp_hf_dataset_column_name: Optional[str] = None,
        gsp_cacheable_dataset_column_name: Optional[str] = None,
        gsp_groups: Optional[int] = None,
        traffic_pattern: Optional[str] = None,
        qps_duration_s: Optional[float] = None,
        adapter_paths: Optional[str] = None,
        lora_names: Optional[str] = None,
        lora_ratio: Optional[float] = None,
        lora_distribution: str = "round_robin",
    ):
        if tokenizer_name:
            self.tokenizer_name = tokenizer_name
        else:
            self.tokenizer_name = model_name

        self._tokenizer = None
        self.dataset_type = dataset_type
        self.hf_dataset = hf_dataset
        self.hf_dataset_split = hf_dataset_split
        # Treat "" and "None" like missing — same convention as gsp_*_config.
        self.hf_dataset_config = (
            None if hf_dataset_config in (None, "", "None") else hf_dataset_config
        )
        self.hf_dataset_column_name = hf_dataset_column_name
        self.jsonl_input_path = jsonl_input_path
        self.jsonl_dataset_column_name = jsonl_dataset_column_name
        self.jsonl_convert_to_chat_request_format = jsonl_convert_to_chat_request_format
        self.jsonl_r2_file_ids = jsonl_r2_file_ids
        self.jsonl_r2_download_dir = jsonl_r2_download_dir
        self.together_api_key = together_api_key
        self.synthetic_input_length = synthetic_input_length
        self.synthetic_output_length = synthetic_output_length
        self.synthetic_input_length_stdev = synthetic_input_length_stdev
        self.synthetic_output_length_stdev = synthetic_output_length_stdev
        self.synthetic_cached_input_length = synthetic_cached_input_length or 0
        self.num_examples = num_examples
        self.stream = stream
        self.max_tokens = max_tokens
        self.skip_eos = skip_eos
        self.temperature = temperature
        self.top_p = top_p
        self.chat = chat
        self.reasoning_effort = reasoning_effort
        self.reasoning = {"enabled": True} if reasoning_enabled else None
        self.same_prompts_in_burst = same_prompts_in_burst
        self.concurrency = concurrency
        self.use_temperature_top_p_from_jsonl_args = use_temperature_top_p_from_jsonl_args
        self.gsp_cached_fraction = gsp_cached_fraction
        self.gsp_hf_dataset = gsp_hf_dataset or DEFAULT_SUFFIX_DATASET
        self.gsp_hf_dataset_split = gsp_hf_dataset_split or DEFAULT_SUFFIX_DATASET_SPLIT
        # Treat the literal strings "None" and "" as "no config" so users can
        # explicitly opt out via CLI without jsonargparse rejecting the empty
        # string. None / "default" are also fine.
        self.gsp_hf_dataset_config = (
            None
            if gsp_hf_dataset_config in (None, "", "None")
            else gsp_hf_dataset_config
        )
        self.gsp_cacheable_dataset = gsp_cacheable_dataset or PREFIX_CACHE_DATASET
        self.gsp_cacheable_dataset_split = gsp_cacheable_dataset_split or "train"
        self.gsp_cacheable_dataset_config = (
            None
            if gsp_cacheable_dataset_config in (None, "", "None")
            else gsp_cacheable_dataset_config
        )
        self.gsp_hf_dataset_column_name = (
            gsp_hf_dataset_column_name or DEFAULT_SUFFIX_DATASET_COLUMN
        )
        self.gsp_cacheable_dataset_column_name = gsp_cacheable_dataset_column_name or "text"
        self.gsp_groups = gsp_groups
        self.traffic_pattern = traffic_pattern
        self.qps_duration_s = qps_duration_s

        # LoRA-related parameters
        self.adapter_paths = adapter_paths.split(",") if adapter_paths else None
        self.lora_names = lora_names.split(",") if lora_names else None
        self.lora_ratio = lora_ratio if lora_ratio is not None else 0.0
        self.lora_distribution = lora_distribution
        self.lora_index = 0

    @property
    def tokenizer(self):
        """Lazy loading of tokenizer to avoid unnecessary downloads."""
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer(self.tokenizer_name)
        return self._tokenizer

    def _pick_adapter_for_request(self, request_idx: int) -> tuple[Optional[str], Optional[str]]:
        """
        Determine the LoRA adapter for a given request based on LoRA configuration.

        Returns:
            Tuple of (adapter_path, lora_name) - one will be set, the other None
        """
        if self.adapter_paths:
            lora_list = self.adapter_paths
            use_names = False
        elif self.lora_names:
            lora_list = self.lora_names
            use_names = True
        else:
            return (None, None)

        if random.random() >= self.lora_ratio:
            return (None, None)

        if self.lora_distribution == "round_robin":
            adapter_idx = self.lora_index % len(lora_list)
            adapter = lora_list[adapter_idx]
            self.lora_index += 1

        elif self.lora_distribution == "random":
            adapter = random.choice(lora_list)

        elif self.lora_distribution == "uniform":
            adapter_idx = request_idx % len(lora_list)
            adapter = lora_list[adapter_idx]

        else:
            adapter = lora_list[0]

        if use_names:
            return (None, adapter)
        else:
            return (adapter, None)

    def prepare_inputs(self) -> List[InferencePayload]:
        if self.dataset_type == "hf":
            all_requests = self._build_hf_inputs()
        elif self.dataset_type == "jsonl":
            all_requests = self._build_jsonl_inputs()
        elif self.dataset_type == "synthetic":
            all_requests = self._build_synthetic_inputs()
        elif self.dataset_type == "sharegpt":
            all_requests = self._build_sharegpt_inputs()
        elif self.dataset_type == "generated-shared-prefix":
            all_requests = self._build_shared_prefix_inputs()
        else:
            raise ValueError("No dataset specified")

        if self.same_prompts_in_burst and self.dataset_type != "generated-shared-prefix":
            all_requests = replicate_inputs_per_batch(all_requests, self.concurrency)

        assert (
            len(all_requests) == self.num_examples
        ), f"Processed number of examples {len(all_requests)} != "
        f"args.num_examples {self.num_examples}"

        return all_requests

    def _build_hf_inputs(self) -> List[InferencePayload]:

        load_kwargs = {"split": self.hf_dataset_split}
        if self.hf_dataset_config is not None:
            load_kwargs["name"] = self.hf_dataset_config
        dataset = datasets.load_dataset(self.hf_dataset, **load_kwargs)

        if self.num_examples:
            if self.num_examples > len(dataset):
                additional_datasets = [dataset] * (self.num_examples // len(dataset))
                remaining_examples_needed = self.num_examples % len(dataset)
                if remaining_examples_needed > 0:
                    additional_datasets.append(dataset.select(range(remaining_examples_needed)))
                dataset = datasets.concatenate_datasets([dataset] + additional_datasets)
            if len(dataset) > self.num_examples:
                dataset = dataset.select(range(self.num_examples))

        all_requests = []
        for idx, example in enumerate(dataset):
            request_kwargs = {}

            if self.chat:
                raw = example[self.hf_dataset_column_name]
                # Accept three shapes: already-a-list (good), JSON-encoded list
                # (decode), or raw plain text (auto-wrap as a user turn).
                if isinstance(raw, list):
                    request_kwargs["messages"] = raw
                elif isinstance(raw, str):
                    stripped = raw.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        try:
                            decoded = json.loads(raw)
                            if isinstance(decoded, list):
                                request_kwargs["messages"] = decoded
                            else:
                                request_kwargs["messages"] = [{"role": "user", "content": raw}]
                        except json.JSONDecodeError:
                            request_kwargs["messages"] = [{"role": "user", "content": raw}]
                    else:
                        # Plain-text column: wrap as a single user message
                        # instead of crashing on json.loads.
                        request_kwargs["messages"] = [{"role": "user", "content": raw}]
                else:
                    raise TypeError(
                        f"hf column '{self.hf_dataset_column_name}' has unsupported type "
                        f"{type(raw).__name__}; expected list, str, or JSON-encoded list."
                    )
            else:
                request_kwargs["prompt"] = example[self.hf_dataset_column_name]

            if "prediction" in example:
                request_kwargs["prediction"] = example["prediction"]
                if isinstance(request_kwargs["prediction"], str):
                    request_kwargs["prediction"] = json.loads(request_kwargs["prediction"])

            request_kwargs["stream"] = self.stream
            request_kwargs["max_tokens"] = self.max_tokens
            request_kwargs["skip_eos"] = self.skip_eos
            request_kwargs["temperature"] = self.temperature
            request_kwargs["top_p"] = self.top_p
            request_kwargs["reasoning_effort"] = self.reasoning_effort
            request_kwargs["reasoning"] = self.reasoning

            adapter_path, lora_name = self._pick_adapter_for_request(idx)
            request_kwargs["adapter_path"] = adapter_path
            request_kwargs["lora_name"] = lora_name

            request = InferencePayload(**request_kwargs)
            all_requests.append(request)

        return all_requests

    def _build_sharegpt_inputs(self) -> List[InferencePayload]:

        input_requests = sample_random_requests(
            input_len=self.synthetic_input_length,
            output_len=self.synthetic_output_length,
            num_prompts=self.num_examples,
            range_ratio=1.0,
            tokenizer=self.tokenizer,
            dataset_path="",
            random_sample=True,
            return_text=True,
        )

        prompts = [row.prompt for row in input_requests[:]]

        requests = []
        for idx, prompt in enumerate(prompts):
            adapter_path, lora_name = self._pick_adapter_for_request(idx)
            requests.append(InferencePayload(
                prompt=prompt,
                stream=self.stream,
                max_tokens=self.synthetic_output_length,
                skip_eos=self.skip_eos,
                temperature=self.temperature,
                top_p=self.top_p,
                adapter_path=adapter_path,
                lora_name=lora_name,
            ))
        return requests

    def _build_jsonl_inputs(self) -> List[InferencePayload]:

        if self.jsonl_r2_file_ids:
            if not self.jsonl_r2_download_dir:
                raise ValueError(
                    "`jsonl_r2_download_dir` must be set if `jsonl_r2_file_ids` is set."
                )

            client = Together(api_key=self.together_api_key)
            file_ids = self.jsonl_r2_file_ids.split(",")
            dataset = []
            for file_id in file_ids:
                download_file_path = os.path.join(
                    self.jsonl_r2_download_dir,
                    f"r2_validation_file_{file_id}.jsonl",
                )
                client.files.retrieve_content(file_id, output=download_file_path)
                with open(download_file_path, "r") as f:
                    dataset.extend([json.loads(line) for line in f])

        elif self.jsonl_input_path:
            with open(self.jsonl_input_path) as f:
                dataset = [json.loads(line) for line in f]

        else:
            raise ValueError("Either `jsonl_r2_file_ids` or `jsonl_input_path` must be set.")

        if self.use_temperature_top_p_from_jsonl_args:
            filtered_texts = []
            for text in dataset:
                is_valid = True
                if text.get("args") is not None:
                    text_args = json.loads(text.get("args"))
                    temperature_value = text_args.get("temperature", self.temperature)
                    if temperature_value is not None and temperature_value <= 0:
                        is_valid = False
                    text["temperature"] = (
                        float(temperature_value)
                        if temperature_value is not None
                        else self.temperature
                    )

                    top_p_value = text_args.get("top_p", self.top_p)
                    if top_p_value is not None and (top_p_value <= 0 or top_p_value >= 1):
                        is_valid = False
                    text["top_p"] = float(top_p_value) if top_p_value is not None else self.top_p
                else:
                    text["temperature"] = self.temperature
                    text["top_p"] = self.top_p

                if is_valid:
                    filtered_texts.append(text)

            dataset = filtered_texts

        if self.num_examples > len(dataset):
            indices = np.random.choice(len(dataset), size=self.num_examples, replace=True)
            dataset = [dataset[i] for i in indices]
        else:
            indices = np.random.choice(len(dataset), size=self.num_examples, replace=False)
            dataset = [dataset[i] for i in indices]

        all_requests = []
        for idx, example in enumerate(dataset):
            request_kwargs = {}

            if self.chat:
                if self.jsonl_convert_to_chat_request_format:
                    request_kwargs["messages"] = [
                        {"role": "user", "content": example[self.jsonl_dataset_column_name]}
                    ]
                else:
                    request_kwargs["messages"] = example[self.jsonl_dataset_column_name]

                if isinstance(request_kwargs["messages"], str):
                    request_kwargs["messages"] = json.loads(request_kwargs["messages"])
            else:
                request_kwargs["prompt"] = example[self.jsonl_dataset_column_name]

            request_kwargs["stream"] = self.stream
            request_kwargs["max_tokens"] = self.max_tokens
            request_kwargs["skip_eos"] = self.skip_eos
            request_kwargs["temperature"] = (
                example["temperature"] if "temperature" in example else self.temperature
            )
            request_kwargs["top_p"] = example["top_p"] if "top_p" in example else self.top_p
            request_kwargs["reasoning_effort"] = (
                example["reasoning_effort"]
                if "reasoning_effort" in example
                else self.reasoning_effort
            )
            request_kwargs["reasoning"] = (
                example["reasoning"] if "reasoning" in example else self.reasoning
            )

            adapter_path, lora_name = self._pick_adapter_for_request(idx)
            request_kwargs["adapter_path"] = adapter_path
            request_kwargs["lora_name"] = lora_name

            request = InferencePayload(**request_kwargs)
            all_requests.append(request)

        return all_requests

    def _build_synthetic_inputs(self) -> List[InferencePayload]:
        assert (
            self.num_examples is not None and self.num_examples > 0
        ), "--num_examples needs to be set to use synthetic prompts for benchmarking."
        assert (
            self.synthetic_input_length is not None and self.synthetic_output_length is not None
        ), "--synthetic_input_length and --synthetic_output_length has to be set when "
        "--dataset_type=synthetic."

        def build_prompt(index: int) -> str:

            synthetic_input_length = int(
                self.synthetic_input_length / FILLER_TOKEN_COUNT
            ) + int(
                (
                    0
                    if self.synthetic_input_length_stdev is None
                    else random.randint(
                        -self.synthetic_input_length_stdev, self.synthetic_input_length_stdev
                    )
                    / FILLER_TOKEN_COUNT
                )
            )

            prefix_length = min(
                self.synthetic_cached_input_length // FILLER_TOKEN_COUNT,
                synthetic_input_length,
            )

            suffix_length = synthetic_input_length - prefix_length

            return (
                FILLER_TOKEN_PHRASE * prefix_length
                + str(index)
                + str(time())[-6:]
                + FILLER_TOKEN_PHRASE * suffix_length
                + "Write a very long essay about San Francisco"
            )

        prompts = [build_prompt(index) for index in range(self.num_examples)]

        requests = []
        for idx, prompt in enumerate(prompts):
            adapter_path, lora_name = self._pick_adapter_for_request(idx)
            requests.append(InferencePayload(
                prompt=prompt,
                stream=self.stream,
                max_tokens=(
                    self.synthetic_output_length
                    if self.synthetic_output_length_stdev is None
                    else (
                        self.synthetic_output_length
                        + random.randint(
                            -self.synthetic_output_length_stdev, self.synthetic_output_length_stdev
                        )
                    )
                ),
                skip_eos=self.skip_eos,
                temperature=self.temperature,
                top_p=self.top_p,
                adapter_path=adapter_path,
                lora_name=lora_name,
            ))
        return requests

    def _build_shared_prefix_inputs(self) -> List[InferencePayload]:
        """
        Generate benchmark requests using two HuggingFace datasets for prefix caching:
        - pg19 dataset provides cacheable prefixes (system prompt region)
        - Openhands dataset provides non-cacheable suffixes (question region)

        Group-based design for prefix caching:
        - Groups = num_examples / concurrency
        - Same cacheable prefix within each group (for effective caching)
        - Different cacheable prefixes across groups
        - Unique questions for each individual request
        """
        assert (
            self.num_examples is not None and self.num_examples > 0
        ), "num_examples must be set for generated-shared-prefix dataset."
        assert (
            self.synthetic_input_length is not None and self.synthetic_output_length is not None
        ), "synthetic_input_length and synthetic_output_length must be set."
        assert 0 <= self.gsp_cached_fraction <= 1, "gsp_cached_fraction must be between 0 and 1"

        num_groups = self.gsp_groups
        assert (
            num_groups is not None and num_groups > 0
        ), "gsp_groups must be provided and greater than 0"

        assert self.num_examples % num_groups == 0, (
            f"num_examples ({self.num_examples}) must be divisible by "
            f"num_groups ({num_groups}) for group-based caching."
        )

        print(
            "Generating fresh shared prefix data with group-based caching "
            "from two HuggingFace datasets..."
        )

        cacheable_dataset = self._fetch_cacheable_source()
        non_cacheable_dataset = self._fetch_suffix_source()

        cached_len = int(self.synthetic_input_length * self.gsp_cached_fraction)
        uncached_len = self.synthetic_input_length - cached_len

        requests_per_group = self.num_examples // num_groups

        print(f"Creating {num_groups} groups with {requests_per_group} requests each")
        print(
            f"gsp_cached_fraction: {self.gsp_cached_fraction}, "
            f"cached_len: {cached_len}, uncached_len: {uncached_len}"
        )

        all_requests = []
        unique_prefixes = set()

        run_offset = int(time() * 1000) % 100000

        for group_idx in tqdm(range(num_groups), desc="Generating Groups"):
            group_prefix_tokens = self._slice_cacheable_prefix(
                cacheable_dataset, group_idx + run_offset, cached_len
            )

            unique_prefixes.add(
                self.tokenizer.decode(group_prefix_tokens[:50], skip_special_tokens=True)
            )

            for req_idx_in_group in range(requests_per_group):
                overall_request_idx = group_idx * requests_per_group + req_idx_in_group

                suffix_tokens = self._slice_suffix_tokens(
                    non_cacheable_dataset, overall_request_idx, uncached_len
                )

                full_prompt_tokens = group_prefix_tokens + suffix_tokens

                full_prompt_tokens = self._pad_or_trim_to_length(
                    full_prompt_tokens, self.synthetic_input_length
                )

                full_prompt = self.tokenizer.decode(full_prompt_tokens, skip_special_tokens=True)

                adapter_path, lora_name = self._pick_adapter_for_request(overall_request_idx)
                request = InferencePayload(
                    prompt=full_prompt,
                    stream=self.stream,
                    max_tokens=self._compute_output_length(),
                    skip_eos=self.skip_eos,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    adapter_path=adapter_path,
                    lora_name=lora_name,
                )
                all_requests.append(request)

        self._display_group_stats(
            all_requests,
            num_groups,
            len(unique_prefixes),
            cached_len,
            uncached_len,
            requests_per_group,
        )

        random.shuffle(all_requests)

        return all_requests

    def _identify_column(self, dataset, chat_mode: bool, dataset_type: str = None) -> str:
        """
        Detect the appropriate column to use from a dataset based on chat mode.
        """

        if isinstance(dataset, list):
            if len(dataset) == 0:
                raise ValueError("Dataset is empty, cannot detect columns")
            available_columns = list(dataset[0].keys())
        else:
            available_columns = list(dataset.column_names)

        if self.dataset_type == "generated-shared-prefix":
            if dataset_type == "cacheable":
                if self.gsp_cacheable_dataset_column_name in available_columns:
                    return self.gsp_cacheable_dataset_column_name
                else:
                    raise ValueError(
                        f"GSP cacheable dataset column "
                        f"'{self.gsp_cacheable_dataset_column_name}' not found. "
                        f"Available columns: {available_columns}"
                    )
            elif dataset_type == "non_cacheable":
                if self.gsp_hf_dataset_column_name in available_columns:
                    return self.gsp_hf_dataset_column_name
                else:
                    raise ValueError(
                        f"GSP non-cacheable dataset column "
                        f"'{self.gsp_hf_dataset_column_name}' not found. "
                        f"Available columns: {available_columns}"
                    )

        if chat_mode:
            if "messages" in available_columns:
                return "messages"
            else:
                text_columns = ["text", "prompt", "input"]
                for col in text_columns:
                    if col in available_columns:
                        return col
                raise ValueError(
                    f"Chat mode enabled but no 'messages' column found in dataset and no suitable "
                    f"text column found. Tried: ['messages'] + {text_columns}. "
                    f"Available columns: {available_columns}"
                )
        else:
            text_columns = ["text", "prompt", "input"]
            for col in text_columns:
                if col in available_columns:
                    return col

            raise ValueError(
                f"No suitable text column found in dataset. Tried: {text_columns}. "
                f"Available columns: {available_columns}"
            )

    def _stream_gsp_dataset(
        self, dataset_name: str, config: str, split: str, num_examples_needed: int = None
    ):
        """
        Load HuggingFace dataset with optional config using streaming.
        """
        if config:
            dataset_stream = datasets.load_dataset(
                dataset_name, config, split=split, streaming=True
            )
        else:
            dataset_stream = datasets.load_dataset(dataset_name, split=split, streaming=True)

        if num_examples_needed is not None:
            buffer_size = int(num_examples_needed * 1.1) + 100
            print(
                f"Streaming {buffer_size} examples from {dataset_name} "
                f"(needed: {num_examples_needed})..."
            )
            examples = []
            for i, example in enumerate(dataset_stream):
                if i >= buffer_size:
                    break
                examples.append(example)
            if not examples:
                raise ValueError(f"Dataset stream from {dataset_name} yielded no examples. ")
            return examples
        else:
            examples = list(dataset_stream)
            if not examples:
                raise ValueError(
                    f"Dataset {dataset_name} is empty. " "Cannot proceed with empty dataset."
                )
            return examples

    def _fetch_cacheable_source(self):
        num_examples_needed = self.gsp_groups if self.gsp_groups else 20
        return self._stream_gsp_dataset(
            self.gsp_cacheable_dataset,
            self.gsp_cacheable_dataset_config,
            self.gsp_cacheable_dataset_split,
            num_examples_needed=num_examples_needed,
        )

    def _fetch_suffix_source(self):
        num_examples_needed = self.num_examples if self.num_examples else 1000
        return self._stream_gsp_dataset(
            self.gsp_hf_dataset,
            self.gsp_hf_dataset_config,
            self.gsp_hf_dataset_split,
            num_examples_needed=num_examples_needed,
        )

    def _slice_cacheable_prefix(
        self, cacheable_dataset, example_index: int, target_length: int
    ) -> List[int]:
        """Extract cacheable prefix tokens from the cacheable dataset."""
        example = cacheable_dataset[example_index % len(cacheable_dataset)]

        column_name = self._identify_column(
            cacheable_dataset, chat_mode=False, dataset_type="cacheable"
        )

        if self.chat:
            messages_data = example[column_name]
            if isinstance(messages_data, str):
                try:
                    messages_data = json.loads(messages_data)
                except json.JSONDecodeError:
                    messages_data = [{"role": "user", "content": messages_data}]

            text = self.tokenizer.apply_chat_template(
                messages_data, tokenize=False, add_generation_prompt=False
            )
        else:
            text = example[column_name]

        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) >= target_length:
            return tokens[:target_length]
        else:
            repeated_tokens = []
            while len(repeated_tokens) < target_length:
                repeated_tokens.extend(tokens)
            return repeated_tokens[:target_length]

    def _unpack_text_from_structure(self, data) -> str:
        """Extract text from chat data structures with validation."""
        try:
            if isinstance(data, dict) and "messages" in data:
                messages = data["messages"]
                if isinstance(messages, list) and len(messages) > 1:
                    if isinstance(messages[1], dict) and "content" in messages[1]:
                        return messages[1]["content"]
            elif isinstance(data, list) and len(data) > 1:
                if isinstance(data[1], dict) and "content" in data[1]:
                    return data[1]["content"]
        except (KeyError, IndexError, TypeError):
            pass

        return str(data)

    def _slice_suffix_tokens(
        self, non_cacheable_dataset, example_index: int, target_length: int
    ) -> List[int]:
        """Extract non-cacheable suffix tokens from non-cacheable dataset."""
        timestamp_ms = int(time() * 1000)
        time_offset = timestamp_ms % 1000
        dataset_index = (example_index + time_offset) % len(non_cacheable_dataset)
        example = non_cacheable_dataset[dataset_index]

        column_name = self._identify_column(
            non_cacheable_dataset, chat_mode=self.chat, dataset_type="non_cacheable"
        )

        try:
            input_data = example[column_name]
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    question = input_data
                else:
                    question = self._unpack_text_from_structure(input_data)
            else:
                question = self._unpack_text_from_structure(input_data)
        except (KeyError, IndexError, TypeError):
            question = str(example[column_name])

        unique_prefix = f" [Request #{example_index} at {timestamp_ms}]"
        question_with_prefix = unique_prefix + question

        tokens = self.tokenizer.encode(question_with_prefix, add_special_tokens=False)

        if len(tokens) >= target_length:
            return tokens[:target_length]
        else:
            repeated_tokens = []
            while len(repeated_tokens) < target_length:
                repeated_tokens.extend(tokens)
            return repeated_tokens[:target_length]

    def _pad_or_trim_to_length(self, tokens: List[int], target_length: int) -> List[int]:
        """Enforce exact token length by truncating or padding."""
        current_len = len(tokens)

        if current_len > target_length:
            return tokens[:target_length]
        elif current_len < target_length:
            pad_token_id = self._resolve_pad_token()
            padding_needed = target_length - current_len
            tokens.extend([pad_token_id] * padding_needed)

        return tokens

    def _resolve_pad_token(self) -> int:
        """Get appropriate padding token ID."""
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        else:
            return self.tokenizer.encode(".", add_special_tokens=False)[0]

    def _display_group_stats(
        self,
        all_requests: List[InferencePayload],
        num_groups: int,
        unique_prefixes: int,
        cached_tokens: int,
        uncached_tokens: int,
        requests_per_group: int,
    ) -> None:
        """Print statistics for group-based two-dataset generation."""
        print("\n=== Group-Based Two-Dataset Shared Prefix Statistics ===")
        print(f"Total requests: {len(all_requests)}")
        print(f"Number of groups: {num_groups}")
        print(f"Requests per group: {requests_per_group}")
        print(f"Unique cacheable prefixes: {unique_prefixes} (should equal {num_groups})")
        print(f"Cacheable prefix length: {cached_tokens} tokens")
        print(f"Non-cacheable suffix length: {uncached_tokens} tokens")
        print(f"Cache fraction: {self.gsp_cached_fraction:.1%}")
        print(f"Cacheable dataset: {self.gsp_cacheable_dataset}")
        print(f"Non-cacheable dataset: {self.gsp_hf_dataset}")
        print(f"Chat mode: {self.chat}")
        print(f"Target input length: {self.synthetic_input_length} tokens")

        if all_requests:
            sample_lengths = [len(self.tokenizer.encode(req.prompt)) for req in all_requests[:5]]
            print(f"Sample prompt lengths (first 5): {sample_lengths}")
            all_correct = all(length == self.synthetic_input_length for length in sample_lengths)
            status = "PASS" if all_correct else "FAIL"
            print(f"Length compliance: {status}")

            if len(all_requests) >= requests_per_group:
                first_group_prompts = [req.prompt for req in all_requests[:requests_per_group]]
                first_group_prefixes = [
                    self.tokenizer.decode(
                        self.tokenizer.encode(prompt, add_special_tokens=False)[:cached_tokens],
                        skip_special_tokens=True,
                    )
                    for prompt in first_group_prompts
                ]
                all_same_prefix = len(set(first_group_prefixes)) == 1
                status = "PASS" if all_same_prefix else "FAIL"
                print(f"First group prefix consistency: {status}")
        print("=" * 55)

    def _compute_output_length(self) -> int:
        """Get output tokens with optional random variation."""
        if self.synthetic_output_length_stdev is None:
            return self.synthetic_output_length
        return self.synthetic_output_length + random.randint(
            -self.synthetic_output_length_stdev, self.synthetic_output_length_stdev
        )
