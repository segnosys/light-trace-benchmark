import os

from jsonargparse import ArgumentParser

from . import REGISTERED_BACKENDS


def parse_bool_string(value: str) -> bool:
    if value.lower() in ["true", "yes", "1"]:
        return True
    elif value.lower() in ["false", "no", "0"]:
        return False
    else:
        raise ValueError(f"Invalid value for boolean argument: {value}")


def backend_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group(
        "Inference API backend configurations", "Settings to initialize inference API clients."
    )
    grp.add_argument(
        "--provider",
        required=True,
        type=str,
        choices=list(REGISTERED_BACKENDS.keys()),
        help="Backend provider name.",
    )
    grp.add_argument(
        "--base_url",
        type=str,
        default="https://api.together.xyz/v1",
        help="Base URL for API endpoint.",
    )
    grp.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for endpoint authentication.",
    )
    grp.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use.",
    )
    grp.add_argument(
        "--force_recounting_completions",
        type=str,
        default="false",
        help="Force recounting of completion tokens even when provided by API.",
    )


def load_pattern_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group(
        "General load pattern configurations",
    )
    grp.add_argument(
        "--traffic_pattern",
        type=str,
        choices=["burst", "open_loop_burst", "concurrent", "qps"],
        default="burst",
        help="Load pattern: 'burst' (closed-loop, waits for slowest), "
             "'open_loop_burst' (true open-loop, fires on fixed cadence), "
             "'concurrent' (N workers each running one-at-a-time), "
             "'qps' (rate-based open-loop with arrival distribution).",
    )
    grp.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Concurrency level.",
    )
    grp.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Depending on which load pattern to use, all QPS levels or concurrency "
        "levels to run the benchmark.",
    )


def batch_mode_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group(
        "Batch mode load configurations",
        "Batch mode sends traffic with client-side batch at configured concurrency level.",
    )
    grp.add_argument(
        "--burst_interval",
        type=float,
        default=0.0325,
        help="Interval between request batches.",
    )
    grp.add_argument(
        "--max_num_burst",
        type=int,
        default=100000,
        help="Max number of batched requests.",
    )


def rate_mode_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group(
        "QPS mode load configurations", "Settings for the QPS levels to use for benchmarking."
    )
    grp.add_argument(
        "--qps_distribution",
        type=str,
        choices=["uniform", "exponential", "constant"],
        default="uniform",
        help="Distribution of wait time between requests.",
    )
    grp.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration time in seconds for each QPS level.",
    )


def inference_request_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group(
        "Inference API request configurations",
        "Settings used to formulate a (/chat)/completion inference API request.",
    )
    grp.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        help="Name of the model to use behind the inference API.",
    )
    grp.add_argument(
        "--chat",
        type=str,
        default="true",
        help="Set flag to use /chat/completion.",
    )
    grp.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate.",
    )
    grp.add_argument(
        "--stream",
        type=str,
        default="true",
        help="Set flag to stream the response.",
    )
    grp.add_argument(
        "--ignore_eos",
        type=str,
        default="true",
        help="Set flag to ignore the end of sequence token.",
    )
    grp.add_argument(
        "--temperature",
        type=float,
        default=None,
    )
    grp.add_argument(
        "--top_p",
        type=float,
        default=None,
    )
    grp.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        help="Reasoning effort to use for the request.",
    )
    grp.add_argument(
        "--reasoning_enabled",
        type=str,
        default="false",
        help="Set flag to enable reasoning.",
    )


def data_source_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group("General input data source related configurations")
    grp.add_argument(
        "--dataset_type",
        type=str,
        default="hf",
        choices=["hf", "jsonl", "synthetic", "sharegpt", "generated-shared-prefix"],
        help="Type of dataset to use.",
    )
    grp.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Maximum number of examples/prompts to use.",
    )
    grp.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Huggingface tokenizer to use.",
    )
    grp.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used across the benchmark.",
    )
    grp.add_argument(
        "--same_prompts_in_burst",
        type=str,
        default="false",
        help=(
            "Set flag to send same prompts in a given batch -- usually used for batch mode "
            "to get consistent timing with speculators in multi-batch."
        ),
    )


def huggingface_data_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group("HuggingFace dataset configurations")
    # NOTE: defaults must be publicly accessible. The previous default
    # (togethercomputer/tore-speed-eval-arena-hard-auto) is private and
    # produced DatasetNotFoundError on fresh installs.
    grp.add_argument(
        "--hf_dataset",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="HuggingFace dataset id. Default is a small public reasoning set; "
             "swap in your own with --hf_dataset_column_name as appropriate.",
    )
    grp.add_argument(
        "--hf_dataset_split",
        type=str,
        default="test",
    )
    grp.add_argument(
        "--hf_dataset_config",
        type=str,
        default=None,
        help="Configuration name for the HuggingFace dataset (e.g. 'main' for "
             "openai/gsm8k). Leave unset for datasets that don't have configs.",
    )
    grp.add_argument(
        "--hf_dataset_column_name",
        type=str,
        default="problem",
        help="Column with the prompt text (for --chat false) or a JSON-encoded "
             "chat-messages list (for --chat true). Default 'problem' matches "
             "MATH-500.",
    )


def jsonl_data_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group("JSONL dataset configurations")
    grp.add_argument(
        "--jsonl_input_path",
        type=str,
        default=None,
        help="Path to the JSONL dataset. Ignored if `jsonl_r2_file_ids` is set.",
    )
    grp.add_argument(
        "--jsonl_dataset_column_name",
        type=str,
        default="prompt",
        help="Name of the column in the JSONL dataset that contains the input prompts.",
    )
    grp.add_argument(
        "--jsonl_convert_to_chat_request_format",
        type=str,
        default="false",
        help=(
            "Set flag to convert the input prompts to the chat request format "
            "(Only for single turn). If False, the input prompts are used as is "
            "in 'prompt' key. If True, the input prompts are converted to the "
            "chat request format and sent as 'messages'."
        ),
    )
    grp.add_argument(
        "--jsonl_r2_file_ids",
        type=str,
        default="",
        help=(
            "Comma-separated list of file-IDs of jsonl files uploaded to R2 using Together python "
            "client. If set, `together_api_key` must be set (default: 'TOGETHER_API_KEY' "
            "environment variable). These jsonl files will be downloaded from R2 to "
            "`jsonl_r2_download_dir` and used as input. `jsonl_input_path` is ignored if "
            "this flag is set."
        ),
    )
    grp.add_argument(
        "--jsonl_r2_download_dir",
        type=str,
        default="/tmp",
        help="Path to the directory where `jsonl_r2_file_ids` will be downloaded from R2.",
    )
    grp.add_argument(
        "--together_api_key",
        type=str,
        default=os.getenv("TOGETHER_API_KEY"),
        help="Together API key, used when jsonl_on_r2 is true.",
    )
    grp.add_argument(
        "--use_temperature_top_p_from_jsonl_args",
        type=str,
        default="false",
        help="Set flag to use temperature and top_p from jsonl args.",
    )


def shared_prefix_data_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group("Generated shared prefix dataset configurations")
    grp.add_argument(
        "--gsp_groups",
        type=int,
        default=20,
        help=("Number of groups for prefix caching. Num_examples must be divisible by gsp_groups."),
    )
    grp.add_argument(
        "--gsp_cached_fraction",
        type=float,
        default=0.8,
        help=(
            "Cached fraction of the prompt (0 to 1.0). Controls what portion "
            "comes from pg19 dataset (cacheable) vs Openhands dataset "
            "(non-cacheable). Default 0.8 means 80% pg19 + 20% Openhands."
        ),
    )

    # NOTE: default must be publicly accessible. Previous default was a private
    # togethercomputer dataset that 404'd for non-Together users.
    grp.add_argument(
        "--gsp_hf_dataset",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="HuggingFace dataset to use for generated shared prefix prompts. "
             "Default is a small public reasoning set.",
    )
    grp.add_argument(
        "--gsp_hf_dataset_split",
        type=str,
        default="test",
        help="Split of the HuggingFace dataset to use.",
    )
    grp.add_argument(
        "--gsp_hf_dataset_config",
        type=str,
        default=None,
        help="Configuration name for the HuggingFace dataset (e.g. 'completions'). "
             "Leave unset (or pass 'None') for datasets without configs.",
    )

    grp.add_argument(
        "--gsp_cacheable_dataset",
        type=str,
        default="emozilla/pg19",
        help="HuggingFace dataset to use for cacheable prefixes (default: emozilla/pg19).",
    )
    grp.add_argument(
        "--gsp_cacheable_dataset_split",
        type=str,
        default="train",
        help="Split of the cacheable HuggingFace dataset to use.",
    )
    grp.add_argument(
        "--gsp_cacheable_dataset_config",
        type=str,
        default=None,
        help="Configuration name for the cacheable HuggingFace dataset.",
    )
    grp.add_argument(
        "--gsp_hf_dataset_column_name",
        type=str,
        default="problem",
        help="Column with prompt text in the non-cacheable dataset. "
             "Default 'problem' matches MATH-500; use 'input' for Openhands.",
    )
    grp.add_argument(
        "--gsp_cacheable_dataset_column_name",
        type=str,
        default="text",
        help="Column name in the cacheable dataset (default: 'text' for emozilla/pg19).",
    )


def synthetic_data_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group(
        "Fake input prompt for benchmarking purpose only."
    )
    grp.add_argument(
        "--synthetic_input_length",
        type=int,
        default=None,
        help="Synthetic input length for all requests.",
    )
    grp.add_argument(
        "--synthetic_output_length",
        type=int,
        default=None,
        help="Generation length for all requests from the synthetic input.",
    )
    grp.add_argument(
        "--synthetic_input_length_stdev",
        type=int,
        default=None,
        help="Standard deviation of synthetic input length.",
    )
    grp.add_argument(
        "--synthetic_output_length_stdev",
        type=int,
        default=None,
        help="Standard deviation of generation length from the synthetic input.",
    )
    grp.add_argument(
        "--synthetic_cached_input_length",
        type=int,
        default=None,
        help="Cached input length for all requests.",
    )


def output_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group("Output configurations")
    grp.add_argument(
        "--generation_output_path",
        type=str,
        default=None,
        help="Path to save the generation results in jsonl format",
    )
    grp.add_argument(
        "--evaluation_output_path",
        type=str,
        default="evaluation_results.csv",
        help="Path to save the evaluation results in csv format.",
    )
    grp.add_argument(
        "--histogram_metrics",
        type=str,
        default="false",
        help="Set flag to save tps and ttft of each request to form histograms.",
    )
    grp.add_argument(
        "--verbose",
        type=str,
        default="false",
        help="Set flag to print verbose logs.",
    )
    grp.add_argument(
        "--engine_log_path",
        type=str,
        default=None,
        help=(
            "Path to the engine log file for calculating acceptance rates. "
            "This argument works with all backends (tgi, sglang, vllm, etc.) and "
            "internally resolves to the appropriate log processing based on --provider."
        ),
    )
    grp.add_argument(
        "--engine_metrics_url",
        type=str,
        default=None,
        help=(
            "Optional Prometheus /metrics URL to poll alongside the run "
            "(e.g. http://localhost:30100/metrics for sglang). Server-reported "
            "queue depth / batch size / KV usage are summarized into the report "
            "so you can cross-check client-observed TPOT."
        ),
    )
    grp.add_argument(
        "--engine_metrics_interval_s",
        type=float,
        default=1.0,
        help="Polling cadence for --engine_metrics_url (default: 1.0 second).",
    )
    grp.add_argument(
        "--pulsar_log_path",
        type=str,
        default=None,
        help=(
            "DEPRECATED: Use --engine_log_path instead. "
            "Path to the pulsar log file for calculating acceptance rates from `tgi` backend. "
            "This argument will be removed in a future version."
        ),
    )
    grp.add_argument(
        "--accept_rate_pattern",
        type=str,
        default=None,
        help=(
            "String pattern to extract acceptance rate from engine log file. "
            "If not specified, defaults are used based on --provider: "
            "tgi/pulsar uses 'acceptance_rate.*?(?:2m.*?\\[0m\\s*|=\\[0m)([0-9.e-]+)', "
            "sglang uses 'accept rate:\\s*([0-9.e-]+)'."
        ),
    )


def tracking_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group("Experiment tracking configurations")
    grp.add_argument(
        "--wandb_enabled",
        type=str,
        default="false",
        help="Set flag to enable experiment tracking via W&B.",
    )
    grp.add_argument(
        "--wandb_project",
        type=str,
        default="lightrace",
        help="W&B project name.",
    )
    grp.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team/username).",
    )
    grp.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated list of tags for the W&B run.",
    )
    grp.add_argument(
        "--wandb_notes",
        type=str,
        default=None,
        help="Notes for the W&B run.",
    )


def adapter_options(parser: ArgumentParser) -> None:
    grp = parser.add_argument_group("LoRA inference configurations")
    grp.add_argument(
        "--adapter_paths",
        type=str,
        default=None,
        help=(
            "Comma-separated list of LoRA adapter paths for inference (e.g., "
            "s3://lora1/...,s3://lora2/...). Must be specified together with --lora_ratio. "
            "Mutually exclusive with --adapter_paths_file and --lora_names."
        ),
    )
    grp.add_argument(
        "--adapter_paths_file",
        type=str,
        default=None,
        help=(
            "Path to a text file containing LoRA adapter paths (one per line). "
            "Lines starting with # are treated as comments. "
            "Mutually exclusive with --adapter_paths and --lora_names."
        ),
    )
    grp.add_argument(
        "--lora_names",
        type=str,
        default=None,
        help=(
            "Comma-separated list of pre-loaded LoRA adapter names for inference (e.g., "
            "adapter1,adapter2,adapter3). Use this when adapters are pre-loaded via "
            "/load_lora_adapter API. Must be specified together with --lora_ratio. "
            "Mutually exclusive with --adapter_paths and --adapter_paths_file."
        ),
    )
    grp.add_argument(
        "--lora_ratio",
        type=float,
        default=None,
        help=(
            "Ratio of requests that will use LoRA adapters (0.0-1.0). "
            "1.0 means all requests use LoRA (LoRA-only), "
            "0.5 means 50%% of requests use LoRA (mixed). "
            "Must be specified together with --adapter_paths, --adapter_paths_file, or --lora_names."
        ),
    )
    grp.add_argument(
        "--lora_distribution",
        type=str,
        choices=["round_robin", "random", "uniform"],
        default="round_robin",
        help=(
            "Strategy for distributing multiple LoRA adapters across requests. "
            "round_robin: cycle through adapters sequentially, "
            "random: randomly select an adapter for each request, "
            "uniform: distribute based on request index."
        ),
    )
