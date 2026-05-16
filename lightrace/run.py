import asyncio
import logging
import random
import sys
import time
from pprint import pprint
from typing import Dict, List, Tuple

import jsonargparse
import numpy as np

from lightrace.cli_options import (
    adapter_options,
    batch_mode_options,
    backend_options,
    data_source_options,
    huggingface_data_options,
    inference_request_options,
    jsonl_data_options,
    load_pattern_options,
    output_options,
    parse_bool_string,
    rate_mode_options,
    shared_prefix_data_options,
    synthetic_data_options,
    tracking_options,
)
from lightrace.input_pipeline import InputPipeline
from lightrace.bench_runner import InferenceBenchRunner
from lightrace.analytics import (
    export_report_csv,
    render_report,
    summarize_benchmark,
)
from lightrace.experiment_tracker import (
    ExperimentTracker,
    capture_invocation,
    split_tag_string,
)


def _extract_extra_eval_metadata(
    parser: jsonargparse.ArgumentParser, argv: List[str]
) -> Tuple[List[str], Dict[str, str]]:
    """
    Remove --extra-* flags from argv before handing the remainder to jsonargparse.
    Supports --extra-key=value and --extra-key value forms.
    """
    filtered_argv: List[str] = []
    extra_metadata: Dict[str, str] = {}

    idx = 0
    while idx < len(argv):
        token = argv[idx]

        if not token.startswith("--extra-"):
            filtered_argv.append(token)
            idx += 1
            continue

        flag = token
        value: str

        if "=" in token:
            flag, value = token.split("=", 1)
        else:
            idx += 1
            if idx >= len(argv):
                parser.error(f"Expected value after {token}")
            value = argv[idx]
            if value.startswith("--"):
                parser.error(f"Expected value after {token}, got flag {value} instead")

        key = flag[len("--extra-"):].strip()
        if not key:
            parser.error("Extra metadata flag must include a name, e.g. --extra-server us-east-1")

        normalized_key = f"extra_{key.replace('-', '_')}"
        if normalized_key in extra_metadata:
            parser.error(f"Duplicate extra metadata flag provided for {normalized_key}")

        extra_metadata[normalized_key] = str(value)
        idx += 1

    return filtered_argv, extra_metadata


def main(argv: List[str] | None = None):
    parser = jsonargparse.ArgumentParser(prog="LightRace Inference Benchmark")
    backend_options(parser)
    inference_request_options(parser)
    load_pattern_options(parser)
    batch_mode_options(parser)
    rate_mode_options(parser)
    data_source_options(parser)
    huggingface_data_options(parser)
    jsonl_data_options(parser)
    synthetic_data_options(parser)
    shared_prefix_data_options(parser)
    output_options(parser)
    tracking_options(parser)
    adapter_options(parser)
    parser.add_argument("--config", action="config", help="Path to benchmarking configs in .yaml")
    if argv is None:
        argv = sys.argv[1:]
    filtered_argv, extra_eval_metadata = _extract_extra_eval_metadata(parser, argv)
    args = parser.parse_args(filtered_argv)
    setattr(args, "extra_eval_metadata", extra_eval_metadata)
    pprint(args.as_dict())

    # Validate LoRA arguments
    lora_source_count = sum([
        bool(args.adapter_paths),
        bool(args.adapter_paths_file),
        bool(args.lora_names),
    ])
    if lora_source_count > 1:
        raise ValueError(
            "Cannot specify more than one of --adapter_paths, --adapter_paths_file, or --lora_names. "
            "Please use only one method to specify LoRA adapters."
        )

    # Load adapter paths from file if specified
    if args.adapter_paths_file:
        try:
            with open(args.adapter_paths_file, "r") as f:
                adapter_paths_list = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        adapter_paths_list.append(line)

                if not adapter_paths_list:
                    raise ValueError(
                        f"No adapter paths found in {args.adapter_paths_file}. "
                        "File should contain at least one non-comment line."
                    )

                args.adapter_paths = ",".join(adapter_paths_list)
                logging.info(
                    f"Loaded {len(adapter_paths_list)} adapter path(s) from "
                    f"{args.adapter_paths_file}"
                )
        except FileNotFoundError:
            raise ValueError(
                f"Adapter paths file not found: {args.adapter_paths_file}. "
                "Please ensure the file exists and the path is correct."
            )

    # Validate that LoRA source and lora_ratio are specified together
    has_lora_source = bool(args.adapter_paths) or bool(args.lora_names)
    has_lora_ratio = args.lora_ratio is not None
    if has_lora_source != has_lora_ratio:
        raise ValueError(
            "LoRA source (--adapter_paths, --adapter_paths_file, or --lora_names) and "
            "--lora_ratio must be specified together. "
            f"Got adapter_paths={args.adapter_paths}, lora_names={args.lora_names}, lora_ratio={args.lora_ratio}"
        )

    logging.basicConfig(level=logging.DEBUG if parse_bool_string(args.verbose) else logging.INFO)

    # Handle engine_log_path and accept_rate_pattern arguments
    engine_log_path = args.engine_log_path
    accept_rate_pattern = args.accept_rate_pattern

    # Handle deprecated pulsar_log_path argument
    if args.pulsar_log_path is not None:
        logging.warning(
            "WARNING: --pulsar_log_path is deprecated and will be removed in a future version. "
            "Please use --engine_log_path instead."
        )
        if engine_log_path is None:
            engine_log_path = args.pulsar_log_path

    # Set default accept_rate_pattern based on provider if not specified
    if accept_rate_pattern is None and engine_log_path is not None:
        if args.provider in ["tgi"]:
            accept_rate_pattern = r"acceptance_rate.*?(?:2m.*?\[0m\s*|=\[0m)([0-9.e-]+)"
            logging.info(f"Using default acceptance rate pattern for {args.provider} backend")
        elif args.provider in ["sglang"]:
            accept_rate_pattern = r"accept rate:\s*([0-9.e-]+)"
            logging.info(f"Using default acceptance rate pattern for {args.provider} backend")
        else:
            logging.warning(
                f"Backend '{args.provider}' does not have log-based acceptance rate tracking. "
                f"--engine_log_path will be ignored."
            )

    if args.dataset_type == "generated-shared-prefix":
        gsp_supported = {"tgi", "sglang", "vllm", "together"}
        if args.provider not in gsp_supported:
            raise ValueError(
                f"Generated Shared Prefix (GSP) dataset type is only supported with backends: "
                f"{sorted(gsp_supported)}. Got backend: {args.provider}"
            )

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    pipeline = InputPipeline(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        dataset_type=args.dataset_type,
        jsonl_input_path=args.jsonl_input_path,
        jsonl_dataset_column_name=args.jsonl_dataset_column_name,
        jsonl_convert_to_chat_request_format=parse_bool_string(
            args.jsonl_convert_to_chat_request_format
        ),
        jsonl_r2_file_ids=args.jsonl_r2_file_ids,
        jsonl_r2_download_dir=args.jsonl_r2_download_dir,
        together_api_key=args.together_api_key,
        hf_dataset=args.hf_dataset,
        hf_dataset_split=args.hf_dataset_split,
        hf_dataset_config=args.hf_dataset_config,
        hf_dataset_column_name=args.hf_dataset_column_name,
        synthetic_input_length=args.synthetic_input_length,
        synthetic_output_length=args.synthetic_output_length,
        synthetic_input_length_stdev=args.synthetic_input_length_stdev,
        synthetic_output_length_stdev=args.synthetic_output_length_stdev,
        synthetic_cached_input_length=args.synthetic_cached_input_length,
        num_examples=args.num_examples,
        stream=parse_bool_string(args.stream),
        max_tokens=args.max_tokens,
        skip_eos=parse_bool_string(args.ignore_eos),
        temperature=args.temperature,
        top_p=args.top_p,
        chat=parse_bool_string(args.chat),
        reasoning_effort=args.reasoning_effort,
        reasoning_enabled=parse_bool_string(args.reasoning_enabled),
        same_prompts_in_burst=parse_bool_string(args.same_prompts_in_burst),
        concurrency=args.concurrency,
        use_temperature_top_p_from_jsonl_args=parse_bool_string(
            args.use_temperature_top_p_from_jsonl_args
        ),
        gsp_cached_fraction=args.gsp_cached_fraction,
        gsp_hf_dataset=args.gsp_hf_dataset,
        gsp_hf_dataset_split=args.gsp_hf_dataset_split,
        gsp_hf_dataset_config=args.gsp_hf_dataset_config,
        gsp_cacheable_dataset=args.gsp_cacheable_dataset,
        gsp_cacheable_dataset_split=args.gsp_cacheable_dataset_split,
        gsp_cacheable_dataset_config=args.gsp_cacheable_dataset_config,
        gsp_hf_dataset_column_name=args.gsp_hf_dataset_column_name,
        gsp_cacheable_dataset_column_name=args.gsp_cacheable_dataset_column_name,
        traffic_pattern=args.traffic_pattern,
        qps_duration_s=args.duration,
        gsp_groups=args.gsp_groups,
        adapter_paths=args.adapter_paths,
        lora_names=args.lora_names,
        lora_ratio=args.lora_ratio,
        lora_distribution=args.lora_distribution,
    )

    levels = (
        [args.concurrency]
        if not args.levels
        else [float(level) for level in args.levels.split(",")]
    )

    if args.levels and parse_bool_string(args.same_prompts_in_burst):
        raise ValueError(
            "Cannot specify both 'levels' and 'same_prompts_in_burst' args. "
            "Use 'concurrency' instead."
        )

    # Initialize experiment tracker outside the loop
    tracker = ExperimentTracker(
        enabled=parse_bool_string(args.wandb_enabled),
        project=args.wandb_project,
        entity=args.wandb_entity,
        tags=split_tag_string(args.wandb_tags),
        notes=args.wandb_notes,
    )

    command_string = capture_invocation()

    if tracker.enabled:
        config = args.as_dict()
        config["levels"] = levels
        if extra_eval_metadata:
            config["extra_eval_metadata"] = extra_eval_metadata
        tracker.start_run(config, command_string, args.evaluation_output_path)

    for step, level in enumerate(levels):
        runner = InferenceBenchRunner(
            provider=args.provider,
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            traffic_pattern=args.traffic_pattern,
            force_recounting_completions=parse_bool_string(args.force_recounting_completions),
        )

        # load the dataset before the timer starts
        loaded_requests = pipeline.prepare_inputs()

        start_time = time.perf_counter()
        results, per_batch_elapsed_times = asyncio.run(
            runner.run_benchmark(
                requests=loaded_requests,
                level=level,
                max_num_burst=args.max_num_burst,
                burst_interval=args.burst_interval,
                qps_distribution=args.qps_distribution,
                qps_duration_s=args.duration,
            )
        )
        total_elapse_time_s = time.perf_counter() - start_time

        if args.generation_output_path:
            with open(args.generation_output_path, "w") as f:
                for result in results:
                    f.write(result.to_json() + "\n")

        # aggregate results
        report = summarize_benchmark(
            traffic_mode=args.traffic_pattern,
            traffic_level=level,
            backend_name=args.provider,
            model_name=args.model_name,
            total_elapse_time_s=total_elapse_time_s,
            results=results,
            num_gpus=args.num_gpus,
            per_batch_elapsed_times=per_batch_elapsed_times,
            histogram_metrics=parse_bool_string(args.histogram_metrics),
            hf_dataset_name=args.hf_dataset if args.dataset_type == "hf" else None,
            engine_log_path=engine_log_path,
            accept_rate_pattern=accept_rate_pattern,
        )
        render_report(report)

        # If mixed LoRA traffic, report separate metrics for LoRA and non-LoRA requests
        has_lora_config = args.adapter_paths or args.lora_names
        if has_lora_config and args.lora_ratio and 0 < args.lora_ratio < 1:
            lora_results = [r for r in results if r.request and (r.request.adapter_path or r.request.lora_name)]
            non_lora_results = [r for r in results if r.request and not r.request.adapter_path and not r.request.lora_name]

            if lora_results:
                print("\n" + "=" * 60)
                print("=== LoRA Requests Metrics ===")
                print("=" * 60)
                lora_report = summarize_benchmark(
                    traffic_mode=args.traffic_pattern,
                    traffic_level=level,
                    backend_name=args.provider,
                    model_name=f"{args.model_name} [LoRA]",
                    total_elapse_time_s=total_elapse_time_s,
                    results=lora_results,
                    num_gpus=args.num_gpus,
                    per_batch_elapsed_times=None,
                    histogram_metrics=parse_bool_string(args.histogram_metrics),
                    hf_dataset_name=args.hf_dataset if args.dataset_type == "hf" else None,
                )
                render_report(lora_report)

            if non_lora_results:
                print("\n" + "=" * 60)
                print("=== Non-LoRA Requests Metrics ===")
                print("=" * 60)
                non_lora_report = summarize_benchmark(
                    traffic_mode=args.traffic_pattern,
                    traffic_level=level,
                    backend_name=args.provider,
                    model_name=f"{args.model_name} [Base]",
                    total_elapse_time_s=total_elapse_time_s,
                    results=non_lora_results,
                    num_gpus=args.num_gpus,
                    per_batch_elapsed_times=None,
                    histogram_metrics=parse_bool_string(args.histogram_metrics),
                    hf_dataset_name=args.hf_dataset if args.dataset_type == "hf" else None,
                )
                render_report(non_lora_report)
        export_report_csv(
            report,
            args.evaluation_output_path,
            extra_eval_metadata=extra_eval_metadata,
        )

        # Log metrics to tracker for this level/row with step number
        tracker.record_metrics(report, step)

    # Log the final CSV file as a tracker artifact (contains all levels)
    tracker.upload_csv_artifact(args.evaluation_output_path)

    # Finish the single tracker run after all levels are complete
    tracker.end_run()


if __name__ == "__main__":
    main()
