# LightRace

Inference endpoint speed benchmarking tool. Measures throughput, latency, and time-to-first-token across different traffic patterns and backends.

## Supported Backends

`openai` | `vllm` | `sglang` | `trtllm`

## Installation

```bash
pip install -e .
```

Or with Docker:

```bash
docker build -t lightrace .
```

## Quick Start

```bash
# Burst mode (default) - synthetic data
lightrace \
  --provider sglang \
  --base_url http://localhost:30000/v1 \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --tokenizer_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_type synthetic \
  --synthetic_input_length 128 \
  --synthetic_output_length 128 \
  --num_examples 20 \
  --concurrency 4 \
  --chat false --stream true --ignore_eos true
```

## Traffic Patterns

LightRace supports three traffic patterns, each simulating a different serving scenario:

### Burst Mode

Sends batched requests at a fixed concurrency level with configurable intervals between batches.

```bash
lightrace \
  --provider sglang \
  --base_url http://localhost:30000/v1 \
  --model_name Qwen/Qwen2.5-72B-Instruct \
  --tokenizer_name Qwen/Qwen2.5-72B-Instruct \
  --traffic_pattern burst \
  --concurrency 16 \
  --max_num_burst 10 \
  --burst_interval 0.0325 \
  --dataset_type synthetic \
  --synthetic_input_length 512 \
  --synthetic_output_length 256 \
  --num_examples 160 \
  --num_gpus 4 \
  --chat false --stream true --ignore_eos true
```

### Concurrent Mode

Maintains N concurrent workers, each sending a new request as soon as the previous one completes.

```bash
lightrace \
  --provider sglang \
  --base_url http://localhost:30000/v1 \
  --model_name Qwen/Qwen2.5-72B-Instruct \
  --tokenizer_name Qwen/Qwen2.5-72B-Instruct \
  --traffic_pattern concurrent \
  --concurrency 32 \
  --dataset_type synthetic \
  --synthetic_input_length 512 \
  --synthetic_output_length 256 \
  --num_examples 128 \
  --num_gpus 4 \
  --chat false --stream true --ignore_eos true
```

### QPS Mode

Sends requests at a target queries-per-second rate with configurable arrival distribution.

```bash
lightrace \
  --provider sglang \
  --base_url http://localhost:30000/v1 \
  --model_name Qwen/Qwen2.5-72B-Instruct \
  --tokenizer_name Qwen/Qwen2.5-72B-Instruct \
  --traffic_pattern qps \
  --levels 4 \
  --duration 30 \
  --qps_distribution uniform \
  --dataset_type synthetic \
  --synthetic_input_length 512 \
  --synthetic_output_length 256 \
  --num_examples 200 \
  --num_gpus 4 \
  --chat false --stream true --ignore_eos true
```

## Dataset Types

| Type | Description |
|---|---|
| `synthetic` | Fixed-length filler prompts for controlled benchmarking |
| `hf` | HuggingFace datasets (default: arena-hard-auto) |
| `jsonl` | Local JSONL files or R2-hosted files |
| `sharegpt` | ShareGPT-format conversation data |
| `generated-shared-prefix` | Two-dataset prefix caching benchmark |

## Benchmark Results

### Qwen2.5-72B-Instruct on 4x NVIDIA B200 (SGLang, TP=4)

Input: 512 tokens, Output: 256 tokens, Synthetic dataset

| Mode | Level | Requests | Failed | User TPS | TTFT p50 (ms) | TTFT p99 (ms) | Job TPS |
|---|---|---|---|---|---|---|---|
| burst | 16 | 160 | 0 | 90.9 | 317 | 776 | 1,246 |
| concurrent | 32 | 128 | 0 | 77.9 | 779 | 974 | 1,982 |
| qps | 4.0 | 120 | 0 | 80.5 | 50 | 396 | 950 |

### Qwen2.5-7B-Instruct on 1x NVIDIA B200 (SGLang)

Input: 128 tokens, Output: 128 tokens, Synthetic dataset

| Mode | Level | Requests | Failed | User TPS | TTFT p50 (ms) | TTFT p99 (ms) | Job TPS |
|---|---|---|---|---|---|---|---|
| burst | 4 | 20 | 0 | 247 | 68 | 489 | 606 |
| concurrent | 8 | 32 | 0 | 255 | 83 | 157 | 1,514 |
| qps | 2.0 | 20 | 0 | 257 | 13 | 40 | 296 |

## Output

Results are saved to CSV (default: `evaluation_results.csv`) and printed as a table:

```
Backend: sglang, Model: Qwen/Qwen2.5-72B-Instruct, GPUs: 4
----------------------------------------  -----------------
Traffic mode:                                         burst
Concurrency level:                                     16.0
Total num. of requests:                                 160
Num. of failed requests:                                  0
Total elapsed time (s):                               32.88
----------------------------------------  -----------------
Prompt length:                                528.3 +/- 0.6
Decode length:                                256.0 +/- 0.0
----------------------------------------  -----------------
Per-request tokens/s:                        90.91 +/- 1.92
Per-request TTFT mean (ms):               363.37 +/- 154.97
Per-request TTFT median (ms):                        316.70
Per-request TTFT P99 (ms):                           775.61
Per-request P99 round-trip latency (ms):               3557
Per-GPU tokens/s:                          321.29 +/- 12.63
Job-level tokens/s (decode):                        1245.75
Job-level actual QPS:                                  4.87
----------------------------------------  -----------------
```

## Advanced Options

### YAML Config

```bash
lightrace --config my_benchmark.yaml
```

### W&B Tracking

```bash
lightrace --wandb_enabled true --wandb_project my-project --wandb_tags "70b,sglang,burst"
```

### LoRA Benchmarking

```bash
lightrace \
  --adapter_paths "s3://bucket/lora1,s3://bucket/lora2" \
  --lora_ratio 0.5 \
  --lora_distribution round_robin \
  ...
```

### Extra Metadata

Attach custom metadata columns to the CSV output:

```bash
lightrace --extra-server us-east-1 --extra-gpu-type b200 ...
```

### Multi-Level Sweeps

Run multiple concurrency or QPS levels in a single invocation:

```bash
lightrace --traffic_pattern burst --levels "4,8,16,32" ...
lightrace --traffic_pattern qps --levels "1,2,4,8" --duration 60 ...
```

## Key Metrics

| Metric | Description |
|---|---|
| **User TPS** | Per-request tokens per second (decode throughput) |
| **TTFT** | Time to first token (prefill latency) |
| **E2E** | End-to-end round-trip latency |
| **Job TPS** | Aggregate decode throughput across all requests |
| **Per-GPU TPS** | Throughput normalized per GPU (burst mode) |
| **Acceptance Rate** | Speculative decoding acceptance ratio (TRT-LLM / engine logs) |
