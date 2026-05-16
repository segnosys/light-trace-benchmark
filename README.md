# LightTrace / optimus-agent-bench

Inference endpoint benchmarking tools. **Two separate entry points**, one per
workload shape:

* **`lightrace`** — `batch` mode. Open-loop / closed-loop synthetic traffic
  (burst, concurrent, QPS). Measures raw throughput and TTFT under controlled
  load. This is the classic LightTrace behavior. Entry point:
  `lightrace.run:main`, console script `lightrace`.
* **`agent/agent_throughput.py`** — agent mode. Interactive code-agent workload
  with multi-turn sessions, growing prefixes (so a realistic fraction of each
  request is cache-hittable), configurable human/machine wait time between
  turns, and an optional QPS sweep wrapper at `agent/runner.py`.

These are two distinct programs with their own CLIs. There is **no unified
`--mode` flag** today — the README originally claimed one but the work to
merge them isn't done. Use the table below to pick the right tool, then read
the section that matches.

## Supported Backends

`openai` | `vllm` | `sglang` | `trtllm` | `anthropic` | `tgi` | `fireworks` | `nvidia_nim` | `together` | `embeddings`

### Prompt-cache reporting

When the backend reports prompt-cache stats, lightrace surfaces them per request and as an aggregate:

| backend | cache fields populated | how to enable |
|---|---|---|
| `anthropic` | `cached_input_tokens` (read) + `cache_creation_input_tokens` (write) | always on — backend marks the system prompt (or last user content block) with `cache_control: {"type":"ephemeral"}` so identical-prefix requests register cache hits |
| `openai` / `sglang` / `vllm` | `cached_input_tokens` from `usage.prompt_tokens_details.cached_tokens` | server must report — for sglang launch with `--enable-cache-report --enable-prefix-cache`; OpenAI does it automatically |
| `trtllm` / `tgi` / `fireworks` / `nvidia_nim` / `together` | (none today — these backends don't surface cache stats) | n/a |

Report rendering adds two extra rows when any backend reports caching:

```
Prompt cache hit rate:     78.4% +/- 12.1%
Total cached input tokens: 318472
```

The pre-built configs `anthropic_cache_demo.yaml` and `sglang_cache_report.yaml` exercise these paths.

## Installation

```bash
pip install -e .
```

Or with Docker:

```bash
docker build -t lightrace .
```

This installs three console scripts:

| command | what it does |
|---|---|
| `lightrace` | batch-mode entry point (open-loop / closed-loop synthetic load) |
| `lightrace-agent` | agent-mode driver (`agent.agent_throughput:main`) — multi-turn workload with growing prefixes |
| `lightrace-agent-sweep` | QPS-sweep wrapper around `lightrace-agent` |

Two sets of pre-built workload profiles ship inside the wheel:

| set | for | discover via |
|---|---|---|
| `lightrace/configs/*.yaml` | `lightrace --config <path>` (batch mode) | `lightrace.configs_dir()` / `lightrace.list_configs()` |
| `agent/workloads/*.yaml` | `lightrace-agent --workload-config <path>` | `agent.workloads_dir()` / `agent.list_workloads()` |

Bundled batch profiles (typical scenarios — pick one and add `--base_url`/`--model_name`):

| name | shape | what it stresses |
|---|---|---|
| `chat_short` | 200 in / 200 out / concurrent=32 | scheduling + decode throughput at ChatGPT-style traffic |
| `rag_doc_qa` | 10K in / 300 out / concurrent=16 | prefill BW + attention on long context |
| `code_completion` | 2K in / 64 out / qps=10 constant | TTFT SLO at sustained low-latency arrivals |
| `reasoning_long_decode` | 2K in / 8K out / concurrent=8 | TPOT stability over long generation + KV pressure |
| `long_prefill_ttft` | 65K in / 100 out / concurrent=4 | TTFT and chunked-prefill sizing |
| `pure_cold_random` | 64K in / 800 out / concurrent=24 | mirrors the cg+profile kernel-decomposition workload |
| `hf_math500_reasoning` / `hf_gsm8k` / `hf_humaneval` | public eval-set prompts | real prompt distribution, not synthetic filler |
| `prefix_cache_80pct` | gsp 80% cached / concurrent=16 | OS prefix-cache effectiveness on shared-prefix traffic |
| `sharegpt_chat` | sharegpt 512/256 / concurrent=24 | chat shape without HF dataset auth |
| `jsonl_template` | replay-from-jsonl skeleton | swap in your own prompts |
| `anthropic_cache_demo` | Claude + `cache_control` markers | Anthropic Messages API + prompt caching |
| `sglang_cache_report` | shared-prefix shape, reports cache hits | sglang radix-cache hit-rate measurement |

Bundled agent profiles (multi-turn shapes):

| name | archetype |
|---|---|
| `code_agent_{16k,50k_cache90,64k_cache935_kimi,128k,200k}` | growing-prefix code-agent variants at different context sizes / cache-hit targets |
| `code_agent_50k_cache90_{kimi,mxfp4}` | same shape with model-specific tokenizers |
| `code_agent_50k_cache94_kimi` | tighter cache-hit target for the same Kimi tokenizer |
| `chat_assistant_short` | ChatGPT-style: short prompts, high session-rotation rate |
| `rag_oneshot` | single-turn RAG (`new_session_rate=1.0`), big retrieved context |

Discover at runtime:

```python
import lightrace, agent
print(lightrace.configs_dir(), lightrace.list_configs())
print(agent.workloads_dir(),   agent.list_workloads())
```

Or pass any of them by absolute path:

```bash
lightrace --config $(python -c 'import lightrace; print(lightrace.configs_dir() / "chat_short.yaml")') \
          --base_url http://localhost:8001/v1 \
          --model_name your/model --tokenizer_name your/tokenizer

lightrace-agent --workload-config $(python -c 'import agent; print(agent.workloads_dir() / "rag_oneshot.yaml")') \
                --server http://localhost:8001 --model your/model --tokenizer your/tokenizer
```

The legacy invocations `python3 agent/agent_throughput.py …` and
`python3 agent/runner.py …` still work from a source checkout. The
`optimus-agent-bench` name from earlier doc copy is **not** a registered
script — earlier README claims of a unified `--mode` flag were aspirational.

---

## Mode 1 — `batch` (entry point: `lightrace`)

Synthetic traffic shaped by one of three patterns. Choose `--traffic_pattern`.

### Burst

Sends batched requests at a fixed concurrency level with configurable intervals
between batches.

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

### Concurrent

N workers, each sends a new request as soon as the previous one completes.

```bash
lightrace \
  --traffic_pattern concurrent \
  --concurrency 32 \
  ...
```

### QPS

Requests arrive at a target rate with configurable inter-arrival distribution.

```bash
lightrace \
  --traffic_pattern qps \
  --levels 4 \
  --duration 30 \
  --qps_distribution uniform \
  ...
```

### Dataset Types (batch mode)

| Type | Description |
|---|---|
| `synthetic` | Fixed-length filler prompts for controlled benchmarking |
| `hf` | HuggingFace datasets (default: arena-hard-auto) |
| `jsonl` | Local JSONL files or R2-hosted files |
| `sharegpt` | ShareGPT-format conversation data |
| `generated-shared-prefix` | Two-dataset prefix caching benchmark |

---

## Mode 2 — agent (entry point: `agent/agent_throughput.py`)

Multi-turn agent workload. Sessions grow turn-by-turn so the server's prefix
cache gets exercised the way it would in production. Two sub-modes choose
where the turn content comes from.

This mode is a separate program from `lightrace` and has its own CLI; see
`python3 agent/agent_throughput.py --help`. The `agent/runner.py` wrapper
sweeps QPS levels by re-invoking `agent_throughput.py`.

### Sub-mode A — random / synthesized turns (default)

Synthesizes turns from random ASCII or, when `LIGHTRACE_AGENT_CORPUS` is set,
from a local code corpus. Prompt and generation lengths are sampled from
configurable lognormal distributions.

```bash
# After pip install:
lightrace-agent \
  --server http://localhost:8001 \
  --model qwen3-30b-a3b-nvfp4 \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --workload-config "$(python3 -c 'import agent; print(agent.workloads_dir() / "code_agent_128k.yaml")')" \
  --max-qps 0.3 --ramp-duration 45 --sustain-duration 300

# Or from a source checkout (legacy form, still works):
python3 agent/agent_throughput.py \
  --server http://localhost:8001 \
  --model qwen3-30b-a3b-nvfp4 \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --workload-config agent/workloads/code_agent_128k.yaml \
  --max-qps 0.3 --ramp-duration 45 --sustain-duration 300
```

YAML profiles under `agent/workloads/` parameterize prompt growth, session
selection bias, and arrival shape. See `agent/README.md` for the full field
reference.

### Sub-mode B — replay-from-dataset (experimental)

Replay real agent traces. Default dataset is
[`Inferact/codex_swebenchpro_traces`](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces)
(610 SWE-bench-pro trials from a Codex-style agent, ~20K LLM calls total,
ShareGPT-format `human↔gpt` alternation).

Each row of the dataset becomes one `ChatSession`. The K-th LLM call uses the
first `2K-1` turns as its prompt; the K-th assistant turn's recorded length
becomes the `max_tokens` for that call. Server gets the real prompt
distribution from a working coding agent.

Note: the prior README copy described `--agent-input dataset` /
`--agent-dataset` flags. Those flags are not currently wired into
`agent_throughput.py --help`; see `agent/README.md` for the actually
supported configuration.

Measured properties of `Inferact/codex_swebenchpro_traces` (100-trace sample,
char-count proxy — confirms the dataset card's 94.2% self-report):

| metric | value |
|---|---|
| Aggregate ideal cache hit | **95.6 %** |
| Per-trial cache hit p10 / p50 / p90 | 87.8 % / 93.2 % / 97.0 % |
| LLM calls per trace | p50 21, mean 27, max 64 |
| Mean prompt length | ~70K tokens |
| First-call prompt length | ~12.5K tokens |
| Generation length p50 | ~280 tokens |

The assistant turns in the dataset are length-preserving Lorem-ipsum
placeholders (Inferact redacted the generated text). For benchmarking this is
the right shape — prefill / decode token counts are the load-bearing
quantities; the actual decoded text is server-generated anyway.

### Interactive wait time

Between LLM calls the driver inserts a wait that simulates the latency outside
the model: tools running, or a human reading and confirming.

```
--agent-wait-machine-secs   default 2.0     # tool-execution wait
--agent-wait-human-secs     default 10.0    # human-in-the-loop wait
--agent-wait-jitter         default 0.0     # 0 = deterministic
```

**Where each wait fires**:

| wait | inserted where |
|---|---|
| **machine** | after every assistant turn within a trace / session (≈ tool execution time) |
| **human** | at session boundaries — between two traces, or when a new session is spun up — (≈ user reviewing & dispatching the next task) |

**Jitter**: variation coefficient (CV = std/mean) for a Gamma-distributed
wait. `jitter=0` collapses to exactly `mean` seconds. `jitter=1.0` is the
classic Poisson inter-arrival (exponential). Floor 0.05 s for machine, 1 s
for human; cap 300 s.

```
sampled_wait ~ Gamma(shape = 1 / jitter², scale = mean × jitter²)
# E = mean, CV = jitter
```

**Recommended values**:

| Scenario | machine | human | jitter |
|---|---|---|---|
| CI / reproducible run (deterministic) | 2.0 | 10.0 | 0.0 |
| Realistic but smooth | 2.0 | 10.0 | 0.3 |
| Realistic burst (Poisson) | 2.0 | 10.0 | 1.0 |
| Long-tail human review | 3.0 | 30.0 | 1.5 |

Set either knob to `0` to disable that wait entirely. To pin a single
deterministic delay (no human/machine split), use `--agent-wait-machine-secs N
--agent-wait-human-secs 0 --agent-wait-jitter 0`.

**Configured via YAML**: the three wait knobs live in the workload YAML so a
run is fully reproducible from a single config file. CLI flags override the
YAML; if neither is given, code defaults (2.0 / 10.0 / 0.0) apply.

```yaml
# agent/workloads/<profile>.yaml
workload:
  # ... existing fields ...

  # Interactive wait between turns (agent mode only)
  agent_wait_machine_secs: 2.0     # tool / compile / test return latency
  agent_wait_human_secs:   10.0    # human review / next-task dispatch
  agent_wait_jitter:       0.0     # CV; 0=deterministic, 1.0=Poisson, >1=long-tail
```

Resolution order (highest wins):

```
CLI flag  >  workload YAML  >  code default
```

---

## Benchmark Results

### Qwen2.5-72B-Instruct on 4x NVIDIA B200 (SGLang, TP=4) — batch mode

Input: 512 tokens, Output: 256 tokens, Synthetic dataset

| Pattern | Level | Requests | Failed | User TPS | TTFT p50 (ms) | TTFT p99 (ms) | Job TPS |
|---|---|---|---|---|---|---|---|
| burst | 16 | 160 | 0 | 90.9 | 317 | 776 | 1,246 |
| concurrent | 32 | 128 | 0 | 77.9 | 779 | 974 | 1,982 |
| qps | 4.0 | 120 | 0 | 80.5 | 50 | 396 | 950 |

### Qwen2.5-7B-Instruct on 1x NVIDIA B200 (SGLang) — batch mode

Input: 128 tokens, Output: 128 tokens, Synthetic dataset

| Pattern | Level | Requests | Failed | User TPS | TTFT p50 (ms) | TTFT p99 (ms) | Job TPS |
|---|---|---|---|---|---|---|---|
| burst | 4 | 20 | 0 | 247 | 68 | 489 | 606 |
| concurrent | 8 | 32 | 0 | 255 | 83 | 157 | 1,514 |
| qps | 2.0 | 20 | 0 | 257 | 13 | 40 | 296 |

---

## Output

### Batch mode

Results are saved to CSV (default: `evaluation_results.csv`) and printed as
a table:

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

### Agent mode

Per-run artifacts land in `benchmarks/<name>/<timestamp>/`:

```
metadata.json     # resolved config
metrics.jsonl     # per-second rolling TPM / TPS / cache / in-flight / sessions
summary.json      # final summary, including phase-by-phase breakdown
```

Look for the **Phase Throughput Breakdown** in stdout; the `sustain` row is
the capacity number to quote. See `agent/README.md` for the column-by-column
explanation.

### Live dashboard

For agent-mode runs there's a Dash/Plotly viewer at `agent/viewer.py`
(invoke via `python -m agent.viewer --data-dir <runs>` or `./agent/run_viewer.sh`).
Install the optional deps with `pip install dash plotly` (not part of
`install_requires`).

The viewer binds to `0.0.0.0:8050` so a remote dev box exposes it at
`http://<host>:8050`. If the port isn't open in the firewall, use SSH local
port forwarding from your laptop:

```bash
ssh -N -L 8050:localhost:8050 user@remote-dev-host
# now open http://localhost:8050 in a browser on the laptop
```

Add the forward to `~/.ssh/config` (`LocalForward 8050 localhost:8050`) to
make it automatic. Multiple viewers on different ports? Add `-L 8051:...`
flags as needed. Through a jump host? `ssh -J jump-host ...`.

See `agent/README.md` for the full SSH port-forward recipes (including
`~/.ssh/config` templates and behind-a-bastion patterns).

---

## Advanced Options

### YAML Config

```bash
lightrace --config my_benchmark.yaml
```

### W&B Tracking

```bash
lightrace --wandb_enabled true --wandb_project my-project \
  --wandb_tags "70b,sglang,burst"
```

### LoRA Benchmarking (batch mode)

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

### Multi-Level Sweeps (batch mode)

```bash
lightrace --traffic_pattern burst --levels "4,8,16,32" ...
lightrace --traffic_pattern qps --levels "1,2,4,8" --duration 60 ...
```

For agent mode, use `agent/runner.py` for QPS sweeps and SLO-driven binary
search; see `agent/README.md`.

---

## Key Metrics

| Metric | Description |
|---|---|
| **User TPS** | Per-request tokens per second (decode throughput) |
| **TTFT** | Time to first token (prefill latency) |
| **TPOT** | Time per output token (excludes TTFT) |
| **E2E** | End-to-end round-trip latency |
| **Job TPS** | Aggregate decode throughput across all requests |
| **Per-GPU TPS** | Throughput normalized per GPU |
| **Input TPM / Cached TPM / Uncached TPM** | (agent mode) tokens-per-minute split by cache reuse |
| **Cache hit %** | (agent mode) `cached_tokens / prompt_tokens` reported by server |
| **Acceptance Rate** | Speculative decoding acceptance ratio (TRT-LLM / engine logs) |
