# agent-bench

Agent workload benchmarking for inference endpoints. The workload is
**multi-turn growing-prefix sessions** — what a real code/chat agent looks
like on the wire.

| invocation | workload shape |
|---|---|
| `agent-bench` (no subcommand) | same as `agent` — the default |
| `agent-bench agent …`  | **primary** — multi-turn code-agent workload with growing prefixes (so a realistic fraction of each request is cache-hittable), configurable human/machine wait time between turns. Backed by `agent.agent_throughput:main`. |
| `agent-bench sweep …`  | QPS sweep / SLO-driven capacity search wrapping `agent`. Backed by `agent.runner:main`. |
| `agent-bench viewer …` | live Dash/Plotly dashboard over `benchmarks/`. Requires the `viewer` extras. Backed by `agent.viewer:main`. |

Run `agent-bench <subcommand> --help` for that subcommand's flags. The
`agent` subcommand has its own `--mode {traffic-replay,realistic,preview}`
for in-mode variations.

`agent-bench` is the only console script the wheel installs. Bare flags with
no subcommand (`agent-bench --server … --model …`) are treated as `agent`
args, since that's the default mode.

Repo layout: `agent/` (driver + workloads), `agentbench/` (top-level CLI
dispatcher).

## Server endpoint

agent-bench drives any OpenAI-compatible `/v1/chat/completions` endpoint
(`--server <url> --model <name>`). sglang, vllm, and OpenAI itself all work.

### Prompt-cache reporting

When the server reports prompt-cache stats in the response `usage`, agent-bench
surfaces the cache-hit rate per request and as an aggregate. It reads either
field, whichever the server populates:

| usage field | reported by |
|---|---|
| `usage.prompt_tokens_details.cached_tokens` | OpenAI / sglang / vllm — for sglang, launch with `--enable-cache-report --enable-prefix-cache` |
| `usage.cache_read_input_tokens` | Anthropic-style usage |

The cache-hit rate then shows up in the agent-mode summary (Cached TPM /
cache hit %).

## Installation

```bash
pip install -e .
```

Or with Docker:

```bash
docker build -t agent-bench .
```

This installs the unified `agent-bench` console script:

| command | what it does |
|---|---|
| `agent-bench` / `agent-bench agent …` | multi-turn workload with growing prefixes (primary) |
| `agent-bench sweep …` | QPS sweep / SLO capacity search wrapping `agent` |
| `agent-bench viewer …` | live Dash/Plotly dashboard (needs `pip install 'agent-bench[viewer]'`) |

Pre-built workload profiles ship inside the wheel under `agent/workloads/*.yaml`
(`agent-bench agent --workload-config <path>`; discover with
`agent.workloads_dir()` / `agent.list_workloads()`).

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
import agent
print(agent.workloads_dir(),  agent.list_workloads())
```

Or pass any of them by absolute path:

```bash
agent-bench agent --workload-config $(python -c 'import agent; print(agent.workloads_dir() / "rag_oneshot.yaml")') \
                  --server http://localhost:8001 --model your/model --tokenizer your/tokenizer
```

The source-checkout invocations `python3 agent/agent_throughput.py …` and
`python3 agent/runner.py …` also work directly.

---

## Primary — `agent-bench agent` (entry point: `agent.agent_throughput:main`)

Multi-turn agent workload. Sessions grow turn-by-turn so the server's prefix
cache gets exercised the way it would in production. Two sub-modes choose
where the turn content comes from.

See `agent-bench agent --help` for the full flag set. The `agent-bench sweep`
wrapper sweeps QPS levels by re-invoking `agent-bench agent`.

### Sub-mode A — random / synthesized turns (default)

Synthesizes turns from random ASCII or, when `AGENT_BENCH_CORPUS` (legacy:
`LIGHTRACE_AGENT_CORPUS`) is set, from a local code corpus. Prompt and
generation lengths are sampled from configurable lognormal distributions.

```bash
# After pip install:
agent-bench agent \
  --server http://localhost:8001 \
  --model qwen3-30b-a3b-nvfp4 \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --workload-config "$(python3 -c 'import agent; print(agent.workloads_dir() / "code_agent_128k.yaml")')" \
  --max-qps 0.3 --ramp-duration 45 --sustain-duration 300

# Or directly from a source checkout:
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

Each row of the dataset becomes one **trace**. The K-th LLM call uses the
first `2K-1` turns as its prompt; the K-th assistant turn's recorded length
becomes the `max_tokens` for that call. The server gets the real prompt
distribution from a working coding agent.

Run it with `--mode dataset-replay`. A fixed pool of `--agent-concurrency`
walkers each pick a trace (round-robin) and replay it call by call, inserting
the interactive waits described below:

```bash
agent-bench agent \
  --mode dataset-replay \
  --server http://localhost:8001 --model your/model \
  --tokenizer your/tokenizer \
  --agent-concurrency 8 \
  --ramp-duration 0 --sustain-duration 300
```

| flag | default | meaning |
|---|---|---|
| `--agent-dataset` | `Inferact/codex_swebenchpro_traces` | HF dataset of agent traces (`conversations` ShareGPT shape) |
| `--agent-dataset-split` | `train` | dataset split to load |
| `--agent-num-traces` | `0` | number of traces to load (`0` = whole split) |
| `--agent-concurrency` | `8` | number of concurrent trace walkers |
| `--agent-wait-machine-secs` | `2.0` | wait after each assistant turn within a trace |
| `--agent-wait-human-secs` | `10.0` | wait at trace boundaries |
| `--agent-wait-jitter` | `0.0` | wait CV (0=deterministic, 1.0=Poisson) |

All flags can also be set under `workload:` in a `--workload-config` YAML
(CLI overrides YAML). The dataset is loaded via HuggingFace `datasets`; point
`HF_HOME` at a volume with free disk if your default cache is small.

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

  # Interactive wait between turns (dataset-replay mode)
  agent_wait_machine_secs: 2.0     # tool / compile / test return latency
  agent_wait_human_secs:   10.0    # human review / next-task dispatch
  agent_wait_jitter:       0.0     # CV; 0=deterministic, 1.0=Poisson, >1=long-tail
```

Resolution order (highest wins):

```
CLI flag  >  workload YAML  >  code default
```

---

## Output

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

## Sweeps

Use `agent-bench sweep` for QPS sweeps and SLO-driven binary search around the
agent workload; see `agent/README.md` for the flag set.

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
