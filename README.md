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
`agent` subcommand has its own `--mode {traffic-replay,realistic,preview,dataset-replay}`
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
  --agent-session-salt-tokens 256 \
  --ramp-duration 0 --sustain-duration 300
```

`--agent-session-salt-tokens 256` gives every replay a unique prefix so the KV
working set grows past the engine's L1 cache and exercises the external KV tier
(see [Reaching the external KV cache](#reaching-the-external-kv-cache)). Drop it
to `0` for a pure, byte-faithful replay.

| flag | default | meaning |
|---|---|---|
| `--agent-dataset` | `Inferact/codex_swebenchpro_traces` | HF dataset of agent traces (`conversations` ShareGPT shape) |
| `--agent-dataset-split` | `train` | dataset split to load |
| `--agent-num-traces` | `0` | number of traces to load (`0` = whole split) |
| `--agent-concurrency` | `8` | number of concurrent trace walkers |
| `--agent-wait-machine-secs` | `2.0` | **fallback** machine wait, used only for calls with no recorded tool wall-time |
| `--agent-wait-human-secs` | `0.0` | human wait at trace boundaries (`0` = autonomous batch, the codex dataset's true shape) |
| `--agent-wait-jitter` | `0.0` | CV for the simulated/fallback waits (0=deterministic, 1.0=Poisson) |
| `--agent-wait-scale` | `1.0` | multiplier on the inter-call machine wait (recorded or fallback); `0` disables, `<1` speeds up replay |
| `--agent-session-salt-tokens` | `0` | prepend a unique per-replay session id (~N tokens); `0` = off. See [Reaching the external KV cache](#reaching-the-external-kv-cache) |

All flags can also be set under `workload:` in a `--workload-config` YAML
(CLI overrides YAML). The dataset is loaded via HuggingFace `datasets`; point
`HF_HOME` at a volume with free disk if your default cache is small.

### Reaching the external KV cache

The full 610-trace dataset is *small*: its entire KV working set is only
~8.5 GB. On a server whose first-tier (L1 / GPU+host) prefix cache is larger
than that, **the whole dataset fits in L1 and nothing ever spills to the
external KV tier** — in steady state the external `gets` stay at 0, and a
reported "external hit ~84–85%" is really the L1 hit, not external reads.

Worse, the round-robin walkers replay the same traces over and over; after the
first pass every re-replay is byte-identical, so it dedups to the same cached
blocks and the working set never grows.

`--agent-session-salt-tokens N` fixes this by prepending a **unique per-replay
session id** (a uuid padded to ~N tokens) as a leading system message:

- The salt is **constant across one replay's calls**, so the within-trace
  growing-prefix cache hits are preserved exactly (turn K still reuses turn
  K-1's prefix).
- The salt **differs across replays/traces**, so each replay produces a fresh
  KV chain that no longer dedups. The working set grows with traffic until it
  exceeds L1 and starts reading the external tier — which is what you want to
  benchmark.

Set it to roughly the unique-prefix size a real deployment carries per session
(e.g. 64–256). `0` (default) keeps pure dataset replay. Larger values fill L1
faster.

Measured properties of `Inferact/codex_swebenchpro_traces` (full 610-trace
dataset; cache-hit ratios use a char-count proxy, lengths tokenized with a
GLM-5.1 tokenizer — consistent with the dataset card's 94.2% self-report):

| metric | value |
|---|---|
| Aggregate ideal cache hit | **96.5 %** |
| Per-trial cache hit p10 / p50 / p90 | 91.2 % / 96.1 % / 98.1 % |
| LLM calls per trace | p50 30, mean 33, max 100 (20,230 total) |
| Mean prompt length | ~63K tokens |
| First-call prompt length | ~12.8K tokens |
| Generation length p50 | ~260 tokens |

The assistant turns in the dataset are length-preserving Lorem-ipsum
placeholders (Inferact redacted the generated text). For benchmarking this is
the right shape — prefill / decode token counts are the load-bearing
quantities; the actual decoded text is server-generated anyway.

### Interactive wait time

Between LLM calls the driver waits for the latency *outside* the model — tool
execution, or a human reviewing. The codex dataset is a **fully autonomous
agent** (every "human" turn is a bash/tool result, no real person) and it
stamps the real tool wall-time on each turn, so the driver replays those
directly instead of guessing:

- **Machine wait (between calls within a trace)** — uses the trace's
  **recorded tool wall-time** when present (~92% of gaps in the default
  dataset). Calls with no recorded time fall back to a simulated wait drawn
  from `--agent-wait-machine-secs` / `--agent-wait-jitter`.
- **Human wait (at trace boundaries)** — `--agent-wait-human-secs`, default
  **`0`** because the dataset is an autonomous batch. Raise it (e.g. 30–60s,
  jitter 1.0) only to model a human-supervised product where someone reviews
  and dispatches the next task.

```
--agent-wait-machine-secs  default 2.0   # FALLBACK only (calls with no recorded time)
--agent-wait-human-secs    default 0.0   # 0 = autonomous batch
--agent-wait-jitter        default 0.0   # CV of the simulated/fallback waits
--agent-wait-scale         default 1.0   # multiply the inter-call wait; 0 disables
```

**Scaling replay speed**: `--agent-wait-scale` multiplies whatever the
inter-call (machine) wait resolves to — recorded or fallback. `0` removes all
inter-call waits (max load), `0.5` runs replay at ~2× speed, `1.0` is true to
the trace.

**Simulated/fallback waits** are Gamma-distributed: `jitter=0` collapses to
exactly `mean`; `jitter=1.0` is the classic Poisson (exponential); `>1` is
long-tail. Floor 0.05 s machine / 1 s human; cap 300 s (recorded waits are
capped at 300 s as well).

```
sampled_wait ~ Gamma(shape = 1 / jitter², scale = mean × jitter²)
# E = mean, CV = jitter
```

**Recommended values**:

| Scenario | machine (fallback) | human | jitter | scale |
|---|---|---|---|---|
| Faithful replay (default) | 2.0 | 0 | 0.0 | 1.0 |
| Max load / stress (no waits) | 2.0 | 0 | 0.0 | 0.0 |
| Faster-than-real replay | 2.0 | 0 | 0.0 | 0.5 |
| Human-supervised product | 2.0 | 30–60 | 1.0 | 1.0 |

**Configured via YAML**: all four knobs live in the workload YAML. CLI flags
override the YAML; if neither is given, code defaults (2.0 / 0.0 / 0.0 / 1.0)
apply.

```yaml
# agent/workloads/<profile>.yaml
workload:
  # ... existing fields ...

  # Inter-call waits (dataset-replay mode). Machine wait prefers the trace's
  # recorded tool wall-time; these cover fallback gaps and scaling.
  agent_wait_machine_secs: 2.0     # fallback when no recorded wall-time
  agent_wait_human_secs:   0.0     # human review gap; 0 = autonomous
  agent_wait_jitter:       0.0     # CV; 0=deterministic, 1.0=Poisson, >1=long-tail
  agent_wait_scale:        1.0     # multiplier on the inter-call wait; 0 disables
  agent_session_salt_tokens: 0     # unique per-replay prefix (~N tok); 0=off, grows working set past L1
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
