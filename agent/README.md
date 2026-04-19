# agent-throughput

Throughput benchmark harness for LLM inference servers, modeled after an
interactive **code-agent** workload:

* Long shared **session prefixes** that grow turn by turn (codebase context,
  tool preamble, diffs, prior turns) — so a realistic fraction of each
  request is cache-hittable.
* Moderate generation length (~300–500 tokens per turn).
* Bursty arrivals (Poisson / Gamma inter-arrival), controllable QPS ramp +
  sustain, optional QPS sweep.

The tool sends requests against any OpenAI-compatible `/v1/chat/completions`
endpoint (sglang, vLLM, TGI, TRT-LLM, etc.) and reports end-to-end latency,
TTFT, cache hit rate, and ramp / sustain / drain **TPM** breakdowns.

---

## Setup

```bash
# Python ≥ 3.10 and these four wheels are all you need for the driver;
# `dash` / `plotly` are only required if you want to run the viewer.
pip install -r requirements.txt

# If your HF model / tokenizer cache lives on a data disk (not $HOME):
export HF_HUB_CACHE=/scratch/huggingface
export HF_HOME=/scratch/huggingface
export TRANSFORMERS_OFFLINE=1      # don't reach out to huggingface.co at runtime
```

The shipped `run*.sh` scripts already set these HF variables; you only
need the exports if you invoke `agent_throughput.py` / `runner.py`
directly and your HF cache isn't in the default `~/.cache/huggingface`.

Docker itself is only required for the sglang launch helpers — the
driver is a pure Python client.

---

## Files

```
agent_throughput.py     main load driver & simulator (sends requests, collects stats)
runner.py               QPS sweep wrapper — runs agent_throughput.py at a
                        series of target QPS values
viewer.py               Dash/Plotly viewer for post-run analysis (optional)

run_sglang_128k.sh      launches sglang in docker with 128K context
run.sh                  one-shot sweep against the 16K profile
run_128k.sh             one-shot single-QPS run against the 128K profile
run_viewer.sh           opens the viewer on a results directory

workloads/
  code_agent_16k.yaml   short-context code-agent workload (16K servers)
  code_agent_128k.yaml  long-context code-agent workload (128K servers)
  code_agent_200k.yaml  extra-long-context variant (200K servers)
```

---

## Quick start — sglang @ 128K + agent test

```bash
# 1. Launch the server (GPU 3, host port 8001, Qwen3-30B-A3B-NVFP4 @ 128K)
./run_sglang_128k.sh 3 8001
# wait for "The server is fired up and ready to roll!" — roughly 1–2 min
docker logs -f sglang-qwen3-nvfp4-128k   # in another shell, optional

# 2. Drive the code-agent test against it
./run_128k.sh http://localhost:8001 qwen3-30b-a3b-nvfp4

# 3. Optional: browse results in a web UI
./run_viewer.sh qps_sweep_results_code_agent 8050
#   then open http://<host>:8050
```

---

## Where is TPM in the output?

`agent_throughput.py` prints a **Phase Throughput Breakdown** table near the
end of each run — this is the source of truth for tokens-per-minute numbers:

```
Phase Throughput Breakdown (input TPM includes cache; uncached = actual prefill work):
  phase     dur(s)  reqs   qps    input TPM   cached TPM   uncached TPM     gen TPM  cache%  TTFT p50  TTFT p90
  -------------------------------------------------------------------------------------------------------------
  ramp        30.0    12  0.40      128,435       92,000         36,435      17,041   71.6%     42.1ms     51.0ms
  sustain    180.0    63  0.35      135,900      124,450         11,450      17,300   91.6%     43.0ms     50.0ms
  drain       19.0     8  0.42      170,000      160,000         10,000      16,200   94.1%     40.0ms     48.0ms
```

Columns:

| column          | meaning                                                            |
|-----------------|--------------------------------------------------------------------|
| `input TPM`     | total prompt tokens received by the server per minute, **including** cache hits |
| `cached TPM`    | portion served from the server's prefix cache (no compute)         |
| `uncached TPM`  | `input - cached` — the **real prefill work** the server performed  |
| `gen TPM`       | completion tokens per minute                                       |
| `cache%`        | `cached / input` over the phase                                    |

Rows:

* **`ramp`** — during the linear QPS ramp-up (initial_qps → max_qps). Cache
  is cold at the start, so expect lower cache% here.
* **`sustain`** — steady state at `max_qps`. **Quote this row as the server's
  capacity number.**
* **`drain`** — requests issued late in sustain that completed after the
  sustain window ended. If the drain window is too short or only caught a
  single request, the row reports `n/a` instead of a nonsense rate.

---

## Example run: Qwen3-30B-A3B-NVFP4 @ 128K, tp=2 on two B200

Launched with `./run_sglang_128k.sh 4,5 8001`
(GPU 4+5, same NUMA — see [Gotchas](#gotchas) for why that matters)
then driven by `./run_128k.sh http://localhost:8001 qwen3-30b-a3b-nvfp4`,
which uses `workloads/code_agent_128k.yaml` (mean prompt ≈ 40K, max prompt
120K, target max QPS 0.2, 45 s ramp, 300 s sustain).

Observed:

```
Total requests sent: 61
Completed: 61       Errors: 0       Success rate: 100%
Actual average QPS: 0.17 (target 0.20)
Actual benchmark duration: 349.1 s

Prompt length   : mean 37,249   p50 21,370   p90 82,965 tokens
Generation      : mean 372      p90 857 tokens (target 500)
TTFT            : p50 96.7 ms   p90 211.5 ms
Cache           : actual 87.2%  / ideal 92.6%  (efficiency 94.2%)

Phase Throughput Breakdown (input TPM includes cache; uncached = actual prefill work):
  phase     dur(s)  reqs   qps    input TPM   cached TPM   uncached TPM     gen TPM  cache%  TTFT p50  TTFT p90
  -------------------------------------------------------------------------------------------------------------
  ramp        45.0    11  0.24      170,320      100,352         69,968       5,027   58.9%    64.9ms    206.9ms
  sustain    300.0    49  0.16      407,176      359,910         47,266       3,740   88.4%   101.5ms    278.5ms
  drain        n/a     1   n/a          n/a          n/a            n/a         n/a     n/a       n/a       n/a
```

How to read the **sustain** row (the capacity number):

* Server ingested **407 K input tokens / min**; 360 K / min served from
  prefix cache, only **47 K / min of real prefill compute**. Cache is doing
  8.5 × of the prefill work.
* Generation output **3.7 K tokens / min** — small because long prompts
  (37 K mean) saturate the server before generation can scale.
* TTFT p50 ≈ 100 ms, p90 ≈ 280 ms — acceptable for this prompt length, and
  shows the tail is in-distribution (no pathological outliers).
* Cache hit is 88.4 % actual vs 92.6 % ideal (what the session model would
  produce with zero eviction) → **cache efficiency 94 %**, meaning the
  server holds onto nearly everything the workload re-uses.

Ramp row is for debugging cold-cache behavior: lower cache% and less input
TPM are expected while sessions first warm up. Don't quote ramp numbers as
capacity — always use **sustain**.

---

## Workload profiles

Each YAML under `workloads/` defines one profile. Key parameters:

| parameter              | meaning                                               |
|------------------------|-------------------------------------------------------|
| `system_prompt_len`    | tokens for synthetic tool / style preamble           |
| `initial_prefix_mean`  | mean opening context per new session                 |
| `new_tokens_mean`      | mean delta added per turn                            |
| `max_prompt_tokens`    | session retires after prefix grows past this        |
| `initial_qps`/`max_qps`| QPS at t=0 and after ramp                            |
| `ramp_duration`        | seconds to linearly ramp to `max_qps`                |
| `sustain_duration`     | seconds to hold at `max_qps`                         |
| `max_inflight`         | backpressure: pause sending if more than N in-flight |
| `poisson` + `poisson_shape` | Gamma inter-arrival shape (1 = exponential, higher = smoother) |
| `tokenizer`            | HF model id or local path (for token counting)       |

Match `max_inflight` to the server's `--max-running-requests`, and keep
`max_prompt_tokens` well under the server's `--context-length` to leave
room for the generated tokens.

### `session_decay_lambda` — how much recency bias?

When the driver picks which existing session gets the next request, each
candidate is weighted by `exp(-lambda * seconds_since_last_use)`. Bigger
lambda means the workload piles on the freshest sessions (tight locality,
higher cache hit); smaller lambda spreads the traffic and evicts more
aggressively. Rough translation of lambda to half-life
(`ln(2) / lambda`):

| lambda | half-life |
|--------|-----------|
| 0.001  | ~11 min   |
| 0.005  | ~2.3 min  |
| 0.01   | ~70 s     |
| 0.02   | ~35 s     |
| 0.05   | ~14 s     |
| 0.1    | ~7 s      |

The `code_agent_*.yaml` profiles default to 0.005–0.006, which keeps a
handful of sessions hot for a couple of minutes — the regime where a real
code-agent user stays mid-task.

---

## Swapping in a different model

Three moving parts have to stay aligned when you change the served model:
the **server** launch flags, the **tokenizer** used by the driver, and the
**context budget** in the workload YAML. Miss one and you get silent bad
numbers.

### 1. Server side (sglang launch)

```
--model-path               new local path, e.g.
                           /hf/hub/models--<org>--<name>/snapshots/<hash>
--served-model-name        the id the driver's --model flag must match
--context-length           ≤ config.max_position_embeddings (or set
                           SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
--tp                       TP size — scale up for bigger models; memory
                           per GPU ≈ weights/tp + per-GPU KV share
--mem-fraction-static      raise to 0.9+ for dense models to squeeze out
                           more KV; lower if OOM during cuda-graph capture
--max-running-requests     your target concurrency cap
--enable-cache-report      keep this on, otherwise cache hit reads 0%
```

For quantized models sglang usually auto-detects from `config.json`'s
`quantization_config`. If it doesn't, pass `--quantization modelopt_fp4`,
`--quantization fp8`, etc. explicitly. For a dtype override (rare) use
`--dtype bfloat16` / `--kv-cache-dtype fp8_e5m2`.

### 2. Workload YAML — the tokenizer is load-bearing

Under `workloads/*.yaml` update at minimum:

```yaml
workload:
  tokenizer:     Qwen/Qwen3-30B-A3B              # or a local path
  max_prompt_tokens: 120000                       # < server --context-length
  gpus:          2                                # for per-GPU TPM display
  max_inflight:  16                               # = --max-running-requests
```

The tokenizer **must** match what the server is doing. If you point sim
at a DeepSeek tokenizer while the server runs Qwen, every prompt length
the driver ships will miscount — your "initial prefix 40K" will show up
on the server as a very different token count, cache pages will land on
different boundaries than expected, and the `Ideal cache hit rate`
number becomes fiction. The driver accepts either a HuggingFace hub id
(resolved from the local cache with `HF_HUB_OFFLINE=1`) or an absolute
local directory that contains `tokenizer.json` / `tokenizer_config.json`.

Parameters you should also review when moving to a very different model:

| parameter              | why it might need to change                          |
|------------------------|------------------------------------------------------|
| `system_prompt_len`    | target depends on the model's typical tool preamble  |
| `initial_prefix_mean`  | scale with model context capacity                    |
| `new_tokens_mean`      | roughly: different tokenizers split code differently |
| `generation_length_mean` | reasoning models emit far more tokens than chat    |
| `acc_len`, `mtp_*`     | only meaningful if the server has speculative decode |

### 3. Driver CLI

```
python3 agent_throughput.py \
  --model           <same string as --served-model-name on the server> \
  --tokenizer       <overrides YAML if given> \
  --workload-config workloads/<profile>.yaml \
  ...
```

Model id mismatches surface as `HTTP 400 model not found`; tokenizer
mismatches are silent, so verify by sending one handcrafted prompt of
known length and checking the server's `usage.prompt_tokens` equals
what the driver planned.

### 4. Smoke check before a full sweep

```bash
curl -sS http://<host>:<port>/v1/models              # model id matches?
curl -sS http://<host>:<port>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],
       "max_tokens":8,"stream":false}'               # 200 + sensible completion?
```

If both look clean, run `agent_throughput.py` with
`--sustain-duration 30 --max-qps 0.2` once to make sure requests flow
end-to-end. Then launch the real run.

---

## How the session model produces cache hits

Each synthetic session is a stack that grows turn-by-turn:

```
New session   : prefix = system_prompt ⧺ initial_random_codebase_chunk
Each turn     : 1. choose a session (new with prob. new_session_rate,
                   otherwise existing via recency weighting)
                2. draw `new_tokens` from lognormal(mean, median)
                3. prompt = session.prefix ⧺ new_content
                4. ship the prompt, stream the reply
                5. on success, session.prefix ⧺= new_content
                6. if session.prefix >= max_prompt_tokens, retire it
```

Because the prefix is reused, the **ideal** cache hit for a request is
`prefix_tokens / total_prompt_tokens`. A turn with an 80 K prefix and 8 K
of new content has an ideal hit rate of 91 %. The server's **actual** hit
rate drops below that when its cache evicts live prefixes — the gap
between ideal and actual is the eviction tax.

Two knobs dominate the emergent hit rate:

1. `new_session_rate` — fewer new sessions means more reuse → higher hits.
2. `new_tokens_mean` vs. `initial_prefix_mean` — if each turn adds little
   and the opening context is large, most of every prompt is cacheable.

### Arrival shape

Requests are sent on a schedule set by the current target QPS (linear
ramp from `initial_qps` to `max_qps`, then flat `sustain_duration`
seconds). Inter-arrival gaps come from a Gamma distribution with shape
`poisson_shape`: 1 reduces to an exponential (classic Poisson, burstiest),
higher shapes smooth toward a uniform spacing.

### Backpressure

Set `max_inflight` to an integer and the driver will stop sending while
the in-flight count is at that ceiling. That lets you set an aspirational
`max_qps` and let the actual rate self-clip at server capacity; the
reported `Actual average QPS` in the summary tells you what the server
could actually absorb.

---

## Tuning the workload

**To match a real production distribution of prompt lengths:** fit
`initial_prefix_{mean, median}` to the p50/mean of session-opening
prompts, and `new_tokens_{mean, median}` to the per-turn growth. Run and
check the reported `Actual Prompt Length Distribution` block against
production — iterate.

**To find a server's sustainable throughput:** set `max_qps` well above
what you expect, cap `max_inflight` to the server's concurrency, and read
the `sustain` row of the phase breakdown. The `Actual average QPS` is
what the server is actually willing to drain.

**To raise the emergent cache hit rate:** shrink `new_tokens_mean`, grow
`initial_prefix_mean`, or drop `new_session_rate`. The ceiling is
`prefix / (prefix + new)` — so the ratio of new to prefix is what
matters.

**For apples-to-apples comparisons across server builds or engine flags:**
pin `random_seed`. With the same seed, the same session history, prompt
lengths, request timestamps, and session-selection sequence get
replayed — any metric delta is server-side.

---

## Metric reference

### Input TPM (the main rate you quote)

```
input_tpm      = sum(prompt_tokens)  / duration * 60
cached_tpm     = sum(cached_tokens)  / duration * 60
uncached_tpm   = input_tpm - cached_tpm                 # actual prefill compute
gen_tpm        = sum(completion_tokens) / duration * 60
```

`prompt_tokens` and `cached_tokens` are whatever the server reports in
`usage` (both must be enabled — see [Gotchas](#gotchas)). When the server
returns `cached_tokens = 0`, `uncached_tpm` collapses into `input_tpm`
and the note under **Cache Statistics** in the summary tells you the
server isn't reporting hits.

### Per-request generation TPS

```
gen_tps      = completion_tokens / generation_time
gen_tps_mtp  = (completion_tokens * acc_len) / (generation_time * mtp_overhead_factor)
```

`generation_time` is wall-clock from the first streamed token to the
last. Samples with `generation_time < 10 ms` are dropped — they come from
network-side chunk batching rather than real decode work, and including
them would add noisy triple-digit spikes to the average.

### Windowed readings in the live dashboard line

The `Prefill:  X tok/s (1s) | Y tok/s (30s)` line is two rolling windows
over sent-tokens (same as `input_tpm / 60`, not `uncached_tpm`). Use the
30 s window for a smoothed steady-state view while the run is ongoing;
once the run ends, use the **Phase Throughput Breakdown** table instead.

---

## Gotchas

### 1. sglang drops `usage` in streams by default

When the client sends `stream: true` WITHOUT `stream_options.include_usage:
true`, sglang's OpenAI-compatible endpoint sets `"usage": null` on every
chunk — no `prompt_tokens`, no cached token count. `agent_throughput.py`
already includes the option; if you adapt it for other tools, keep this in
mind.

### 2. sglang cache reporting needs a launch flag

Launch sglang with `--enable-cache-report` to populate
`usage.prompt_tokens_details.cached_tokens` on cache hits. Without the
flag, `Actual cache hit rate` will read 0% even when the server is
caching. `agent_throughput.py` prints a warning in that case.

### 3. Qwen3-30B-A3B past 40K needs override

Qwen3-30B-A3B's native RoPE only covers 40K. To serve at 128K, set
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` in the container env (already
done in `run_sglang_128k.sh`).

---

## Manual usage

One-shot single run:

```bash
python3 agent_throughput.py \
  --server http://localhost:8001 \
  --model  qwen3-30b-a3b-nvfp4 \
  --workload-config workloads/code_agent_128k.yaml \
  --max-qps 0.3 --ramp-duration 45 --sustain-duration 300 \
  --name my-run --data-dir ./benchmarks
```

QPS sweep:

```bash
python3 runner.py \
  --name my-sweep \
  --workload-config workloads/code_agent_16k.yaml \
  --start-qps 0.2 --end-qps 1.0 --step 0.2 \
  --sustain-duration 300 \
  --max-inflight 16 \
  --server http://localhost:8001 \
  --model  qwen3-30b-a3b-nvfp4 \
  --results-dir qps_sweep_results_code_agent
```

`runner.py` seeds each QPS point with a distinct random seed (derived from
the run's wall-clock timestamp) so the server's prefix cache doesn't
illegally carry warm-up work between points.

### Preview mode (no HTTP)

Pass `--mode preview` to `agent_throughput.py` to exercise the full
scheduling + prompt-building + tokenization pipeline **without actually
hitting the server**. Handy for:

* sanity-checking a new workload YAML (the planned prompt / generation
  distributions are printed),
* measuring pure client-side overhead before pointing at a production
  server.

### Realistic mode (response-driven arrivals)

`--mode realistic` replaces the open-loop QPS schedule with
session-coupled arrivals: a session sends its next turn only after the
previous reply lands, plus a sampled "think time" delay. Useful when you
want to model a fixed population of concurrent users rather than hitting
a target arrivals rate. Requires `think_time_*`, `session_lifetime_*`,
`max_sessions`, and `session_abandon_rate` in the YAML — see the flags in
`agent_throughput.py --help`.

---

## Output layout

A single run with `--name X --data-dir benchmarks` writes:

```
benchmarks/X/YYYY-MM-DD-HH-MM-SS/
  metadata.json      # resolved config for the run
  metrics.jsonl      # per-second rolling TPM / TPS / cache / in-flight /
                     # session stats — each record now carries a `phase`
                     # field (ramp | sustain | drain) so the viewer can
                     # filter for steady-state without timestamp math
  summary.json       # machine-readable final summary: overall + phase-by-phase
                     # (input / cached / uncached / visible_gen / reasoning TPM,
                     #  TTFT percentiles, cache efficiency, context snapshot)
```

A sweep via `runner.py --name Y --results-dir Z` adds a layer above:

```
Z/Y_YYYYMMDD_HHMMSS/
  test_config.json       # sweep-level config (QPS values, seeds per point, etc.)
  original_workload.yaml # the workload YAML you passed in
  temp_config.yaml       # last per-point YAML with overrides applied
  results.json           # per-point status / elapsed / summary_json pointer
  results.csv            # flat table (one row per point) — open in pandas/Excel
  summary.md             # human-readable markdown table of sustain metrics
  summary.txt            # legacy one-liner per QPS (kept for grep)
  qps_<q>_output.log     # full console output of each single-QPS run
```

The viewer reads `benchmarks/*/metrics.jsonl`; `summary.json` (per run)
and `results.csv` / `summary.md` (per sweep) are what downstream scripts
(CI, pandas notebooks) should key on instead of parsing console logs.

### SLO-driven auto capacity search

Instead of a manual linear sweep you can let the runner binary-search for
the highest QPS that still clears a TTFT / success-rate budget:

```bash
python3 runner.py \
  --name capacity \
  --workload-config workloads/code_agent_128k.yaml \
  --auto-search \
  --slo-ttft-p90-ms   400 \
  --slo-success-rate  0.99 \
  --auto-min-qps 0.1 --auto-max-qps 2.0 \
  --auto-tolerance 0.1 --auto-max-probes 8 \
  --sustain-duration 180 --max-inflight 16 \
  --server http://localhost:8001 --model qwen3-30b-a3b-nvfp4 \
  --results-dir qps_sweep_results_code_agent
```

The search first probes the two endpoints of `[auto-min-qps, auto-max-qps]`:

* if the floor already violates the SLO it bails (server can't meet it),
* if the ceiling already passes it reports "capacity ≥ ceiling",
* otherwise it bisects until the range narrows below `--auto-tolerance`
  or `--auto-max-probes` runs have been spent.

Every probe still produces a full per-run `summary.json`, and the sweep
artifacts (`results.csv`, `summary.md`) get rewritten incrementally so
you can inspect partial progress while the search is running.

---

## Using a non-sglang backend

The driver speaks plain OpenAI `/v1/chat/completions` with streaming plus
`stream_options.include_usage: true`, so any server that honors that
contract works. Two things to double-check per backend:

* **Where it puts `cached_tokens`** — OpenAI-style servers report it at
  `usage.prompt_tokens_details.cached_tokens`; Anthropic-style servers
  put it at `usage.cache_read_input_tokens`. The driver reads both.
  vLLM exposes it only when `--enable-prefix-caching` is on and the
  request includes `cache_salt`; TGI does not report cached_tokens via
  the OpenAI route at all (check `/metrics` instead).
* **Whether it emits a final usage chunk** — if the last streamed chunk
  has `usage == null`, the driver falls back to the *planned* prompt
  length and logs
  `INFO: Server not returning prompt_tokens`. Flip on whatever
  per-server flag is the equivalent of sglang's `--enable-cache-report`.
