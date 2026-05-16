# Tests

Pytest tests for the fixes added on `fix/functional-bugs`.

```bash
# from the repo root, in a venv with lightrace installed:
pip install pytest pyyaml
pytest tests/ -v
```

Each file targets one slice:

| file | what it covers |
|---|---|
| `test_anthropic_backend.py` | Anthropic Messages API: request body shape, system-hoisting, `cache_control` marker placement, prompt-split when `cacheable_prefix` set, SSE event parsing (message_start / content_block_delta / message_delta / non-streaming / error / unknown) |
| `test_openai_cache_parsing.py` | OpenAI / sglang / vllm decode reads `usage.prompt_tokens_details.cached_tokens` → `FragmentInfo.cached_input_tokens` |
| `test_ideal_cache_rate.py` | All branches of `estimate_ideal_cache_hit_rate`: `same_prompts_in_burst`, `synthetic_cached_input_length`, `generated-shared-prefix`, Anthropic 1024-token minimum, shapes that return None |
| `test_analytics_per_device.py` | `_compute_per_device` burst path AND fallback path (the concurrent/qps `Per-GPU = 0.00` bug) |
| `test_cache_aggregation.py` | `_extract_cache_hit_rate` mean + None handling; `_sum_cached_input_tokens` aggregation |
| `test_sglang_compat.py` | Vendored `get_tokenizer` (local-path detection, repo-id fallthrough) + `sample_random_requests` (count, length, error paths) |
| `test_input_pipeline_hf.py` | HF reader with `chat=true` auto-wraps plain-text columns (was a `JSONDecodeError` crash) |
| `test_package_layout.py` | `agent/` and `lightrace/configs/` are accessible after install; YAML presets parse and declare a provider; `anthropic` is registered |

These tests don't need network access or a real inference server — they exercise the pure-Python code paths via fakes/mocks. The wandb client integration and aiohttp HTTP loop are intentionally out of scope.
