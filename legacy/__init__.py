from pathlib import Path as _Path

from legacy import backends

REGISTERED_BACKENDS = {
    "together": backends.TogetherBackend,
    "openai": backends.OpenAIBackend,
    "vllm": backends.VllmBackend,
    "tgi": backends.TgiBackend,
    "fireworks": backends.FireworksBackend,
    "nvidia_nim": backends.NvidiaNIMBackend,
    "sglang": backends.SGLangBackend,
    "trtllm": backends.TRTLLMBackend,
    "anthropic": backends.AnthropicBackend,
    "embeddings": backends.OpenAIVectorBackend,
}


def configs_dir():
    """Filesystem path to the bundled batch-mode workload config YAMLs.

    Pass any of these to `agent-bench batch --config <path>` to skip writing
    your own YAML for a typical serving scenario.
    """
    return _Path(__file__).parent / "configs"


def list_configs():
    """Return the names of bundled batch-mode workload configs (no .yaml)."""
    return sorted(p.stem for p in configs_dir().glob("*.yaml"))
