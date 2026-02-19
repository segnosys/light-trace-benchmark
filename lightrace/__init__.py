from lightrace import backends

REGISTERED_BACKENDS = {
    "together": backends.TogetherBackend,
    "openai": backends.OpenAIBackend,
    "vllm": backends.VllmBackend,
    "tgi": backends.TgiBackend,
    "fireworks": backends.FireworksBackend,
    "nvidia_nim": backends.NvidiaNIMBackend,
    "sglang": backends.SGLangBackend,
    "trtllm": backends.TRTLLMBackend,
    "embeddings": backends.OpenAIVectorBackend,
}
