from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lightrace",
    version="0.0.1",
    description="LightRace Inference Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["lightrace", "lightrace.*", "agent", "agent.*"]),
    # Bundle the workload YAMLs so `pip install lightrace` (no git clone) still
    # ships them — addresses the wheel-only-has-batch bug.
    package_data={
        "agent": ["workloads/*.yaml"],
        "lightrace": ["configs/*.yaml"],
    },
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # NOTE: pin transformers to a version that handles local-filesystem
        # tokenizer paths well; 4.56.2 + huggingface_hub 0.36 reject abs paths
        # via validate_repo_id. Bump if you hit that.
        "transformers>=4.56.2",
        "orjson",
        "datasets>=4.0",
        "aiohttp>=3.10",
        "jsonargparse",
        "tabulate",
        "together",
        "Jinja2>=3.1.0",
        "wandb",
        "pybase64",
        # Previously: sglang==0.5.2. Removed — the two helpers we used
        # (get_tokenizer, sample_random_requests) are now vendored in
        # lightrace/_sglang_compat.py to avoid downgrading the server's sglang.
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "mypy",
            "pytest",
            "pytest-xdist",
            "pytest-asyncio",
            "pytest-aiohttp",
            "types-tabulate",
            "httpx",
        ],
    },
    entry_points={
        "console_scripts": [
            "lightrace = lightrace.run:main",
            # Agent mode (multi-turn / code-agent workload). Was previously
            # only reachable via `python3 agent/agent_throughput.py` from a
            # git clone — now installable via pip too.
            "lightrace-agent = agent.agent_throughput:main",
            "lightrace-agent-sweep = agent.runner:main",
        ],
    },
)
