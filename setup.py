from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agent-bench",
    version="0.0.1",
    description=(
        "agent-bench: agent workload benchmarking for inference endpoints. "
        "Multi-turn growing-prefix sessions against an inference endpoint."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Two Python packages ship: `agent` (driver + workloads) and `agentbench`
    # (top-level CLI dispatcher).
    packages=find_packages(include=[
        "agent", "agent.*",
        "agentbench", "agentbench.*",
    ]),
    # Bundle workload YAMLs so `pip install agent-bench` (no git clone) still
    # ships them.
    package_data={
        "agent": ["workloads/*.yaml"],
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
    ],
    extras_require={
        # Optional Dash/Plotly viewer for agent-mode runs. Not pulled into
        # the base install so a `pip install agent-bench` of just the bench
        # client stays small. Install with `pip install 'agent-bench[viewer]'`.
        "viewer": [
            "dash>=2.0.0",
            "plotly>=5.0.0",
        ],
        "dev": [
            "pre-commit",
            "mypy",
            "pytest",
            "pytest-xdist",
            "pytest-asyncio",
            "pytest-aiohttp",
            "types-tabulate",
            "httpx",
            "ruff",
        ],
    },
    entry_points={
        "console_scripts": [
            # Single unified entry point — dispatches to subcommands:
            #   agent-bench                 default → agent mode
            #   agent-bench agent  …        multi-turn growing-prefix workload
            #   agent-bench sweep  …        QPS sweep / SLO capacity search
            #   agent-bench viewer …        live Dash/Plotly dashboard
            "agent-bench = agentbench.cli:main",
        ],
    },
)
