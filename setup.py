from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lightrace",
    version="0.0.1",
    description="LightRace Inference Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["lightrace", "lightrace.*"]),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "transformers==4.56.2",
        "orjson",
        "datasets==4.1.1",
        "aiohttp==3.12.15",
        "jsonargparse",
        "tabulate",
        "together",
        "Jinja2>=3.1.0",
        "wandb",
        "sglang==0.5.2",
        "pybase64",
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
        ],
    },
)
