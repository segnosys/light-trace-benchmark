"""
agent-throughput driver and code-agent workload profiles.

This subpackage ships as part of the lightrace wheel so `pip install lightrace`
gets you both batch (`lightrace`) and agent (`lightrace-agent`) entry points.
The workload YAMLs live next to this file under `workloads/`; resolve them
with `agent.workloads_dir()` or by `--workload-config <abs path>`.
"""
from pathlib import Path


def workloads_dir() -> Path:
    """Filesystem path to the bundled workload YAML directory."""
    return Path(__file__).parent / "workloads"


def list_workloads():
    """Return the names of bundled workload YAML files (no path, no .yaml)."""
    return sorted(p.stem for p in workloads_dir().glob("*.yaml"))
