"""Unified `agent-bench` entry point.

This tool is an **agent benchmark** — multi-turn growing-prefix workloads
against an inference endpoint.

Subcommand routing:

    agent-bench                  -> agent.agent_throughput.main   (default)
    agent-bench agent  …         -> agent.agent_throughput.main
    agent-bench sweep  …         -> agent.runner.main             (QPS sweep around agent)
    agent-bench viewer …         -> agent.viewer.main             (live dashboard)
"""
from __future__ import annotations

import sys
from typing import List, Sequence


SUBCOMMANDS = ("agent", "sweep", "viewer")


def _print_usage() -> None:
    print(
        "usage: agent-bench [{agent,sweep,viewer}] [args...]\n"
        "\n"
        "  (no subcommand)    same as `agent` — the default mode\n"
        "  agent              multi-turn agent workload with growing prefixes\n"
        "  sweep              QPS sweep / SLO capacity search wrapping `agent`\n"
        "  viewer             live Dash/Plotly dashboard over benchmarks/\n"
        "\n"
        "Run `agent-bench <subcommand> --help` for that subcommand's flags.",
        file=sys.stderr,
    )


def _dispatch(subcommand: str, rest: Sequence[str]) -> None:
    """Invoke the chosen subcommand's main() with `rest` as its argv."""
    # The mains read sys.argv directly. Rewrite it so they see only their own
    # args, then restore on the way out so re-invocation in-process stays clean
    # (matters for tests).
    saved_argv = sys.argv
    sys.argv = [f"agent-bench {subcommand}", *rest]
    try:
        if subcommand == "agent":
            from agent.agent_throughput import main as _main
        elif subcommand == "sweep":
            from agent.runner import main as _main
        elif subcommand == "viewer":
            from agent.viewer import main as _main
        else:  # unreachable; guarded by caller
            raise AssertionError(subcommand)
        _main()
    finally:
        sys.argv = saved_argv


def main(argv: List[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    # No args at all — default to `agent` mode rather than printing help. The
    # tool's whole point is the agent workload; `agent --help` is one step away.
    if not args:
        _dispatch("agent", [])
        return

    if args[0] in ("-h", "--help"):
        _print_usage()
        return

    head = args[0]
    if head in SUBCOMMANDS:
        _dispatch(head, args[1:])
        return

    # Bare flag form `agent-bench --server … --model …` with no subcommand —
    # treat the flags as `agent` args, since that's the default mode.
    if head.startswith("-"):
        _dispatch("agent", args)
        return

    print(f"agent-bench: unknown subcommand {head!r}", file=sys.stderr)
    _print_usage()
    sys.exit(2)


if __name__ == "__main__":
    main()
