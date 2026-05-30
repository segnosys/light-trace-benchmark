"""
Tests for the wheel-packaging fixes: agent/workloads/ should ship as part of
the installed package, and the helper accessors should return paths that
actually exist on disk.
"""

import agent


def test_workloads_dir_exists():
    p = agent.workloads_dir()
    assert p.exists() and p.is_dir(), f"missing dir: {p}"


def test_list_workloads_returns_known_profiles():
    names = set(agent.list_workloads())
    # A few profiles we ship; if any of these vanish, packaging probably broke.
    must_have = {
        "code_agent_16k",
        "code_agent_128k",
        "code_agent_50k_cache90_kimi",
        "chat_assistant_short",
        "rag_oneshot",
    }
    assert must_have <= names, f"missing workloads: {must_have - names}"


def test_workload_yaml_files_are_nonempty():
    p = agent.workloads_dir()
    for yaml_path in p.glob("*.yaml"):
        assert yaml_path.stat().st_size > 0, f"empty workload yaml: {yaml_path}"
