"""
Tests for the wheel-packaging fixes: agent/ and lightrace/configs/ should
ship as part of the installed package, and the helper accessors should
return paths that actually exist on disk.
"""
import os

import pytest

import agent
import lightrace


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


def test_configs_dir_exists():
    p = lightrace.configs_dir()
    assert p.exists() and p.is_dir(), f"missing dir: {p}"


def test_list_configs_returns_known_presets():
    names = set(lightrace.list_configs())
    must_have = {
        "chat_short",
        "rag_doc_qa",
        "code_completion",
        "reasoning_long_decode",
        "long_prefill_ttft",
        "pure_cold_random",
        "hf_math500_reasoning",
        "hf_gsm8k",
        "hf_humaneval",
        "prefix_cache_80pct",
        "sharegpt_chat",
        "jsonl_template",
        "anthropic_cache_demo",
        "sglang_cache_report",
    }
    assert must_have <= names, f"missing configs: {must_have - names}"


def test_config_yamls_are_parseable():
    """Each shipped config must parse as YAML — silently corrupt YAML hurts."""
    yaml = pytest.importorskip("yaml")
    for cfg_path in lightrace.configs_dir().glob("*.yaml"):
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{cfg_path.name} did not parse to dict"
        # Every preset must at least declare a provider.
        assert "provider" in data, f"{cfg_path.name} missing 'provider'"


def test_anthropic_backend_registered():
    """Bug #9 dataset entry — make sure the new provider is wired."""
    assert "anthropic" in lightrace.REGISTERED_BACKENDS
