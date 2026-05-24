"""Tests for the `trace-replay` dataset_type added to `legacy.input_pipeline`.

The point of `trace-replay` is to consume recorded sharegpt-style
conversation traces (e.g. HuggingFace's
`Inferact/codex_swebenchpro_traces`) and expand each trial into one
request per user turn with growing prefix — so cross-restart KV-cache
experiments can replay bit-identical prompts without depending on
model output reproducibility.

These tests pin the contract:

- Role mapping `human` → `user`, `gpt` → `assistant`.
- Strict alternation; trailing assistant turns trimmed so every
  emitted request ends on `user`.
- Each trial expands to K requests where K = user-turn count, with
  the K-th request carrying the first K user / assistant pairs.
- Deterministic across runs given (file, n, strategy, max_turns, seed).
- Selection strategies (smallest / largest / random / first) work
  as documented.
- Malformed trials (multimodal, assistant-led, unknown role) are
  skipped without crashing.
"""

import json
import tempfile

import pytest

from legacy.input_pipeline import (
    InputPipeline,
    _cap_user_turns,
    _select_trials_by_strategy,
    _trial_to_messages,
)


# ----------------------------------------------------------------------
# Fixtures — synthetic trials in dataset-native shape
# ----------------------------------------------------------------------


def _trial(*pairs):
    """Build a `{"conversations": [...]}` trial in native shape."""
    return {
        "conversations": [{"from": role, "value": text} for role, text in pairs]
    }


@pytest.fixture
def alternating_trial():
    return _trial(
        ("human", "u1"),
        ("gpt", "a1"),
        ("human", "u2"),
        ("gpt", "a2"),
        ("human", "u3"),
    )


@pytest.fixture
def trial_file(tmp_path, alternating_trial):
    """Write a 3-trial corpus to a temp JSON file and return the path."""
    corpus = [
        alternating_trial,
        _trial(("human", "single")),
        _trial(("human", "x"), ("gpt", "y"), ("human", "z")),
    ]
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(corpus))
    return p


# ----------------------------------------------------------------------
# _trial_to_messages — schema parsing
# ----------------------------------------------------------------------


def test_trial_to_messages_alternates(alternating_trial):
    msgs = _trial_to_messages(alternating_trial)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]


def test_trial_to_messages_trims_trailing_assistant():
    msgs = _trial_to_messages(_trial(("human", "u"), ("gpt", "a")))
    # Trailing assistant must be dropped so we always send "user-last".
    assert [m["role"] for m in msgs] == ["user"]


def test_trial_to_messages_rejects_empty():
    assert _trial_to_messages({"conversations": []}) is None
    assert _trial_to_messages({}) is None
    assert _trial_to_messages(None) is None


def test_trial_to_messages_rejects_assistant_first():
    """A chat history that starts with an assistant turn can't be
    sent as-is to most chat templates. Drop the trial."""
    assert _trial_to_messages(_trial(("gpt", "hello"))) is None


def test_trial_to_messages_rejects_multimodal_value():
    """Non-string `value` (tool calls, images) — refuse loudly via
    None rather than coerce to garbage."""
    trial = {"conversations": [{"from": "human", "value": {"type": "image"}}]}
    assert _trial_to_messages(trial) is None


def test_trial_to_messages_rejects_unknown_role():
    """If a future dataset version introduces `system` / `tool` /
    `function` roles, we want a clear regression signal, not a silent
    coercion."""
    assert _trial_to_messages(_trial(("system", "be helpful"), ("human", "x"))) is None


# ----------------------------------------------------------------------
# _cap_user_turns
# ----------------------------------------------------------------------


def test_cap_user_turns_truncates(alternating_trial):
    msgs = _trial_to_messages(alternating_trial)
    capped = _cap_user_turns(msgs, max_user_turns=2)
    assert [m["role"] for m in capped] == ["user", "assistant", "user"]


def test_cap_user_turns_zero_means_no_cap(alternating_trial):
    msgs = _trial_to_messages(alternating_trial)
    assert _cap_user_turns(msgs, max_user_turns=0) == msgs


def test_cap_user_turns_ends_on_user(alternating_trial):
    msgs = _trial_to_messages(alternating_trial)
    capped = _cap_user_turns(msgs, max_user_turns=2)
    assert capped[-1]["role"] == "user"


# ----------------------------------------------------------------------
# _select_trials_by_strategy
# ----------------------------------------------------------------------


def _msgs(byte_size: int) -> list:
    return [{"role": "user", "content": "u" * byte_size}]


def test_select_smallest():
    trials = [_msgs(100), _msgs(300), _msgs(200)]
    picked = _select_trials_by_strategy(trials, n=2, strategy="smallest", seed=0)
    assert [len(m[0]["content"]) for m in picked] == [100, 200]


def test_select_largest():
    trials = [_msgs(100), _msgs(300), _msgs(200)]
    picked = _select_trials_by_strategy(trials, n=2, strategy="largest", seed=0)
    assert [len(m[0]["content"]) for m in picked] == [300, 200]


def test_select_first_preserves_order():
    trials = [_msgs(100), _msgs(300), _msgs(200)]
    picked = _select_trials_by_strategy(trials, n=2, strategy="first", seed=0)
    assert [len(m[0]["content"]) for m in picked] == [100, 300]


def test_select_random_is_seeded():
    trials = [_msgs(i) for i in range(1, 11)]
    a = _select_trials_by_strategy(trials, n=5, strategy="random", seed=42)
    b = _select_trials_by_strategy(trials, n=5, strategy="random", seed=42)
    assert a == b


def test_select_unknown_strategy_raises():
    with pytest.raises(ValueError, match="trace_replay_strategy"):
        _select_trials_by_strategy([_msgs(1)], n=1, strategy="???", seed=0)


def test_select_n_none_returns_all():
    trials = [_msgs(i) for i in (1, 2, 3, 4)]
    picked = _select_trials_by_strategy(trials, n=None, strategy="first", seed=0)
    assert len(picked) == 4


# ----------------------------------------------------------------------
# End-to-end via InputPipeline.prepare_inputs()
# ----------------------------------------------------------------------


def _make_pipeline(trial_file, **overrides):
    """Build a pipeline configured for trace-replay against `trial_file`."""
    defaults = dict(
        model_name="x",
        dataset_type="trace-replay",
        stream=True,
        max_tokens=1,
        skip_eos=False,
        temperature=0.0,
        top_p=None,
        chat=True,
        num_examples=None,  # auto-set from the corpus
        tokenizer_name=None,
        trace_replay_input_path=str(trial_file),
        trace_replay_num_trials=None,
        trace_replay_strategy="first",
        trace_replay_max_turns_per_trial=0,
        trace_replay_seed=42,
    )
    defaults.update(overrides)
    return InputPipeline(**defaults)


def test_prepare_inputs_expands_trial_to_growing_prefix(trial_file):
    """The alternating_trial in trial_file[0] has 3 user turns →
    3 requests with growing prefix. Trial[1] has 1 user turn → 1
    request. Trial[2] has 2 user turns → 2 requests. Total 6."""
    pipe = _make_pipeline(trial_file)
    out = pipe.prepare_inputs()
    assert len(out) == 6
    # All requests are chat-mode (messages, not prompt).
    for req in out:
        assert req.messages is not None
        assert req.prompt is None
        # Every request ends on a user turn.
        assert req.messages[-1]["role"] == "user"


def test_prepare_inputs_growing_prefix_is_a_real_prefix(trial_file):
    """The K-th request from a trial must be a prefix of the (K+1)-th
    request from the same trial. This is the invariant that makes
    KV-cache reuse possible — without it the engine sees brand-new
    prompts and prefix caching collapses."""
    pipe = _make_pipeline(trial_file)
    out = pipe.prepare_inputs()
    # alternating_trial → out[0..2]: 1 msg, 3 msgs, 5 msgs
    assert len(out[0].messages) == 1
    assert len(out[1].messages) == 3
    assert len(out[2].messages) == 5
    # Prefix check:
    assert out[1].messages[: len(out[0].messages)] == out[0].messages
    assert out[2].messages[: len(out[1].messages)] == out[1].messages


def test_prepare_inputs_num_examples_auto_set(trial_file):
    """`--num_examples` is informational for trace-replay; the count
    is determined by the corpus. Auto-assignment prevents the
    `assert len(all_requests) == self.num_examples` check from
    rejecting valid runs."""
    pipe = _make_pipeline(trial_file)
    out = pipe.prepare_inputs()
    assert pipe.num_examples == len(out)


def test_prepare_inputs_num_examples_explicit_mismatch_raises(trial_file):
    """If the user pinned --num_examples and the corpus produced a
    different count, the existing assertion fires — same contract
    as every other mode."""
    pipe = _make_pipeline(trial_file, num_examples=99)
    with pytest.raises(AssertionError):
        pipe.prepare_inputs()


def test_prepare_inputs_num_trials_limits(trial_file):
    """num_trials=1 should pick the first trial only (3 requests
    for the alternating_trial when strategy=first)."""
    pipe = _make_pipeline(trial_file, trace_replay_num_trials=1)
    out = pipe.prepare_inputs()
    assert len(out) == 3


def test_prepare_inputs_max_turns_per_trial(trial_file):
    """Capping user turns to 2 should shave alternating_trial down
    to 2 requests (5 msgs total) — net trial1 1 + trial2 2 + trial0
    capped 2 = 5 total."""
    pipe = _make_pipeline(trial_file, trace_replay_max_turns_per_trial=2)
    out = pipe.prepare_inputs()
    # alternating_trial (capped to 2 user turns): 2 requests
    # single trial (1 user turn): 1 request
    # last trial (2 user turns, no cap effect): 2 requests
    assert len(out) == 5


def test_prepare_inputs_requires_input_path():
    pipe = _make_pipeline(trial_file=None, trace_replay_input_path=None)
    with pytest.raises(ValueError, match="trace_replay_input_path"):
        pipe.prepare_inputs()


def test_prepare_inputs_requires_chat_mode(trial_file):
    pipe = _make_pipeline(trial_file, chat=False)
    with pytest.raises(ValueError, match="chat=true"):
        pipe.prepare_inputs()


def test_prepare_inputs_rejects_non_list_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"trials": []}))  # dict, not list
    pipe = _make_pipeline(p)
    with pytest.raises(ValueError, match="JSON list of trials"):
        pipe.prepare_inputs()


def test_prepare_inputs_is_deterministic(trial_file):
    """Same (file, strategy, seed, max_turns) → same requests, bit
    for bit. This is the property the synthetic modes can't promise
    for cross-restart KV experiments."""
    a = _make_pipeline(trial_file).prepare_inputs()
    b = _make_pipeline(trial_file).prepare_inputs()
    for x, y in zip(a, b, strict=True):
        assert x.messages == y.messages


def test_prepare_inputs_skips_invalid_trials(tmp_path):
    """Multimodal / assistant-led trials are dropped silently; the
    valid ones still produce requests. This matches the same
    behavior as `bench/deterministic_l3/corpus_swebench.py` in
    RocServe — a corpus written there replays here with the same
    semantics."""
    corpus = [
        {"conversations": [{"from": "gpt", "value": "leading assistant"}]},  # bad
        _trial(("human", "good")),  # ok
        {"conversations": []},  # bad
    ]
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps(corpus))
    pipe = _make_pipeline(p)
    out = pipe.prepare_inputs()
    assert len(out) == 1
    assert out[0].messages == [{"role": "user", "content": "good"}]


def test_corpus_from_rocserve_replays_unchanged(tmp_path):
    """The RocServe-side `bench/deterministic_l3/corpus_swebench.py`
    emits a dict `{"config": {...}, "sessions": [{messages: [...]}]}`,
    which is **not** the same shape this loader expects (it wants a
    flat list of trials). Document the asymmetry: agent-bench's
    trace-replay mode consumes the *raw* HF dataset JSON, not a
    pre-converted RocServe corpus.

    This test pins that boundary so a future refactor doesn't silently
    accept the wrong shape and produce empty / garbage requests."""
    rocserve_corpus = {
        "config": {"source_dataset": "test"},
        "sessions": [
            {"session_id": 0, "messages": [{"role": "user", "content": "hello"}]},
        ],
    }
    p = tmp_path / "rocserve-style.json"
    p.write_text(json.dumps(rocserve_corpus))
    pipe = _make_pipeline(p)
    with pytest.raises(ValueError, match="JSON list of trials"):
        pipe.prepare_inputs()
