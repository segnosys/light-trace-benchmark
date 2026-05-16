"""
Tests for the input_pipeline HF reader bugfix: when chat=true is set on a
dataset column containing plain text, the original code called json.loads
and crashed. Now it should auto-wrap as a single user message.
"""
import json
from unittest.mock import patch, MagicMock

from lightrace.input_pipeline import InputPipeline


def _fake_dataset(rows):
    """Build a list-like object that looks like a HuggingFace Dataset."""
    ds = MagicMock()
    ds.__iter__ = lambda self: iter(rows)
    ds.__len__ = lambda self: len(rows)
    ds.select = lambda r: _fake_dataset([rows[i] for i in r])
    return ds


def _pipeline(*, chat, column="problem"):
    return InputPipeline(
        model_name="x", dataset_type="hf",
        stream=True, max_tokens=16, skip_eos=True,
        temperature=0.0, top_p=None, chat=chat,
        num_examples=2, tokenizer_name=None,
        hf_dataset="HuggingFaceH4/MATH-500",
        hf_dataset_split="test",
        hf_dataset_column_name=column,
    )


def test_chat_true_plain_text_column_auto_wraps_as_user_message():
    """The fix: plain text shouldn't be json.loads()'d."""
    pipe = _pipeline(chat=True)
    rows = [
        {"problem": "What is 2+2?"},
        {"problem": "Solve x^2 - 4 = 0"},
    ]
    with patch("lightrace.input_pipeline.datasets.load_dataset",
               return_value=_fake_dataset(rows)):
        out = pipe.prepare_inputs()

    assert len(out) == 2
    for item, row in zip(out, rows):
        assert item.messages == [{"role": "user", "content": row["problem"]}]
        assert item.prompt is None


def test_chat_true_json_string_column_decodes():
    """If the column already contains a JSON-encoded chat list, use it."""
    pipe = _pipeline(chat=True)
    rows = [
        {"problem": json.dumps([{"role": "user", "content": "msg-A"}])},
        {"problem": json.dumps([{"role": "user", "content": "msg-B"}])},
    ]
    with patch("lightrace.input_pipeline.datasets.load_dataset",
               return_value=_fake_dataset(rows)):
        out = pipe.prepare_inputs()
    assert out[0].messages == [{"role": "user", "content": "msg-A"}]


def test_chat_true_list_column_passes_through():
    """Already-decoded list of messages: don't re-decode."""
    pipe = _pipeline(chat=True)
    rows = [
        {"problem": [{"role": "user", "content": "hello"}]},
    ]
    with patch("lightrace.input_pipeline.datasets.load_dataset",
               return_value=_fake_dataset(rows * 2)):
        out = pipe.prepare_inputs()
    assert out[0].messages == [{"role": "user", "content": "hello"}]


def test_chat_false_uses_prompt_field():
    """With chat=false we put the text in `prompt` and skip messages entirely."""
    pipe = _pipeline(chat=False)
    rows = [{"problem": "raw prompt text"}, {"problem": "another"}]
    with patch("lightrace.input_pipeline.datasets.load_dataset",
               return_value=_fake_dataset(rows)):
        out = pipe.prepare_inputs()
    assert out[0].prompt == "raw prompt text"
    assert out[0].messages is None
