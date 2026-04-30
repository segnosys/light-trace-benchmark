"""Build a tokenized agent-style corpus: real code (sglang fork) + a small mix
of natural-language explanations. Saved as a JSON list of (kind, path, tokens).

Run once; consumers (lightrace) load the JSON and slice deterministically.

Usage:
    python3 build_agent_corpus.py \
        --tokenizer /mnt/vast/john/huggingface/MiniMax-M2.5 \
        --out /mnt/vast/jiejing/agent_corpus.json
"""
import argparse
import glob
import json
import os
import random
import sys

from transformers import AutoTokenizer


# Curated short technical-prose snippets (sound like code-review / dev-doc).
NL_BLOCKS = [
    "This module implements the inner attention loop for the spec-decoding draft worker. "
    "When a draft batch arrives we compute logits, run the verify kernel, and emit the "
    "accepted tokens back to the parent scheduler before reseeding the next round.",
    "Note: the page allocator deliberately overcommits by ~5% so that bursty prefill "
    "doesn't immediately stall on a single eviction. The radix tree handles the actual "
    "lifetime tracking.",
    "TODO: when piecewise cuda graph is enabled and tp_size > 1 we need to re-broadcast "
    "the warm-up tensors before the second capture. Skipping it causes the scheduler "
    "to hang silently on rank > 0.",
    "We measured this at TP=2/MI355X with cuda graph on, batch sizes [1, 4, 16, 32, 64]. "
    "End-to-end decode latency was 9.7 ms at K=1 and 41.3 ms at K=128, dominated by "
    "MoE GEMM and paged-attention KV reads.",
    "Rationale for choosing num_draft_tokens=5 over 4: the verify-batch grows 5x linearly "
    "but the per-step overhead is dominated by attention KV scan, not by candidate count. "
    "At long context this dominates anyway.",
    "Reviewer: please double check the dtype cast on line 142 -- on AMD/HIP the bf16 path "
    "needs an explicit `.to()` because the autograd graph keeps the float32 lift around "
    "from the embedding layer.",
    "If the test fails with `expected mat1 and mat2 to have the same dtype`, set "
    "`--dtype bfloat16` at server launch. The default `auto` resolves the target to "
    "fp16 from the gguf metadata which doesn't match the bf16-trained EAGLE3 draft.",
    "The acceptance length on a randomly generated benchmark workload tends to under-"
    "represent real usage. Rerun against the agent corpus (this one) to get a fair "
    "comparison number; on coding-style data we observed ~30% higher acceptance.",
]


def collect_files(root: str, patterns) -> list:
    out = []
    for pat in patterns:
        for f in glob.glob(os.path.join(root, pat), recursive=True):
            try:
                size = os.path.getsize(f)
                if 200 <= size <= 60_000:
                    out.append(f)
            except OSError:
                pass
    return sorted(set(out))


def build(tokenizer_path: str, out_path: str, code_root: str, max_files: int):
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )

    files = collect_files(
        code_root,
        [
            "python/**/*.py",
            "sgl-kernel/csrc/**/*.cu",
            "sgl-kernel/csrc/**/*.cpp",
            "sgl-kernel/csrc/**/*.cc",
            "sgl-kernel/include/**/*.h",
            "sgl-kernel/python/**/*.py",
        ],
    )
    print(f"found {len(files)} candidate files", file=sys.stderr)

    rng = random.Random(1337)
    rng.shuffle(files)
    files = files[:max_files]

    out_records = []
    total_tokens = 0
    for f in files:
        try:
            content = open(f).read()
        except (OSError, UnicodeDecodeError):
            continue
        ids = tok.encode(content, add_special_tokens=False)
        if len(ids) < 30 or len(ids) > 8000:
            continue
        rel = os.path.relpath(f, code_root)
        out_records.append({"kind": "code", "path": rel, "tokens": ids})
        total_tokens += len(ids)

    # Add NL blocks
    for i, txt in enumerate(NL_BLOCKS):
        ids = tok.encode(txt, add_special_tokens=False)
        out_records.append({"kind": "nl", "path": f"_review_note_{i}", "tokens": ids})
        total_tokens += len(ids)

    print(f"corpus: {len(out_records)} entries, {total_tokens:,} tokens total",
          file=sys.stderr)
    json.dump(out_records, open(out_path, "w"))
    print(f"wrote {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-root", default="/mnt/vast/jiejing/sglang-fork")
    ap.add_argument("--max-files", type=int, default=600)
    args = ap.parse_args()
    build(args.tokenizer, args.out, args.code_root, args.max_files)


if __name__ == "__main__":
    main()
