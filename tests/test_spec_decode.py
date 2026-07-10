"""Speculative Decoding smoke test (MVP).

场景 1 (自测): draft == target == Qwen3-0.6B
  - 期望接受率 100% (完全一致), 每 step 出 k+1 个 token
  - 结果与非 spec 应完全一致

场景 2 (真实): target = Qwen3-8B, draft = Qwen3-0.6B
  - 期望接受率 30-70%
  - 结果与非 spec (纯 8B) 应基本一致 (贪婪等价保证)
"""
import argparse
import os
import sys
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


TARGET_06B = os.path.join(ROOT, "Qwen3-0.6B")
TARGET_8B = os.path.join(ROOT, "Qwen3-8B")


def run(target_path: str, draft_path: str | None, prompts: list[str], k: int, max_tokens: int, tag: str):
    print(f"\n===== {tag} =====")
    print(f"target: {target_path}")
    print(f"draft : {draft_path}")
    print(f"k     : {k}")
    tok = AutoTokenizer.from_pretrained(target_path)

    llm = LLM(
        target_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        speculative_model=draft_path,
        num_speculative_tokens=k,
    )
    # 简单 chat format
    prompt_texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    sp = SamplingParams(temperature=0.001, max_tokens=max_tokens)  # 近似贪婪
    t0 = time.perf_counter()
    outs = llm.generate(prompt_texts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    total_tokens = sum(len(o["token_ids"]) for o in outs)
    print(f"耗时 {dt:.2f}s, 生成 {total_tokens} tokens, 速率 {total_tokens / dt:.1f} tok/s")
    # 打印接受率 (若走了 spec)
    stats = getattr(llm.model_runner, "spec_stats", None)
    if stats and stats["draft_total"] > 0:
        rate = stats["draft_accepted"] / stats["draft_total"]
        print(f"接受率 {rate * 100:.1f}% ({stats['draft_accepted']}/{stats['draft_total']}), spec steps={stats['steps']}")
    for i, o in enumerate(outs):
        print(f"--- [{i}] {prompts[i][:30]}...")
        print(f"    {o['text'][:120]!r}")
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["self", "real", "baseline_06b", "baseline_8b"], required=True)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    prompts = [
        "What is 1 + 1?",
        "Say hello briefly.",
    ]

    if args.mode == "self":
        # draft == target == 0.6B
        run(TARGET_06B, TARGET_06B, prompts, args.k, args.max_tokens, "SELF: 0.6B + 0.6B (贪婪应 100% 接受)")
    elif args.mode == "real":
        run(TARGET_8B, TARGET_06B, prompts, args.k, args.max_tokens, "REAL: 8B target + 0.6B draft")
    elif args.mode == "baseline_06b":
        run(TARGET_06B, None, prompts, args.k, args.max_tokens, "BASELINE: 0.6B (no spec)")
    elif args.mode == "baseline_8b":
        run(TARGET_8B, None, prompts, args.k, args.max_tokens, "BASELINE: 8B (no spec)")


if __name__ == "__main__":
    main()
