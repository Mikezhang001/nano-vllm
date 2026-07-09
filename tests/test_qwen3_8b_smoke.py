"""Qwen3-8B smoke test:
- 验证 nano-vllm 能否直接加载/推理 Qwen3-8B (投机解码的 target 候选)
- 不改动 example.py, 单独脚本
- 简短 prompt + 少量 tokens, 只验证功能, 不测吞吐
"""
import os
import sys
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_PATH = os.path.join(ROOT, "Qwen3-8B")


def main():
    print(f"== Qwen3-8B smoke test ==")
    print(f"model path: {MODEL_PATH}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, total mem = {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    t0 = time.perf_counter()
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
    )
    load_time = time.perf_counter() - t0
    print(f"[load] LLM 加载耗时 {load_time:.1f}s")
    print(f"[load] GPU 已用 {torch.cuda.memory_allocated() / 1024**3:.1f} GB, 峰值 {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")

    sp = SamplingParams(temperature=0.6, max_tokens=64)
    prompts = ["introduce yourself briefly", "what is 2+2?"]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    gen_time = time.perf_counter() - t0
    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    print(f"[gen] 生成 {total_tokens} tokens 耗时 {gen_time:.2f}s, 速度 {total_tokens / gen_time:.1f} tok/s")

    for i, (p, o) in enumerate(zip(prompts, outputs)):
        print(f"\n--- Prompt #{i} ---")
        # 只打印 chat template 里的 user content 部分
        print(f"Prompt (tail): ...{p[-80:]!r}")
        print(f"Completion   : {o['text']!r}")

    print("\n[ok] Qwen3-8B smoke test 通过")


if __name__ == "__main__":
    main()
