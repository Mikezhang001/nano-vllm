"""Spec 多 seq batch 真实场景 (8B target + 0.6B draft).

验证:
1. 不同 batch size (1, 2, 4, 8) 的接受率和速率
2. 多 seq batch 不同 accept 率下 rollback 正确性
3. 极端: prompts 长度差异大
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

TARGET_8B = os.path.join(ROOT, "Qwen3-8B")
DRAFT_06B = os.path.join(ROOT, "Qwen3-0.6B")


def run_bench(prompts, max_tokens=40, k=3, use_spec=True):
    tok = AutoTokenizer.from_pretrained(TARGET_8B)
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    kwargs = dict(
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=16,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
    )
    if use_spec:
        kwargs["speculative_model"] = DRAFT_06B
        kwargs["num_speculative_tokens"] = k

    llm = LLM(TARGET_8B, **kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    total_tokens = sum(len(o["token_ids"]) for o in outs)
    tok_per_s = total_tokens / dt
    stats = getattr(llm.model_runner, "spec_stats", None)
    accept_rate = None
    if stats and stats["draft_total"] > 0:
        accept_rate = stats["draft_accepted"] / stats["draft_total"]
    return dt, total_tokens, tok_per_s, accept_rate, outs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "spec", "compare"], default="compare")
    ap.add_argument("--bs", type=int, default=4, help="batch size")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    prompts_pool = [
        "What is 1 + 1?",
        "Explain gravity in one sentence.",
        "Say hi.",
        "What color is the sky and why?",
        "Write a haiku about winter.",
        "Name three planets.",
        "What is 2 * 3?",
        "Translate 'hello' to French.",
    ]
    prompts = prompts_pool[: args.bs]

    if args.mode in ("baseline", "spec"):
        dt, n, tps, ar, outs = run_bench(prompts, args.max_tokens, args.k, use_spec=(args.mode == "spec"))
        import json
        out_file = f"/tmp/spec_bench_{args.mode}_bs{args.bs}.json"
        with open(out_file, "w") as f:
            json.dump({
                "dt": dt, "total_tokens": n, "tok_per_s": tps, "accept_rate": ar,
                "outputs": [o["token_ids"] for o in outs],
            }, f)
        print(f"[{args.mode} bs={args.bs}] {tps:.1f} tok/s  accept={ar}  dt={dt:.2f}s tokens={n}")
        return

    # compare
    import subprocess, json
    for mode in ("baseline", "spec"):
        r = subprocess.run([sys.executable, "-u", __file__, "--mode", mode,
                            "--bs", str(args.bs), "-k", str(args.k),
                            "--max-tokens", str(args.max_tokens)],
                           capture_output=False, text=True, timeout=1200)
        if r.returncode != 0:
            print(f"[{mode}] FAILED"); sys.exit(1)

    with open(f"/tmp/spec_bench_baseline_bs{args.bs}.json") as f:
        b = json.load(f)
    with open(f"/tmp/spec_bench_spec_bs{args.bs}.json") as f:
        s = json.load(f)

    print(f"\n===== bs={args.bs} k={args.k} max_tokens={args.max_tokens} =====")
    print(f"baseline: {b['tok_per_s']:.1f} tok/s ({b['dt']:.2f}s / {b['total_tokens']} tok)")
    print(f"spec    : {s['tok_per_s']:.1f} tok/s ({s['dt']:.2f}s / {s['total_tokens']} tok)")
    print(f"          accept_rate={s['accept_rate']*100:.1f}%")
    print(f"speedup : {s['tok_per_s'] / b['tok_per_s']:.2f}x")

    # 输出对比 (宽松: 前 20 token 一致)
    strict_ok = prefix_ok = 0
    for i, (bo, so) in enumerate(zip(b["outputs"], s["outputs"])):
        if bo == so:
            strict_ok += 1
            print(f"  ✓ [{i}] identical len={len(bo)}")
        else:
            prefix_len = min(len(bo), len(so), 20)
            if bo[:prefix_len] == so[:prefix_len]:
                prefix_ok += 1
                # 找 diff
                for j in range(min(len(bo), len(so))):
                    if bo[j] != so[j]:
                        print(f"  ≈ [{i}] prefix ok, diff@{j} base_len={len(bo)} spec_len={len(so)}")
                        break
            else:
                print(f"  ✗ [{i}] prefix diff!")
                for j in range(min(len(bo), len(so))):
                    if bo[j] != so[j]:
                        print(f"      diff@{j}: base={bo[j]} spec={so[j]}")
                        break
    print(f"  一致: {strict_ok} 完全 / {prefix_ok} 前缀 / {len(b['outputs']) - strict_ok - prefix_ok} FAIL")


if __name__ == "__main__":
    main()
