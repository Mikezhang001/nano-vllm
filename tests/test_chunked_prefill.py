"""验证 chunked prefill:
- 强制小 max_num_batched_tokens 让长 prompt 被切成多个 chunk
- 混入短 prompt 观察是否命中「同 batch 内多序列不同 q_len 的 prefill」以及混合批
- 打印每 step 的组成 (decode 条数 / prefill chunk 大小列表 / is_decode_only)
- 用低温度做确定性对比: chunked vs 正常, 语义应一致 (bit-level 可能因浮点归约顺序略有差异)

由于 nccl process_group 每进程只能 init 一次, 用 --mode {chunked,normal} 分子进程跑,
外层再对比两次子进程的 token_ids 是否一致。
"""
import argparse
import json
import os
import subprocess
import sys

# 允许在 tests/ 下直接执行时也能 import 到项目根的 nanovllm 包
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams
from nanovllm.engine.scheduler import Scheduler

PATH = os.path.join(ROOT, "Qwen3-0.6B")


def run_once(max_num_batched_tokens: int, tag: str):
    _orig = Scheduler.schedule
    stats = []

    def schedule_with_log(self):
        seqs, is_decode_only = _orig(self)
        decode_cnt = sum(1 for s in seqs if s.num_scheduled_tokens == 1 and not s.is_prefill)
        prefill_chunks = [s.num_scheduled_tokens for s in seqs
                          if s.num_scheduled_tokens > 1 or s.is_prefill]
        stats.append((decode_cnt, prefill_chunks, is_decode_only))
        return seqs, is_decode_only

    Scheduler.schedule = schedule_with_log
    llm = LLM(
        PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=8,
        max_model_len=4096,
    )
    long_prompt = "Tell me a story about a brave knight. " * 60   # ~600+ tokens
    short1 = "Hello"
    short2 = "Hi there"
    sp = SamplingParams(temperature=0.01, max_tokens=16)
    outputs = llm.generate([long_prompt, short1, short2], sp, use_tqdm=False)

    print(f"\n===== {tag} (max_num_batched_tokens={max_num_batched_tokens}) =====", flush=True)
    print(f"{'step':>4} {'decode':>6} {'prefill_chunks':>24} {'is_decode_only'}", flush=True)
    for i, (d, p, dec) in enumerate(stats):
        print(f"{i:>4} {d:>6} {str(p):>24} {dec}", flush=True)
    mixed = sum(1 for d, p, _ in stats if p and d > 0)
    pure_pf = sum(1 for d, p, _ in stats if p and d == 0)
    pure_dec = sum(1 for d, p, _ in stats if not p)
    multi_prefill = sum(1 for d, p, _ in stats if len(p) > 1)
    print(f"总 step: {len(stats)} | 纯 prefill: {pure_pf} | 纯 decode: {pure_dec} | "
          f"prefill+decode 混合: {mixed} | 多序列 prefill 同批: {multi_prefill}", flush=True)
    for i, o in enumerate(outputs):
        print(f"  out[{i}]: {o['text']!r}", flush=True)

    # 把 token_ids 用特殊前缀打到 stdout, 供外层解析
    payload = json.dumps([o["token_ids"] for o in outputs])
    print(f"__TOKENIDS__ {payload}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chunked", "normal", "both"], default="both")
    args = parser.parse_args()

    if args.mode == "chunked":
        run_once(max_num_batched_tokens=128, tag="CHUNKED (small budget)")
    elif args.mode == "normal":
        run_once(max_num_batched_tokens=16384, tag="NORMAL (large budget)")
    else:
        # 外层驱动: 分两个子进程各跑一次, 再解析 token_ids 做等价对比
        results = {}
        for mode in ("chunked", "normal"):
            print(f"\n>>> spawning child process: mode={mode}", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--mode", mode],
                capture_output=True, text=True,
            )
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            token_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("__TOKENIDS__ ")]
            if not token_lines:
                print(f"[ERROR] mode={mode} 未捕获 token_ids", flush=True)
                return
            results[mode] = json.loads(token_lines[-1][len("__TOKENIDS__ "):])

        print("\n===== 确定性对比: chunked vs normal =====", flush=True)
        for i, (a, b) in enumerate(zip(results["chunked"], results["normal"])):
            print(f"  out[{i}] chunked==normal: {a == b} "
                  f"(len {len(a)} vs {len(b)})", flush=True)


if __name__ == "__main__":
    main()
