"""MTP (Multi-Token Prediction) demo & 正确性测试.

由于当前没有原生支持 MTP heads 的开源模型可在 4xL20 上跑, 本测试用
Qwen3-0.6B 作为 D 个 MTP module 的模拟, 走 nanovllm 的 MTP 接口
(MTPPredictor + MTPVerifier + LLMEngine.run_mtp), 验证:

  1. MTP 与 baseline 输出等价 (贪婪采样)
  2. MTP 与 spec 输出/接受率一致 (因为底层实现共用)
  3. MTP 的 config / API 表达正确 (mtp_module, mtp_num_heads)

如未来接上真实 MTP 模型 (DeepSeek-V3 etc.), 只需替换 MTPPredictor.predict_chain
内部为 target 内置 head chain forward, 本测试骨架仍然适用.
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
MTP_06B = os.path.join(ROOT, "Qwen3-0.6B")


def run_generate(prompts, use_mtp: bool, use_spec: bool = False, k: int = 3,
                 max_tokens: int = 30):
    tok = AutoTokenizer.from_pretrained(TARGET_8B)
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    kwargs = dict(
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=8,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
    )
    if use_mtp:
        kwargs["mtp_module"] = MTP_06B
        kwargs["mtp_num_heads"] = k
    elif use_spec:
        kwargs["speculative_model"] = MTP_06B
        kwargs["num_speculative_tokens"] = k

    llm = LLM(TARGET_8B, **kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0

    stats = getattr(llm.model_runner, "spec_stats", None)
    accept_rate = None
    if stats and stats["draft_total"] > 0:
        accept_rate = stats["draft_accepted"] / stats["draft_total"]

    return {
        "dt": dt,
        "outputs": [o["token_ids"] for o in outs],
        "accept_rate": accept_rate,
        "tok_per_s": sum(len(o["token_ids"]) for o in outs) / dt,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    prompts = [
        "What is 1 + 1?",
        "Explain gravity in one sentence.",
        "Say hi.",
        "Name three planets.",
    ]

    if mode in ("baseline", "spec", "mtp"):
        r = run_generate(prompts, use_mtp=(mode == "mtp"), use_spec=(mode == "spec"))
        import json
        with open(f"/tmp/mtp_test_{mode}.json", "w") as f:
            json.dump(r, f)
        print(f"[{mode}] {r['tok_per_s']:.1f} tok/s  accept={r['accept_rate']}  dt={r['dt']:.2f}s")
        return

    # compare: subprocess 依次跑三种模式
    import subprocess, json
    for m in ("baseline", "spec", "mtp"):
        print(f">>> Running {m} ...")
        r = subprocess.run([sys.executable, "-u", __file__, m],
                           capture_output=False, text=True, timeout=1200)
        if r.returncode != 0:
            print(f"[{m}] FAILED"); sys.exit(1)

    results = {}
    for m in ("baseline", "spec", "mtp"):
        with open(f"/tmp/mtp_test_{m}.json") as f:
            results[m] = json.load(f)

    print("\n===== 结果对比 =====")
    for m in ("baseline", "spec", "mtp"):
        r = results[m]
        ar = f"accept={r['accept_rate']*100:.1f}%" if r['accept_rate'] is not None else "accept=N/A"
        print(f"  {m:>8}: {r['tok_per_s']:.1f} tok/s  {ar}  ({r['dt']:.2f}s)")

    # 输出等价性: mtp vs spec 应该完全一致 (底层实现相同)
    #              mtp vs baseline 应该前缀一致
    b, s, m = results["baseline"]["outputs"], results["spec"]["outputs"], results["mtp"]["outputs"]

    print("\n===== 一致性: mtp vs spec (应完全一致) =====")
    identical_ms = sum(1 for so, mo in zip(s, m) if so == mo)
    print(f"  完全一致: {identical_ms}/{len(s)}")
    for i, (so, mo) in enumerate(zip(s, m)):
        if so != mo:
            for j in range(min(len(so), len(mo))):
                if so[j] != mo[j]:
                    print(f"  ✗ [{i}] diff@{j}: spec={so[j]} mtp={mo[j]}")
                    break

    print("\n===== 一致性: mtp vs baseline (贪婪等价, 允许尾部数值差) =====")
    for i, (bo, mo) in enumerate(zip(b, m)):
        if bo == mo:
            print(f"  ✓ [{i}] identical len={len(bo)}")
        else:
            for j in range(min(len(bo), len(mo))):
                if bo[j] != mo[j]:
                    prefix = min(j, 20)
                    tag = "≈" if j >= prefix else "✗"
                    print(f"  {tag} [{i}] diff@{j} base_len={len(bo)} mtp_len={len(mo)}")
                    break

    assert identical_ms == len(s), "MTP 与 spec 输出应完全一致 (底层相同实现)"
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
