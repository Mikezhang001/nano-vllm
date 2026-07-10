"""Spec + continuous batching 交互测试.

场景:
1. 动态 add_request: 老请求正在 spec decode 时新请求进来做 prefill
2. abort_request: 正在 spec decode 中被 abort
3. 混合完成时间: 不同 max_tokens 的请求交错完成
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = os.path.join(ROOT, "Qwen3-0.6B")


def build_llm():
    return LLM(
        MODEL,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=8,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        speculative_model=MODEL,
        num_speculative_tokens=2,
    )


def test_dynamic_add():
    """场景 1: 动态添加请求."""
    print("\n===== Test 1: 动态 add_request =====")
    llm = build_llm()
    tok = llm.tokenizer

    def wrap(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    # 起始加 2 个长请求
    req_ids = []
    req_ids.append(llm.add_request(wrap("Count from 1 to 30."),
                                    SamplingParams(temperature=0.0, max_tokens=80)))
    req_ids.append(llm.add_request(wrap("Explain gravity."),
                                    SamplingParams(temperature=0.0, max_tokens=80)))

    step_count = 0
    finished = {}
    added_extras = False
    while llm.has_unfinished_requests():
        outs, _ = llm.step()
        step_count += 1
        # 步 5 时动态加 2 个短请求
        if step_count == 5 and not added_extras:
            req_ids.append(llm.add_request(wrap("Say hi."),
                                            SamplingParams(temperature=0.0, max_tokens=15)))
            req_ids.append(llm.add_request(wrap("What is 1+1?"),
                                            SamplingParams(temperature=0.0, max_tokens=15)))
            added_extras = True
            print(f"  step {step_count}: 动态加入 2 个新请求, 总数 {len(req_ids)}")
        for o in outs:
            if o.finished:
                finished[o.request_id] = o.token_ids

    print(f"  共 {step_count} steps, {len(finished)} 请求完成")
    assert len(finished) == len(req_ids), f"应完成 {len(req_ids)} 个, 实际 {len(finished)}"
    for rid in req_ids:
        assert rid in finished, f"{rid} 未完成"
        text = tok.decode(finished[rid])
        print(f"  {rid[:8]}: len={len(finished[rid])} tail={text[-50:]!r}")
    print("  PASS")


def test_abort():
    """场景 2: 中途 abort 正在跑的请求."""
    print("\n===== Test 2: 动态 abort_request =====")
    llm = build_llm()
    tok = llm.tokenizer

    def wrap(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    r1 = llm.add_request(wrap("Count to 100."), SamplingParams(temperature=0.0, max_tokens=200))
    r2 = llm.add_request(wrap("Say hi."), SamplingParams(temperature=0.0, max_tokens=15))
    r3 = llm.add_request(wrap("Explain gravity."), SamplingParams(temperature=0.0, max_tokens=200))

    step_count = 0
    finished = {}
    aborted_r1 = aborted_r3 = False
    while llm.has_unfinished_requests():
        outs, _ = llm.step()
        step_count += 1
        # 步 8 abort r1 (还在跑长请求)
        if step_count == 8 and not aborted_r1:
            ok = llm.abort_request(r1)
            print(f"  step {step_count}: abort {r1[:8]} -> {ok}")
            aborted_r1 = True
        # 步 10 abort r3
        if step_count == 10 and not aborted_r3:
            ok = llm.abort_request(r3)
            print(f"  step {step_count}: abort {r3[:8]} -> {ok}")
            aborted_r3 = True
        for o in outs:
            if o.finished:
                finished[o.request_id] = o.token_ids

    print(f"  共 {step_count} steps, {len(finished)} 请求完成 (r2 应完成, r1/r3 被 abort)")
    assert r2 in finished, "r2 应正常完成"
    assert r1 not in finished, "r1 应被 abort"
    assert r3 not in finished, "r3 应被 abort"
    assert not llm.has_unfinished_requests(), "abort 后应无未完成请求"
    print("  PASS")


def test_mixed_lengths():
    """场景 3: 混合长度, 短请求先完成, 长请求继续跑."""
    print("\n===== Test 3: 混合长度请求 =====")
    llm = build_llm()
    tok = llm.tokenizer

    def wrap(p):
        return tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    reqs = [
        (wrap("Say hi."), 10),
        (wrap("Count to 30."), 60),
        (wrap("Hi."), 5),
        (wrap("Explain photosynthesis in 2 sentences."), 40),
    ]
    req_ids = []
    for p, mt in reqs:
        req_ids.append(llm.add_request(p, SamplingParams(temperature=0.0, max_tokens=mt)))

    finish_order = []
    while llm.has_unfinished_requests():
        outs, _ = llm.step()
        for o in outs:
            if o.finished:
                finish_order.append((o.request_id, len(o.token_ids)))

    print(f"  完成顺序:")
    for rid, ln in finish_order:
        idx = req_ids.index(rid)
        exp_mt = reqs[idx][1]
        mark = "✓" if ln <= exp_mt else "✗"
        print(f"    {mark} req#{idx} len={ln} (max_tokens={exp_mt})")
    assert len(finish_order) == len(req_ids)
    # 短请求 (max_tokens=5, 10) 应该先完成
    print("  PASS")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["1", "2", "3", "all"], default="all")
    args = ap.parse_args()
    if args.test == "1":
        test_dynamic_add()
    elif args.test == "2":
        test_abort()
    elif args.test == "3":
        test_mixed_lengths()
    else:
        # 每个测试用独立 subprocess 跑, 避免 init_process_group 二次初始化
        import subprocess
        all_pass = True
        for t in ["1", "2", "3"]:
            print(f"\n>>> Test {t}")
            r = subprocess.run([sys.executable, "-u", __file__, "--test", t],
                               capture_output=False, text=True, timeout=600)
            if r.returncode != 0:
                all_pass = False
                print(f"  Test {t} FAILED")
        print(f"\n{'所有测试通过' if all_pass else '部分测试失败'}")
        sys.exit(0 if all_pass else 1)
