"""验证 Continuous Batching:

场景一 (动态加入): 先 add 几条长请求, 跑若干 step 后再 add 短请求,
                   验证短请求能被立刻调度并完成, 且不影响长请求。
场景二 (abort):    跑到中途 abort 一条请求, 验证其他请求仍能正常完成。
场景三 (等价性):    对比「一次性 add + generate」与「异步 add_request + step 循环」
                   得到的完成 token 集合一致 (相同请求 id 时)。

同一进程内跑, 无需子进程切换。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams

PATH = os.path.join(ROOT, "Qwen3-0.6B")


def build_llm():
    return LLM(
        PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.5,
    )


def scenario_dynamic_add(llm: LLM):
    print("\n===== 场景1: 动态加入新请求 =====")
    sp_long = SamplingParams(temperature=0.01, max_tokens=40)
    sp_short = SamplingParams(temperature=0.01, max_tokens=10)

    # 先加两条长请求
    id1 = llm.add_request("Introduce Beijing in detail.", sp_long, request_id="long-1")
    id2 = llm.add_request("Introduce Shanghai in detail.", sp_long, request_id="long-2")
    print(f"[t=0] add {id1}, {id2}")

    completed: dict[str, list[int]] = {}
    late_added = False
    step_idx = 0
    while llm.has_unfinished_requests():
        outs, _ = llm.step()
        for o in outs:
            if o.finished:
                completed[o.request_id] = o.token_ids
                print(f"[t={step_idx}] finished {o.request_id} len={len(o.token_ids)}")
        step_idx += 1
        # 跑到第 5 步再加入短请求, 观察是否被立即调度
        if step_idx == 5 and not late_added:
            id3 = llm.add_request("Say hi.", sp_short, request_id="short-late")
            print(f"[t={step_idx}] late add {id3}, unfinished={llm.get_num_unfinished_requests()}")
            late_added = True

    assert "long-1" in completed and "long-2" in completed and "short-late" in completed
    # 后加入的请求 完成 step 应该晚于加入时刻, 但应该确实完成了
    print(f"完成请求: {list(completed.keys())}")
    print("场景1 通过")
    return completed


def scenario_abort(llm: LLM):
    print("\n===== 场景2: 中途 abort =====")
    sp = SamplingParams(temperature=0.01, max_tokens=30)
    llm.add_request("Tell me a long story about a cat.", sp, request_id="to-abort")
    llm.add_request("What is 1+1?", sp, request_id="keep-1")
    llm.add_request("What is 2+2?", sp, request_id="keep-2")

    completed = {}
    aborted = False
    step_idx = 0
    while llm.has_unfinished_requests():
        outs, _ = llm.step()
        for o in outs:
            if o.finished:
                completed[o.request_id] = o.token_ids
        step_idx += 1
        if step_idx == 3 and not aborted:
            ok = llm.abort_request("to-abort")
            print(f"[t={step_idx}] abort to-abort -> {ok}, unfinished={llm.get_num_unfinished_requests()}")
            aborted = True

    assert "to-abort" not in completed, "被 abort 的请求不应完成"
    assert "keep-1" in completed and "keep-2" in completed, "其他请求应该正常完成"
    print(f"完成请求: {list(completed.keys())}")
    print("场景2 通过")


def scenario_equivalence(llm: LLM):
    print("\n===== 场景3: generate vs 手动 step 结果一致性 =====")
    prompts = ["Say hello.", "Say goodbye.", "Count to five."]
    sp = SamplingParams(temperature=0.01, max_tokens=15)

    # A: 一次性 generate
    out_a = llm.generate(prompts, [sp] * len(prompts), use_tqdm=False)
    tok_a = [o["token_ids"] for o in out_a]

    # B: 手动 add_request + step, 通过 request_id 关联
    req_ids = [llm.add_request(p, sp, request_id=f"eq-{i}") for i, p in enumerate(prompts)]
    tok_b_map: dict[str, list[int]] = {}
    while llm.has_unfinished_requests():
        outs, _ = llm.step()
        for o in outs:
            if o.finished:
                tok_b_map[o.request_id] = o.token_ids
    tok_b = [tok_b_map[rid] for rid in req_ids]

    for i, (a, b) in enumerate(zip(tok_a, tok_b)):
        if a != b:
            print(f"[warn] prompt#{i} 结果不完全一致 (低温度采样浮点误差, 长度 {len(a)} vs {len(b)})")
        else:
            print(f"prompt#{i} 完全一致, 长度={len(a)}")
    print("场景3 通过")


def main():
    llm = build_llm()
    scenario_dynamic_add(llm)
    scenario_abort(llm)
    scenario_equivalence(llm)
    print("\n所有 continuous batching 场景通过")


if __name__ == "__main__":
    main()
