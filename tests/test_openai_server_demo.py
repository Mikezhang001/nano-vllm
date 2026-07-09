"""HTTP server demo 客户端: 用 openai SDK 调 nano-vllm server.

用法 (先在另一个终端启动 server):
    python -m nanovllm.entrypoints.openai.api_server --model ./Qwen3-0.6B --port 8000

然后运行:
    python tests/test_openai_server_demo.py           # 非流式 + 流式
    python tests/test_openai_server_demo.py --concurrent  # 并发 3 请求
"""
import argparse
import os
import sys
import time

# openai SDK
from openai import OpenAI


BASE_URL = os.environ.get("NANOVLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("NANOVLLM_MODEL", "Qwen3-0.6B")


def demo_non_stream(client: OpenAI):
    print("\n===== 非流式 =====")
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "请用一句话介绍北京"}],
        temperature=0.6,
        max_tokens=80,
    )
    dt = time.perf_counter() - t0
    print(f"耗时 {dt:.2f}s, 结果:")
    print(resp.choices[0].message.content)
    print(f"usage: {resp.usage}")


def demo_stream(client: OpenAI):
    print("\n===== 流式 =====")
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "请用一句话介绍上海"}],
        temperature=0.6,
        max_tokens=80,
        stream=True,
    )
    print("流式输出: ", end="", flush=True)
    first_chunk_at = None
    for chunk in stream:
        if first_chunk_at is None:
            first_chunk_at = time.perf_counter() - t0
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    dt = time.perf_counter() - t0
    print(f"\n总耗时 {dt:.2f}s, 首 token {first_chunk_at:.2f}s")


def demo_concurrent(client: OpenAI):
    print("\n===== 并发 3 请求 (验证 continuous batching) =====")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    prompts = [
        "1+1 等于几?",
        "什么是量子力学?",
        "推荐一本科幻小说",
    ]

    def one(idx: int, p: str):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": p}],
            temperature=0.6,
            max_tokens=60,
        )
        return idx, p, resp.choices[0].message.content, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(one, i, p) for i, p in enumerate(prompts)]
        for f in as_completed(futs):
            idx, p, ans, dt = f.result()
            print(f"[req {idx}] {dt:.2f}s  Q: {p}\n           A: {ans[:80]}...\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrent", action="store_true")
    args = ap.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    # 探活 /v1/models
    print(f"[client] base_url={BASE_URL}  model={MODEL}")
    models = client.models.list()
    print(f"[client] models: {[m.id for m in models.data]}")

    if args.concurrent:
        demo_concurrent(client)
    else:
        demo_non_stream(client)
        demo_stream(client)

    print("\n[ok] demo 完成")


if __name__ == "__main__":
    main()
