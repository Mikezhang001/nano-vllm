"""OpenAI 兼容 HTTP 服务 (demo 版).

启动示例:
  python -m nanovllm.entrypoints.openai.api_server \
      --model ./Qwen3-0.6B --port 8000

支持:
  - GET  /health
  - GET  /v1/models
  - POST /v1/chat/completions        (stream / non-stream)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.entrypoints.openai.engine_loop import EngineLoop
from nanovllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ModelCard,
    ModelList,
    UsageInfo,
)


# 全局单例 (由 lifespan 初始化)
class _AppState:
    llm: Optional[LLM] = None
    tokenizer = None
    engine_loop: Optional[EngineLoop] = None
    model_name: str = "nanovllm-model"     # 对外暴露的模型名
    model_path: str = ""


state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动前 llm 已由 main() 初始化, 这里只是启动后台线程
    print("[server] starting engine loop ...")
    state.engine_loop.start()
    yield
    print("[server] stopping engine loop ...")
    state.engine_loop.stop()


app = FastAPI(title="nano-vllm OpenAI server (demo)", lifespan=lifespan)


# ============ 工具函数 ============

def _now() -> int:
    return int(time.time())


def _make_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _build_prompt(messages: list[ChatMessage]) -> tuple[str, list[int]]:
    """把 messages 转为 prompt text + token_ids (走 chat template)."""
    tok = state.tokenizer
    conv = [{"role": m.role, "content": m.content} for m in messages]
    prompt_text = tok.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tok.encode(prompt_text)
    return prompt_text, prompt_ids


def _to_sampling_params(req: ChatCompletionRequest) -> SamplingParams:
    """OpenAI 请求 -> nano-vllm SamplingParams (只映射支持字段)."""
    # SamplingParams 断言 temperature > 1e-10
    temp = max(req.temperature, 1e-6) if req.temperature is not None else 1.0
    max_tokens = req.max_tokens if req.max_tokens and req.max_tokens > 0 else 128
    return SamplingParams(temperature=temp, max_tokens=max_tokens, ignore_eos=False)


# ============ Endpoints ============

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelCard(id=state.model_name, created=_now())]).model_dump()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, raw_request: Request):
    if req.n != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported in demo")
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    prompt_text, prompt_ids = _build_prompt(req.messages)
    sp = _to_sampling_params(req)
    request_id = _make_id("chatcmpl")
    q = state.engine_loop.submit(prompt_ids, sp, request_id=request_id)

    if req.stream:
        return StreamingResponse(
            _stream_generator(request_id, req.model or state.model_name, q, raw_request),
            media_type="text/event-stream",
        )
    return await _non_stream_generator(request_id, req.model or state.model_name, prompt_ids, q, raw_request)


# ============ 非流式 ============

async def _non_stream_generator(
    request_id: str,
    model_name: str,
    prompt_ids: list[int],
    q,
    raw_request: Request,
):
    tok = state.tokenizer
    completion_ids: list[int] = []
    finished = False
    finish_reason = "stop"
    try:
        while True:
            out = await q.get()
            if out is None:                  # sentinel, 已 abort
                finish_reason = "abort"
                break
            completion_ids = out.token_ids   # RequestOutput 是全量
            if out.finished:
                finished = True
                # 判断截断
                if len(completion_ids) >= _to_sampling_params_max(out):
                    finish_reason = "length"
                break
            if await raw_request.is_disconnected():
                state.engine_loop.abort(request_id)
                finish_reason = "abort"
                break
    except Exception as e:
        state.engine_loop.abort(request_id)
        raise HTTPException(status_code=500, detail=str(e))

    completion_text = tok.decode(completion_ids, skip_special_tokens=True) if completion_ids else ""
    resp = ChatCompletionResponse(
        id=request_id,
        created=_now(),
        model=model_name,
        choices=[ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=completion_text),
            finish_reason=finish_reason if finished else "stop",
        )],
        usage=UsageInfo(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion_ids),
            total_tokens=len(prompt_ids) + len(completion_ids),
        ),
    )
    return JSONResponse(resp.model_dump())


def _to_sampling_params_max(out) -> int:
    """RequestOutput 不带 max_tokens, 这里不精细区分, 返回极大值让 finish_reason=stop."""
    return 10**9


# ============ 流式 SSE ============

async def _stream_generator(request_id: str, model_name: str, q, raw_request: Request):
    tok = state.tokenizer
    created = _now()

    def _pack(choice: ChatCompletionStreamChoice) -> bytes:
        chunk = ChatCompletionStreamResponse(
            id=request_id, created=created, model=model_name, choices=[choice],
        )
        return f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n".encode("utf-8")

    # 首帧: role
    yield _pack(ChatCompletionStreamChoice(index=0, delta=DeltaMessage(role="assistant")))

    buffer_ids: list[int] = []
    last_text_len = 0
    try:
        while True:
            if await raw_request.is_disconnected():
                state.engine_loop.abort(request_id)
                break
            out = await q.get()
            if out is None:
                break
            # 用 delta_token_ids 累计
            for tid in out.delta_token_ids:
                buffer_ids.append(tid)
            new_text = tok.decode(buffer_ids, skip_special_tokens=True) if buffer_ids else ""
            delta_text = new_text[last_text_len:]
            last_text_len = len(new_text)

            if delta_text:
                yield _pack(ChatCompletionStreamChoice(
                    index=0, delta=DeltaMessage(content=delta_text),
                ))
            if out.finished:
                yield _pack(ChatCompletionStreamChoice(
                    index=0, delta=DeltaMessage(), finish_reason="stop",
                ))
                break
    except Exception as e:
        state.engine_loop.abort(request_id)
        # 通过 SSE 传递错误
        err = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
    finally:
        yield b"data: [DONE]\n\n"


# ============ 启动入口 ============

def build_arg_parser():
    p = argparse.ArgumentParser(description="nano-vllm OpenAI-compatible server (demo)")
    p.add_argument("--model", required=True, help="模型目录, 如 ./Qwen3-0.6B")
    p.add_argument("--served-model-name", default=None, help="对外暴露的模型名 (默认取 --model 的 basename)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true", default=True,
                   help="默认强制 eager, demo 无需 cuda graph")
    return p


def main():
    args = build_arg_parser().parse_args()
    model_path = os.path.abspath(args.model)
    assert os.path.isdir(model_path), f"model dir not found: {model_path}"

    print(f"[server] loading model from {model_path} ...")
    state.model_path = model_path
    state.model_name = args.served_model_name or os.path.basename(model_path.rstrip("/"))
    state.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    state.llm = LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    state.engine_loop = EngineLoop(state.llm)

    print(f"[server] listening on http://{args.host}:{args.port}, served model = {state.model_name!r}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
