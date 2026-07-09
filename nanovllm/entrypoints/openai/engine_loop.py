"""后台线程驱动 LLMEngine.step(), 把每 step 的增量输出按 request_id
fan-out 到对应的 asyncio.Queue, 供 HTTP handler 消费."""
from __future__ import annotations
import asyncio
import threading
import time
from typing import Optional

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import RequestOutput


class EngineLoop:
    """线程安全的 LLM 请求分发器.

    - HTTP handler (asyncio) 调 submit() 提交请求, 拿到一个 asyncio.Queue
    - 后台线程持续 llm.step(), 把每 request 的 RequestOutput 塞入对应队列
    - handler 从 queue 里逐步读取, finished 时 queue 会收到最后一条 (finished=True)
    """

    STOP_SENTINEL = None   # None 表示流结束

    def __init__(self, llm: LLM):
        self.llm = llm
        self._engine_lock = threading.Lock()
        self._queues: dict[str, asyncio.Queue] = {}
        # 记录每个 request 所属的 event loop, 用于 threadsafe 投递
        self._loops: dict[str, asyncio.AbstractEventLoop] = {}
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="EngineLoop", daemon=True)
        self._thread.start()

    def stop(self):
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # ============ HTTP handler 使用的接口 ============

    def submit(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        request_id: str,
    ) -> asyncio.Queue:
        """提交一个请求, 返回用于读增量输出的 asyncio.Queue."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        with self._engine_lock:
            self._queues[request_id] = q
            self._loops[request_id] = loop
            self.llm.add_request(prompt_token_ids, sampling_params, request_id=request_id)
        return q

    def abort(self, request_id: str):
        """客户端断线时调用."""
        with self._engine_lock:
            self.llm.abort_request(request_id)
            q = self._queues.pop(request_id, None)
            loop = self._loops.pop(request_id, None)
        if q is not None and loop is not None:
            # 投递 sentinel 让读端退出
            try:
                loop.call_soon_threadsafe(q.put_nowait, self.STOP_SENTINEL)
            except RuntimeError:
                pass

    # ============ 后台线程主循环 ============

    def _run(self):
        while not self._shutdown.is_set():
            # 没有活跃请求时 sleep 一下, 避免空转
            with self._engine_lock:
                has_work = self.llm.has_unfinished_requests()
            if not has_work:
                time.sleep(0.005)
                continue

            with self._engine_lock:
                outputs, _ = self.llm.step()

            # 把每条输出投递到对应 queue
            for out in outputs:
                q = self._queues.get(out.request_id)
                loop = self._loops.get(out.request_id)
                if q is None or loop is None:
                    continue
                try:
                    loop.call_soon_threadsafe(q.put_nowait, out)
                except RuntimeError:
                    # loop 已关闭
                    pass
                if out.finished:
                    # 清理映射, 但保留 queue 供 handler 消费剩余
                    self._queues.pop(out.request_id, None)
                    self._loops.pop(out.request_id, None)
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, self.STOP_SENTINEL)
                    except RuntimeError:
                        pass
