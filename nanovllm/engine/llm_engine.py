import atexit
from dataclasses import dataclass, field
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


@dataclass
class RequestOutput:
    """本 step 一条序列的增量输出。

    - 若 finished=True, token_ids 是「完整」的 completion; delta_token_ids 是本 step 新增的
    - 若 finished=False, 二者语义相同 (完整已生成部分 + 本 step 增量)
    """
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]                # 到目前为止所有 completion token
    delta_token_ids: list[int] = field(default_factory=list)   # 本 step 新增部分 (0 或 1 个)
    finished: bool = False


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # ============ Continuous Batching 异步接口 ============

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> str:
        """异步添加一条请求, 立刻返回 request_id, 不阻塞。"""
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, request_id=request_id)
        self.scheduler.add(seq)
        return seq.request_id

    def abort_request(self, request_id: str) -> bool:
        """取消一条尚未完成的请求。"""
        return self.scheduler.abort(request_id)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def get_num_unfinished_requests(self) -> int:
        return self.scheduler.get_num_unfinished_requests()

    def step(self) -> tuple[list[RequestOutput], int]:
        """推进一步。返回:
          - request_outputs: 本 step 有变化的所有序列 (含新 token / finished)
          - num_tokens: 正数为 prefill token 数, 负数为 decode 序列数 (统计用)
        """
        seqs, is_decode_only = self.scheduler.schedule()
        if is_decode_only:
            num_tokens = -len(seqs)
        else:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs
                             if seq.num_cached_tokens + seq.num_scheduled_tokens != seq.num_tokens
                             or seq.num_scheduled_tokens > 1)
            if num_tokens == 0:
                num_tokens = -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_decode_only)
        # 生成增量输出前记录旧长度
        outputs: list[RequestOutput] = []
        if token_ids is not None:
            for seq, tid in zip(seqs, token_ids):
                delta = [tid] if tid is not None else []
                # 提前构造对象; finished 状态在 postprocess 后再修正
                outputs.append(RequestOutput(
                    request_id=seq.request_id,
                    prompt_token_ids=seq.prompt_token_ids,
                    token_ids=list(seq.completion_token_ids) + delta,
                    delta_token_ids=delta,
                    finished=False,
                ))
        self.scheduler.postprocess(seqs, token_ids)
        # 补齐 finished 状态
        for out, seq in zip(outputs, seqs):
            out.finished = seq.is_finished
        return outputs, num_tokens

    # ============ 向后兼容 ============

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 记录添加顺序, 输出时按此顺序返回
        req_ids: list[str] = []
        for prompt, sp in zip(prompts, sampling_params):
            req_ids.append(self.add_request(prompt, sp))

        results: dict[str, list[int]] = {}
        prefill_throughput = decode_throughput = 0.
        while self.has_unfinished_requests():
            t = perf_counter()
            step_outputs, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for out in step_outputs:
                if out.finished:
                    results[out.request_id] = out.token_ids
                    pbar.update(1)
        pbar.close()
        outputs = [
            {
                "text": self.tokenizer.decode(results[rid]),
                "token_ids": results[rid],
            }
            for rid in req_ids
        ]
        return outputs
