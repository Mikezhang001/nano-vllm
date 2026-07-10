import atexit
from dataclasses import dataclass, field
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
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
        # ---------- Speculative Decoding / MTP 分支 ----------
        # 仅在 全 decode + 启用 spec/mtp + 每条 seq 都恰好有 1 个待写入 token (last_token 尚未入 KV) 时启用
        # MTP 与 spec 走同一条路径, 仅命名/接口层不同 (未来实现独立)
        mtp_enabled = getattr(self.model_runner, "mtp_enabled", False)
        spec_enabled = getattr(self.model_runner, "spec_enabled", False)
        use_accel = (
            is_decode_only
            and (spec_enabled or mtp_enabled)
            and all(s.num_committed_tokens == s.num_tokens - 1 for s in seqs)
        )
        # 保留原变量名, 后续代码统一用 use_spec 判断 (spec 和 mtp 语义等价)
        use_spec = use_accel
        # k 值: mtp 走 k_mtp, spec 走 k_spec
        k_accel = self.model_runner.k_mtp if mtp_enabled else self.model_runner.k_spec
        if use_spec:
            # 预扩容 target block_table: draft/mtp 会追加 k 个 token, 需要 block 能覆盖到 num_tokens+k
            k = k_accel
            bm = self.scheduler.block_manager
            for seq in seqs:
                needed_len = seq.num_tokens + k     # 保守估计: 最多再涨 k 个候选
                needed_blocks = (needed_len + seq.block_size - 1) // seq.block_size
                while len(seq.block_table) < needed_blocks:
                    if not bm.free_block_ids:
                        # 显存不足, 放弃 spec/mtp, 走常规
                        break
                    seq.block_table.append(bm._allocate_block())
                else:
                    continue
                # 若上面 break 了 (显存不足), 不走 spec/mtp
                use_spec = False
                break
        if use_spec:
            # spec 走 run_spec, mtp 走 run_mtp
            method = "run_mtp" if mtp_enabled else "run_spec"
            accepted_lists = self.model_runner.call(method, seqs)
            outputs: list[RequestOutput] = []
            # 因为 run_spec 内部已经把 accepted tokens 追加到 seq.token_ids 且更新 num_tokens
            # 我们需要在这里 "回滚 num_tokens 到调度前", 走 scheduler.postprocess 的正常流程
            # 但 scheduler.postprocess 假设每 seq 每 step 涨 1 token, 无法直接用
            # 因此: 手工处理 postprocess + 生成 RequestOutput
            for seq, accepted in zip(seqs, accepted_lists):
                # 计算 delta 部分 (本 step 新增的 token 列表)
                delta = list(accepted)
                # completion tokens 已经在 seq.token_ids 里 (run_spec 已 append)
                # 我们只做: block_manager.hash / may_append / EOS 判定
                # 逐个 token 检查 EOS/max_tokens, 一旦命中就截断
                # 注意: 需要维护 block_table 长度. 目前 append_token 仅在 scheduler.may_append 里扩展.
                # spec 场景下我们跳过 hash_blocks (对 prefix caching 不友好但简单)
                # BlockManager.may_append 按 seq.num_tokens 判断
                # 需要对本 step 涨的每个 token 都 may_append 一次
                # (先重置 seq 状态到 accept 前, 再逐个 append+may_append)
                # 反向: 直接根据当前 num_tokens 补齐 block_table
                self._extend_block_table_after_spec(seq)

                # EOS / max_tokens 截断: 找到第一个触发的位置
                cut_at = None
                for j, tid in enumerate(delta):
                    completion_len = (seq.num_tokens - len(delta) + j + 1) - seq.num_prompt_tokens
                    if (not seq.ignore_eos and tid == self.scheduler.eos) or completion_len >= seq.max_tokens:
                        cut_at = j + 1
                        break
                if cut_at is not None and cut_at < len(delta):
                    # 截断 seq.token_ids 到 cut_at
                    to_remove = len(delta) - cut_at
                    seq.token_ids = seq.token_ids[:-to_remove]
                    seq.num_tokens -= to_remove
                    seq.last_token = seq.token_ids[-1]
                    seq.num_committed_tokens = seq.num_tokens
                    # draft KV 里可能仍含被截断位置的 KV, 但 draft 有效前缀语义是
                    # "已含 [0..值-1] 的 KV". 截断后新 num_tokens < 原值, 我们只需保证
                    # num_draft_committed <= num_tokens (未来 catch-up 会正确处理更长范围).
                    # 保守设成 min(现值, num_tokens), 确保不变量.
                    seq.num_draft_committed_tokens = min(seq.num_draft_committed_tokens, seq.num_tokens)
                    delta = delta[:cut_at]

                is_done = (
                    (not seq.ignore_eos and delta and delta[-1] == self.scheduler.eos)
                    or seq.num_completion_tokens >= seq.max_tokens
                )
                outputs.append(RequestOutput(
                    request_id=seq.request_id,
                    prompt_token_ids=seq.prompt_token_ids,
                    token_ids=list(seq.completion_token_ids),
                    delta_token_ids=delta,
                    finished=is_done,
                ))
                if is_done:
                    seq.status = SequenceStatus.FINISHED
                    self.scheduler.block_manager.deallocate(seq)
                    if seq in self.scheduler.running:
                        self.scheduler.running.remove(seq)
                    self.scheduler.request_map.pop(seq.request_id, None)
            return outputs, num_tokens

        # ---------- 常规路径 ----------
        token_ids = self.model_runner.call("run", seqs, is_decode_only)
        outputs: list[RequestOutput] = []
        if token_ids is not None:
            for seq, tid in zip(seqs, token_ids):
                delta = [tid] if tid is not None else []
                outputs.append(RequestOutput(
                    request_id=seq.request_id,
                    prompt_token_ids=seq.prompt_token_ids,
                    token_ids=list(seq.completion_token_ids) + delta,
                    delta_token_ids=delta,
                    finished=False,
                ))
        self.scheduler.postprocess(seqs, token_ids)
        # 常规路径: 每 step forward 会写入 上一步的 last_token 到 KV.
        # 此步后 KV 中已有 seq[0..num_tokens-2] (旧 last_token 的 KV 刚在本 step forward 中写入),
        # 新采样的 token (num_tokens-1) 的 KV 尚未写入.
        # 所以 num_committed_tokens = num_tokens - 1
        for seq in seqs:
            seq.num_committed_tokens = max(0, seq.num_tokens - 1)
        for out, seq in zip(outputs, seqs):
            out.finished = seq.is_finished
        return outputs, num_tokens

    def _extend_block_table_after_spec(self, seq):
        """spec accept 后 seq.num_tokens 涨了 len(accepted), 补齐 block_table."""
        bm = self.scheduler.block_manager
        needed_blocks = seq.num_blocks
        while len(seq.block_table) < needed_blocks:
            # 直接从 free 拿 (spec 前 scheduler 已经预留了 1 block, 这里可能还差 0 or 1)
            if not bm.free_block_ids:
                raise RuntimeError("no free block for spec expansion, consider larger num_kvcache_blocks")
            seq.block_table.append(bm._allocate_block())

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
