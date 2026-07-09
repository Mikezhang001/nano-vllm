from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """混合批 (chunked prefill) 调度器。

    每个 step 同时调度 decode 序列和 prefill chunk:
      1) 先把 running 队列中能 append KV 的序列各预占 1 token (decode)
      2) 用剩余 token 预算给 waiting 队列里的序列切 prefill chunk; 任意序列都可被 chunk
      3) prefill 完成的最后一个 chunk 会触发采样, 产出第一个 decode token

    is_decode_only=True 时本 batch 全为 decode 单 token, 可走 CUDA Graph 快速路径;
    否则 (含任何 prefill chunk) 走 varlen 路径。
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # request_id -> Sequence, 用于 abort / 状态查询
        self.request_map: dict[str, Sequence] = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def has_unfinished_requests(self) -> bool:
        return not self.is_finished()

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        self.request_map[seq.request_id] = seq

    def abort(self, request_id: str) -> bool:
        """外部主动取消一条请求。返回是否命中。"""
        seq = self.request_map.get(request_id)
        if seq is None or seq.is_finished:
            return False
        if seq in self.waiting:
            self.waiting.remove(seq)
        if seq in self.running:
            self.running.remove(seq)
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FINISHED
        self.request_map.pop(seq.request_id, None)
        return True

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs: list[Sequence] = []
        decode_seqs: list[Sequence] = []
        num_batched_tokens = 0

        # ---------- 1) decode: 给 running 中的序列各占 1 token ----------
        # 维持 FIFO; KV 不足时按 LIFO 抢占 running 队尾
        while self.running and len(decode_seqs) < self.max_num_seqs:
            if num_batched_tokens + 1 > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            preempted = False
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    preempted = True
                    break
            if preempted:
                break
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            decode_seqs.append(seq)
            num_batched_tokens += 1
        # 暂不放回 running, 等 postprocess 之后由调用方维持
        scheduled_seqs.extend(decode_seqs)

        # ---------- 2) prefill: 用剩余 token 预算切 chunk ----------
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break  # KV 不足
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.block_manager.allocate(seq, num_cached_blocks)
            else:
                # 被抢占恢复 或 chunked prefill 中途
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            scheduled_seqs.append(seq)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 本 step 完成 prefill, 出队列
                self.waiting.popleft()
            else:
                # 还需要继续 chunk, 不出队列, 也不再调度后续 waiting (保证 FIFO 简单语义)
                break

        # 放回 running 头部, 保持 FIFO
        if decode_seqs:
            self.running.extendleft(reversed(decode_seqs))

        assert scheduled_seqs, "no sequence scheduled; possibly all preempted"

        is_decode_only = len(decode_seqs) == len(scheduled_seqs)
        return scheduled_seqs, is_decode_only

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int | None]):
        """处理本 step 各序列的产出。

        token_ids[i] 为 None 表示该序列本 step 没有采样位置 (chunked prefill 中途)。
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if token_id is None:
                # chunked prefill 中间段
                continue
            # 出了一个新 token
            seq.append_token(token_id)
            # 若刚完成 prefill, 加入 running 队列
            if seq.status == SequenceStatus.WAITING:
                seq.status = SequenceStatus.RUNNING
                seq.is_prefill = False
                self.running.append(seq)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                self.request_map.pop(seq.request_id, None)
