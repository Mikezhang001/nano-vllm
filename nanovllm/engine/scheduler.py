from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:#环：等待 -> 预填充建缓存 -> 逐字解码 -> (如果爆显存) -> 销毁缓存回退到等待 -> 重新预填充建缓存 -> 继续逐字解码。
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens#总的减去cache命中的,就是蹭其他sequence的prefill的值
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)#（全局大名单）：是调度器的一个长期的全局属性
            scheduled_seqs.append(seq)# （本轮小名单）：是 schedule 函数里的一个临时的局部变量
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()#左拿
            while not self.block_manager.can_append(seq):#当要申请新块时候不够，一直驱逐
                if self.running:
                    self.preempt(self.running.pop())# 还有其他人在排队，那就从队尾（最晚来的）抓一个倒霉蛋，没收它的内存（preempt）
                else:
                    self.preempt(seq) # 所有人全被踢光了，只剩当前这个请求自己了，内存居然还不够！
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))#左放
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):#显存耗尽（OOM)
        seq.status = SequenceStatus.WAITING#改状态
        self.block_manager.deallocate(seq)#KV cache回收
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)#seqs 和 token_ids 对应的， 每轮一个
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
