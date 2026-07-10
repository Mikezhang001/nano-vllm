import os
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        # ---------- Speculative Decoding 相关 ----------
        self.spec_enabled = config.speculative_model is not None
        self.k_spec = config.num_speculative_tokens if self.spec_enabled else 0
        self.draft_model = None
        self.draft_kv_cache = None
        self.draft_num_blocks = 0

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 加载 draft 模型 (在 target KV 分配后, 剩余显存里再切一块给 draft KV)
        if self.spec_enabled:
            self._init_draft_model()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    # ============ Draft 模型初始化 ============

    def _init_draft_model(self):
        """加载 draft 模型 (Qwen3 系列, 与 target 同 tokenizer/vocab).

        - 与 target 共享 CUDA context, 独立参数
        - draft 只用于 decode, 独立分配 KV cache
        - draft 目前只支持 tp=1 (最常见场景), 不做 world_size > 1
        """
        assert self.world_size == 1, "speculative decoding with tp>1 not implemented yet"
        draft_cfg = self.config.draft_hf_config
        # 使用 target 的 dtype (强制统一, 简化)
        # 注意: draft 与 target 的 dtype 可能不同, 但 attention 层 dtype 由 tensor 传入决定
        self.draft_model = Qwen3ForCausalLM(draft_cfg)
        load_model(self.draft_model, self.config.speculative_model)
        # draft 独立 KV cache 分配 (方案 A: 完全独立)
        self._allocate_draft_kv_cache()

    def _allocate_draft_kv_cache(self):
        cfg = self.config
        draft_cfg = cfg.draft_hf_config
        num_kv_heads = draft_cfg.num_key_value_heads  # draft tp=1
        head_dim = getattr(draft_cfg, "head_dim", draft_cfg.hidden_size // draft_cfg.num_attention_heads)
        block_bytes = 2 * draft_cfg.num_hidden_layers * self.block_size * num_kv_heads * head_dim * draft_cfg.dtype.itemsize

        if cfg.num_speculative_kvcache_blocks > 0:
            num_blocks = cfg.num_speculative_kvcache_blocks
        else:
            # 用剩余显存的一部分, 至少能装下 max_num_seqs * max_model_len
            free, total = torch.cuda.mem_get_info()
            # 保守: 用剩余显存的 50%, 至少支持 max_num_seqs * max_model_len tokens
            budget = int(free * 0.5)
            num_blocks_by_free = budget // block_bytes
            min_tokens = cfg.max_num_seqs * cfg.max_model_len
            min_blocks = (min_tokens + self.block_size - 1) // self.block_size
            num_blocks = max(min_blocks, num_blocks_by_free)
            # 上限保护: 不超过 target 的 num_kvcache_blocks * 2
            num_blocks = min(num_blocks, cfg.num_kvcache_blocks * 2)
        assert num_blocks > 0
        self.draft_num_blocks = num_blocks
        self.draft_kv_cache = torch.empty(
            2, draft_cfg.num_hidden_layers, num_blocks, self.block_size, num_kv_heads, head_dim,
            dtype=draft_cfg.dtype,
        )
        layer_id = 0
        for module in self.draft_model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.draft_kv_cache[0, layer_id]
                module.v_cache = self.draft_kv_cache[1, layer_id]
                layer_id += 1

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        if self.spec_enabled:
            del self.draft_kv_cache
            self.draft_model = None
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        # warmup 走纯 prefill 路径 (无 block_tables)
        self.run(seqs, is_decode_only=False)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 若启用投机, 预留一部分显存给 draft 模型权重和 KV
        util = config.gpu_memory_utilization
        if config.speculative_model is not None:
            # 粗略估计: draft 权重 ~ 0.6B * 2 bytes = 1.2GB; draft KV 会再吃掉一些 (由 _allocate_draft_kv_cache 走剩余显存);
            # 这里给 target 少留一点空间, 让 draft 权重能进来
            util = min(util, 0.55)
        config.num_kvcache_blocks = int(total * util - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_mixed(self, seqs: list[Sequence]):
        """构造混合批 (prefill chunk + decode) 的输入 tensor。

        - decode 序列: num_scheduled_tokens == 1
        - prefill 序列: num_scheduled_tokens >= 1, 可能 < num_prompt_tokens (chunked)
        - logits_indices: 每条序列「采样位置」在 hidden 中的下标; 未完成 prefill 的中间 chunk 为 -1
        """
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        logits_indices: list[int] = []  # 仅包含「需采样」位置在 hidden 中的下标
        has_block_table = False
        cursor = 0  # hidden tensor 中的下标游标
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # 仅在调度后到达序列末尾时才需要采样 (chunked prefill 中间段跳过)
            if end == seq.num_tokens:
                logits_indices.append(cursor + seqlen_q - 1)
            cursor += seqlen_q

            if not seq.block_table:    # warmup 路径, 无需 slot_mapping
                continue
            has_block_table = True
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        block_tables = self.prepare_block_tables(seqs) if has_block_table else None
        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = (torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
                          if slot_mapping else None)
        logits_indices_t = (torch.tensor(logits_indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
                            if logits_indices else None)
        set_context(True, cu_seqlens_q_t, cu_seqlens_k_t, max_seqlen_q, max_seqlen_k,
                    slot_mapping_t, None, block_tables, logits_indices_t)
        return input_ids_t, positions_t

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], mask: list[bool] | None = None):
        if mask is None:
            temperatures = [seq.temperature for seq in seqs]
        else:
            temperatures = [seq.temperature for seq, m in zip(seqs, mask) if m]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_decode_only: bool):
        # 仅在「全 decode 单 token」且未禁用 Graph 且 batch 不超过 graph 容量时走 CUDA Graph
        if not is_decode_only or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_decode_only: bool) -> list[int | None]:
        if is_decode_only:
            input_ids, positions = self.prepare_decode(seqs)
            logits = self.run_model(input_ids, positions, is_decode_only=True)
            if self.rank == 0:
                temperatures = self.prepare_sample(seqs)
                token_ids = self.sampler(logits, temperatures).tolist()
            else:
                token_ids = None
            reset_context()
            return token_ids
        # 混合批 / 含 prefill chunk
        input_ids, positions = self.prepare_mixed(seqs)
        logits = self.run_model(input_ids, positions, is_decode_only=False)
        # logits 此时只包含「需要采样」的序列 (由 ParallelLMHead 用 logits_indices 抽取)
        if self.rank == 0:
            # 哪些序列本 step 出 token
            sample_mask = [s.num_cached_tokens + s.num_scheduled_tokens == s.num_tokens for s in seqs]
            sampled_ids: list[int | None] = [None] * len(seqs)
            if any(sample_mask):
                temperatures = self.prepare_sample(seqs, sample_mask)
                token_ids = self.sampler(logits, temperatures).tolist()
                idx = 0
                for i, m in enumerate(sample_mask):
                    if m:
                        sampled_ids[i] = token_ids[idx]
                        idx += 1
        else:
            sampled_ids = None
        reset_context()
        return sampled_ids

    # ============ Speculative Decoding: 核心 API ============

    @torch.inference_mode()
    def run_spec(self, seqs: list[Sequence]) -> list[list[int]]:
        """投机解码 step. 要求所有 seqs 都是 decode 状态 (draft 与 target KV 已就绪).

        流程 (贪婪等价, MVP):
          for i in 0..k:
            draft_step()  # draft 1-token decode, 追加到 seq.token_ids 与 draft KV
          target_verify()  # target 一次 q_len=k+1 的 varlen decode
          按位比对贪婪 argmax, 找到第一个不一致位置 j;
          seq 接受 j 个 draft token + 1 个 target 修正 token = j+1 个新 token
          rollback: 撤销 seq.token_ids/draft KV/target KV 里被拒的部分

        返回: 每条序列本 step 新接受的 token id 列表 (长度 1..k+1)
        """
        assert self.spec_enabled and self.rank == 0
        k = self.k_spec

        DEBUG = os.environ.get("SPEC_DEBUG", "") == "1"

        # 记录 verify 前的状态, 用于 rollback
        original_num_tokens = [seq.num_tokens for seq in seqs]

        # ---------- 1) draft 生成 k 个 token ----------
        for step in range(k):
            self._draft_step_and_append(seqs)
        if DEBUG:
            for i, seq in enumerate(seqs):
                print(f"[SPEC] seq{i} draft={seq.draft_token_ids} orig_N={original_num_tokens[i]}")

        # ---------- 2) target 一次 verify (q_len = k+1) ----------
        target_logits = self._target_verify(seqs, k)   # shape (N * (k+1), vocab)

        # ---------- 3) 采样 target 每个位置的 next token ----------
        temperatures = self.prepare_sample(seqs)               # (N,)
        temperatures = temperatures.repeat_interleave(k + 1)   # (N*(k+1),)
        target_tokens = self.sampler(target_logits, temperatures).tolist()  # list len N*(k+1)
        if DEBUG:
            for i, seq in enumerate(seqs):
                base = i * (k + 1)
                print(f"[SPEC] seq{i} target={target_tokens[base:base+k+1]}")

        # ---------- 4) 贪婪等价比对 + rollback ----------
        return self._verify_and_accept(seqs, target_tokens, k, original_num_tokens)

    def _verify_and_accept(
        self,
        seqs: list[Sequence],
        target_tokens: list[int],
        k: int,
        original_num_tokens: list[int],
    ) -> list[list[int]]:
        """通用 verify-and-accept 逻辑, 供 Speculative Decoding / MTP 复用.

        Args:
            seqs: 本 step 参与投机的 seq 列表. 要求 seq.draft_token_ids 已含本 step k 个 draft.
            target_tokens: target 一次 forward 得到的每条 seq 的 k+1 个 next-token 采样,
                          扁平化为长度 N*(k+1) 的 list, seq i 对应 [i*(k+1) : (i+1)*(k+1)].
            k: 每 seq draft 的 token 数.
            original_num_tokens: 每条 seq 进入本 spec step 前的 num_tokens (含 last_token,
                                 不含本 step 追加的 k 个 draft), 用于 rollback.

        Returns:
            每 seq 本 step 新接受的 token 列表 (长度 1..k+1).

        副作用:
            - 更新 seq.token_ids / num_tokens / last_token / num_committed_tokens
            - 清空 seq.draft_token_ids
            - 更新 self.spec_stats
        """
        if not hasattr(self, "spec_stats"):
            self.spec_stats = {"draft_total": 0, "draft_accepted": 0, "steps": 0}

        results: list[list[int]] = []
        for i, seq in enumerate(seqs):
            base = i * (k + 1)
            target_slice = target_tokens[base : base + k + 1]
            draft_slice = seq.draft_token_ids
            accepted: list[int] = []
            for j in range(k):
                if target_slice[j] == draft_slice[j]:
                    accepted.append(draft_slice[j])
                else:
                    accepted.append(target_slice[j])   # 修正 token
                    break
            else:
                accepted.append(target_slice[k])       # bonus token

            results.append(accepted)
            accepted_draft = k if len(accepted) == k + 1 else len(accepted) - 1
            self.spec_stats["draft_total"] += k
            self.spec_stats["draft_accepted"] += accepted_draft

            # rollback + apply
            N_orig = original_num_tokens[i]
            seq.token_ids = seq.token_ids[:N_orig]
            for tid in accepted:
                seq.token_ids.append(tid)
            seq.num_tokens = N_orig + len(accepted)
            seq.last_token = seq.token_ids[-1]
            seq.draft_token_ids = []
            seq.num_committed_tokens = max(0, seq.num_tokens - 1)
        self.spec_stats["steps"] += 1
        return results

    def _draft_step_and_append(self, seqs: list[Sequence]):
        """让 draft 模型对每个 seq 生成 1 个 token, 追加到 seq.token_ids + draft KV.

        MVP 策略: 每次调用时都对每条 seq 重新 lazy prefill draft KV.
        - 优点: 完全避免 draft KV rollback 逻辑, 正确性最简单
        - 缺点: draft 侧每 step 有 O(N) forward 开销. 0.6B 模型上仍显著快于 8B 一次 decode
        - 每 seq 首次访问时 _draft_num_committed 会被重置, 全量重跑
        """
        # 首次进入时为每条 seq 分配 draft block table (lazy)
        for seq in seqs:
            if not hasattr(seq, "_draft_block_table") or not seq._draft_block_table:
                num_blocks_per_seq = (self.config.max_model_len + self.block_size - 1) // self.block_size
                start_block = seq.seq_id * num_blocks_per_seq
                if start_block + num_blocks_per_seq > self.draft_num_blocks:
                    raise RuntimeError(
                        f"draft KV cache too small for seq_id={seq.seq_id}, "
                        f"need block [{start_block}, {start_block + num_blocks_per_seq}), "
                        f"but only {self.draft_num_blocks} draft blocks"
                    )
                seq._draft_block_table = list(range(start_block, start_block + num_blocks_per_seq))
                seq._draft_num_committed = 0
            # 检查 draft KV 是否与当前 seq.num_tokens 对齐 (每次都重新对齐, MVP)
            # 若 draft 已 commit 长度 != seq.num_tokens - 1 (上一步 target 修正 token 后需要补写),
            # 则重新 prefill 到 num_tokens - 1 位置 (最后 1 个 token 会被本次 draft step 作为 query 写入)
            # 实际做法: 每步都全量 prefill (0..num_tokens-1), 最后 1 个 token 由这次 decode step 写.
            # 更简单: 直接把 [0..num_tokens-1] 全写好, 然后 decode 1 步 (query=last_token, 写位置 num_tokens-1)
            # 但 last_token 的 KV 也需要写入.
            # 决策: 每步全量重写 [0..num_tokens-1] (含 last), 然后 decode 步 forward 1 个新位置
            self._draft_full_prefill_seq(seq)

        # 现在 draft KV 里 [0..num_tokens-1] 都已就绪. decode 1 步生成 next token.
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        max_block_len = max(len(seq._draft_block_table) for seq in seqs)
        block_tables = []
        for seq in seqs:
            # 新位置 = num_tokens (即将写入的 slot); query 用当前 last_token
            # 但 last_token 的 KV 已经在 _draft_full_prefill_seq 写好了 (位置 num_tokens-1)
            # decode 1 步意味着: query=seq[num_tokens-1] 位置的 hidden -> next token logit
            # 但 last_token 已在 KV, 我们不需要再写入 -- 用 flash_attn_with_kvcache 走纯 decode 路径
            # 但那需要 num_tokens 已经是新 token 之后的; 这里我们要的是 P(next | seq[0..num_tokens-1])
            # flash_attn_with_kvcache 会用 cache_seqlens 判断有效 KV 长度
            # 所以: cache_seqlens = num_tokens (含 last_token), input 是 last_token 作 query
            # slot_mapping = -1 (last_token KV 已存在)
            tok = seq.last_token
            pos = len(seq) - 1   # last_token 位置
            input_ids.append(tok)
            positions.append(pos)
            context_lens.append(len(seq))
            slot_mapping.append(-1)   # 不重复写
            bt = seq._draft_block_table + [-1] * (max_block_len - len(seq._draft_block_table))
            block_tables.append(bt)

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables_t = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(False, slot_mapping=slot_mapping_t, context_lens=context_lens_t, block_tables=block_tables_t)
        hidden = self.draft_model(input_ids_t, positions_t)
        logits = self.draft_model.compute_logits(hidden)
        temperatures = self.prepare_sample(seqs)
        next_tokens = self.sampler(logits, temperatures).tolist()
        reset_context()

        # 追加到 seq
        for seq, tid in zip(seqs, next_tokens):
            seq.token_ids.append(tid)
            seq.num_tokens += 1
            seq.last_token = tid
            seq.draft_token_ids.append(tid)

    def _draft_full_prefill_seq(self, seq: Sequence):
        """把 seq.token_ids[0..num_tokens-1] 全量写入 draft KV.

        MVP 策略: 每次 spec step 前都调用一次, 完全避免 rollback 复杂度.
        """
        n = seq.num_tokens
        assert n >= 1
        slot_mapping = []
        for pos in range(n):
            block_idx = pos // self.block_size
            slot_in_block = pos % self.block_size
            slot_mapping.append(seq._draft_block_table[block_idx] * self.block_size + slot_in_block)

        input_ids_t = torch.tensor(seq.token_ids[:n], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(list(range(n)), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor([0, n], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor([0, n], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables_t = torch.tensor([seq._draft_block_table], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(True,
                    cu_seqlens_q=cu_seqlens_q_t, cu_seqlens_k=cu_seqlens_k_t,
                    max_seqlen_q=n, max_seqlen_k=n,
                    slot_mapping=slot_mapping_t, block_tables=block_tables_t)
        _ = self.draft_model(input_ids_t, positions_t)
        reset_context()
        seq._draft_num_committed = n

    def _target_verify(self, seqs: list[Sequence], k: int) -> torch.Tensor:
        """target 一次 forward 验证 k 个 draft token.

        进入前的不变量:
          - seq.num_committed_tokens = N (target KV 里已有 seq[0..N-1] 的 K/V)
          - seq.token_ids 长度 = N + 1 + k
            (N-1 是历史, N 是上一 step 采样出但 KV 未写的 last_token, N+1..N+k 是 draft 的 k 个)
          - 即 seq.num_tokens == N + 1 + k

        forward:
          - query 序列 = seq[N .. N+k] (共 k+1 个 token), 全部写入 KV
          - key/value 长度 = N + k + 1 (含新写入位置)
          - 得到每个 query 位置对应的 next-token logits, 共 k+1 个
        """
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping: list[int] = []
        max_seqlen_q = k + 1
        max_seqlen_k = 0

        for seq in seqs:
            N = seq.num_committed_tokens
            assert seq.num_tokens == N + 1 + k, (
                f"target_verify invariant broken: num_tokens={seq.num_tokens} "
                f"num_committed={N} k={k}"
            )
            q_start = N            # 从 last_token 开始
            q_end = N + k + 1      # exclusive, 含 k 个 draft
            input_ids.extend(seq.token_ids[q_start:q_end])
            positions.extend(range(q_start, q_end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + (q_end - q_start))
            cu_seqlens_k.append(cu_seqlens_k[-1] + q_end)   # key 长度 = q_end (即 0..q_end-1)
            max_seqlen_k = max(max_seqlen_k, q_end)
            for pos in range(q_start, q_end):
                block_idx = pos // self.block_size
                slot_in_block = pos % self.block_size
                assert block_idx < len(seq.block_table), (
                    f"block_table too short: pos={pos} block_idx={block_idx} "
                    f"len(block_table)={len(seq.block_table)}"
                )
                slot_mapping.append(seq.block_table[block_idx] * self.block_size + slot_in_block)

        block_tables_t = self.prepare_block_tables(seqs)
        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        logits_indices_t = torch.arange(len(input_ids), dtype=torch.int64, device="cuda")

        set_context(True, cu_seqlens_q_t, cu_seqlens_k_t, max_seqlen_q, max_seqlen_k,
                    slot_mapping_t, None, block_tables_t, logits_indices_t)
        hidden = self.model(input_ids_t, positions_t)
        logits = self.model.compute_logits(hidden)   # shape (N*(k+1), vocab)
        reset_context()
        return logits

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
