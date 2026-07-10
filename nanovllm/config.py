import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # ---------- Speculative Decoding ----------
    # 若为 None 则关闭投机解码
    speculative_model: str | None = None
    # 每 step draft 生成的候选 token 数 (k), target verify 时长度 k+1
    num_speculative_tokens: int = 3
    # draft 模型独立的 KV cache 块数, -1 表示按 gpu_memory_utilization 剩余动态分配
    num_speculative_kvcache_blocks: int = -1
    # 内部字段: draft 模型的 hf_config, __post_init__ 填充
    draft_hf_config: AutoConfig | None = None
    # ---------- Multi-Token Prediction (MTP, DeepSeek-V3 style) ----------
    # 说明: 真实 MTP 中 target 模型自带 D 个 MTP module, 每个 module 输入
    # (target_hidden, next_token_embed), 链式产 D 个 next-token 候选.
    # 本 MVP 中没有能在 4xL20 上跑的原生 MTP 开源模型 (DeepSeek-V3 需 >600GB),
    # 因此用一个小模型 (通常与 draft 相同) 模拟 D 个 MTP module 的行为,
    # 复用现有 spec 的 draft-generate + target-verify 流水线, 仅在语义/接口层暴露 MTP.
    # 若同时设置 speculative_model 与 mtp_module, spec 优先关闭, 走 mtp 路径.
    mtp_module: str | None = None
    # MTP module 数量 D, target verify 时读 D+1 个位置
    mtp_num_heads: int = 3
    # MTP module 的 KV cache 块数, -1 动态分配
    num_mtp_kvcache_blocks: int = -1
    # 内部: MTP module 的 hf_config
    mtp_hf_config: AutoConfig | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.speculative_model is not None:
            assert os.path.isdir(self.speculative_model), f"draft model dir not found: {self.speculative_model}"
            assert self.num_speculative_tokens >= 1
            self.draft_hf_config = AutoConfig.from_pretrained(self.speculative_model)
            # 投机解码前提: draft/target 词表一致
            assert self.draft_hf_config.vocab_size == self.hf_config.vocab_size, (
                f"draft vocab_size ({self.draft_hf_config.vocab_size}) != "
                f"target vocab_size ({self.hf_config.vocab_size})"
            )
        if self.mtp_module is not None:
            assert os.path.isdir(self.mtp_module), f"mtp module dir not found: {self.mtp_module}"
            assert self.mtp_num_heads >= 1
            self.mtp_hf_config = AutoConfig.from_pretrained(self.mtp_module)
            assert self.mtp_hf_config.vocab_size == self.hf_config.vocab_size, (
                f"mtp vocab_size ({self.mtp_hf_config.vocab_size}) != "
                f"target vocab_size ({self.hf_config.vocab_size})"
            )
