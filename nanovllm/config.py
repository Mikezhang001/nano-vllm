import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:#batch是时间维度上的迭代, 1 依赖于 0
    model: str
    max_num_batched_tokens: int = 16384 #一个batch中最多token
    max_num_seqs: int = 512 #一个batch中多少sequence
    max_model_len: int = 4096 #一个sequence的token长度
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len


# 1. 传统的模型推理（带 Padding）
# 传统的输入形状确实是 [batch_size, seq_len, hidden_dim]。

# 问题：如果 Sequence A 长度是 10，Sequence B 长度是 2，为了凑成规整的矩阵，B 必须补充 8 个无用的 0（Padding）。这会白白浪费大量 GPU 算力和显存。
# 2. vLLM 的高性能推理（展平去 Padding）
# 为了消灭 Padding，vLLM 采用了 Flatten（展平） 技术。
# 它直接把 batch_size 和 seq_len 这两个维度合并了，把所有不同长度的 Sequence 首尾相连，拼成一条线。

# 输入张量形状变成了：[total_num_tokens, hidden_dim]。
# 举例：如果 Batch 里面有 3 个 Sequence，长度分别是 10、2、5。传统做法是变成 [3, 10, hidden_dim]（共 30 个 token，包含大量 Padding）；而 vLLM 的做法是合并成 [17, hidden_dim]（纯有效 token）。
# 这就是为什么配置里使用的是 max_num_batched_tokens = 16384，它限制的是这个一维数组的总长度。
# 3. 那 Attention 怎么算？
# 既然全展平了，模型怎么区分哪几个 Token 属于序列 A，哪几个属于序列 B？
# 引擎会额外维护一个偏移量记录表（如 cu_seq_lens = [0, 10, 12, 17]），并在计算 Attention 时调用高度优化的算子（如 FlashAttention 或 xFormers）