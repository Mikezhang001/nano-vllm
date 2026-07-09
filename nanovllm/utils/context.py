from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # True 表示走 varlen 路径(纯 prefill 或混合批); False 表示走纯 decode 单 token 的 with_kvcache 路径
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # 每条序列「用于采样的 token 在 hidden 中的下标」；
    # 对未完成 prefill 的 chunk 用 -1 表示无需采样
    logits_indices: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None, logits_indices=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables, logits_indices)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
