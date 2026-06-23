import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)#triton竟然是simd
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)#可以看出key目前是连续的
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)#放就根据映射表不连续了
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel(): #排除warmup 阶段
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # 排除 warmup 和 prefix cache(无prefix cache)
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,#block_tables空不空，决定如何读取k和v
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
                                       #context.block_tables需要统一补全为最长seq的block数，见prepare_block_tables方法
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o


    # q=torch.randn(6, 2, 4),  # 总共有 6 个 Token，每个 Token 的 Query 是 [num_heads, head_dim]
    # k=torch.randn(6, 2, 4),  # Key 的形状与 Query 相同
    # v=torch.randn(6, 2, 4),  # Value 的形状与 Query 相同
    # max_seqlen_q=3,          # Query 的最大序列长度（Batch 中最长的句子长度）
    # cu_seqlens_q=torch.tensor([0, 2, 5, 6]),  # Query 的累积序列长度
    # max_seqlen_k=3,          # Key 的最大序列长度（通常等于 max_seqlen_q）
    # cu_seqlens_k=torch.tensor([0, 2, 5, 6]),  # Key 的累积序列长度
    # softmax_scale=1 / 2,     # Softmax 缩放因子（1 / sqrt(head_dim)）
    # causal=True,             # 启用因果掩码，防止看到未来的 Token
    # block_table=None         # 分页内存表（这里假设没有分页）