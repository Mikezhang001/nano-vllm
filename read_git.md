# Nano-vLLM 源码阅读指南

## 📖 项目概述

Nano-vLLM 是一个轻量级的 vLLM 实现,核心目标是:
- **可读性**: ~1200 行 Python 代码实现完整推理引擎
- **高性能**: 通过 PagedAttention、Prefix Caching、Tensor Parallelism 等技术达到 vLLM 级别性能
- **教学性**: 清晰展示 LLM 推理引擎的核心原理

## 🗺️ 推荐阅读顺序

### 第一阶段: 理解入口和整体流程 (30分钟)

**1. `example.py`** - 从使用开始
```python
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(prompts, sampling_params)
```
- 理解 API 设计: 模仿 vLLM 的接口风格
- 注意 chat template 的使用方式

**2. `nanovllm/llm.py` → `nanovllm/engine/llm_engine.py`** - 主流程
```
LLM.generate() 
  → add_request() 创建 Sequence
  → step() 循环执行:
      1. scheduler.schedule() 选择 batch
      2. model_runner.run() 执行推理
      3. scheduler.postprocess() 处理输出
```

**关键概念**:
- **Prefill 阶段**: 首次处理完整 prompt,生成第一个 token
- **Decode 阶段**: 逐个生成后续 token
- **Scheduler**: 管理 waiting/running 队列,决定每步执行哪些序列

**3. `nanovllm/config.py`** - 配置参数
- `kvcache_block_size=256`: KV Cache 的分块大小
- `max_num_batched_tokens=16384`: 单次 batch 最多 token 数
- `gpu_memory_utilization=0.9`: GPU 显存利用率

---

### 第二阶段: 调度系统 (45分钟)

**4. `nanovllm/engine/sequence.py`** - 序列抽象
```python
class Sequence:
    block_size = 256          # 每个 block 的 token 数
    token_ids: list[int]      # 完整 token 序列
    block_table: list[int]    # 物理块号映射
    num_cached_tokens: int    # 已缓存的 token 数(prefix cache)
```
- **Block 切分**: 序列按 256 token 切分为多个 block
- **状态机**: WAITING → RUNNING → FINISHED

**5. `nanovllm/engine/scheduler.py`** - 调度策略
```python
def schedule():
    # 优先尝试 prefill (从 waiting 队列)
    if can_prefill():
        allocate_blocks()
        return seqs, is_prefill=True
    
    # 否则执行 decode (从 running 队列)
    else:
        check_memory()
        preempt_if_needed()  # 显存不足时抢占
        return seqs, is_prefill=False
```
- **Preemption**: 显存不足时,将低优先级序列回退到 waiting 队列
- **Batch 策略**: 最大化吞吐量同时避免 OOM

**6. `nanovllm/engine/block_manager.py`** - 内存管理 (核心!)
```python
class BlockManager:
    free_blocks: deque[int]     # 空闲物理块
    used_blocks: set[int]       # 已用物理块
    hash_to_block: dict         # prefix hash → block_id
```
**Prefix Caching 机制**:
1. 每个完整 block 计算 hash (基于 token_ids + 前序 block hash)
2. 新请求时,检查 hash 是否已存在
3. 命中则复用 KV Cache,只计算未缓存部分

```python
# 示例: 两个序列共享前缀
Seq A: [t0, t1, ..., t255, t256]  → block 3 (hash=0xabcd)
Seq B: [t0, t1, ..., t255, t257]  → 复用 block 3,新建 block 7
```

---

### 第三阶段: 模型执行 (60分钟)

**7. `nanovllm/engine/model_runner.py`** - 执行引擎
```python
def run(seqs, is_prefill):
    if is_prefill:
        input_ids, positions = prepare_prefill(seqs)
        # 构建 slot_mapping: 每个 token → KV Cache 槽位
        # 构建 cu_seqlens: 变长序列的偏移量
    else:
        input_ids, positions = prepare_decode(seqs)
        # 每个序列只有 1 个新 token
    
    logits = model(input_ids, positions)
    return sampler(logits)
```

**关键数据结构**:
- `slot_mapping`: 一维张量,记录每个 token 写入 KV Cache 的物理位置
- `cu_seqlens_q/k`: 累积序列长度,用于 varlen attention
- `block_tables`: 二维张量,记录每个序列使用的物理块号

**8. `nanovllm/models/qwen3.py`** - Qwen3 模型实现
```python
class Qwen3Attention:
    qkv_proj: QKVParallelLinear  # Q/K/V 合并投影
    o_proj: RowParallelLinear    # 输出投影
    rotary_emb: RotaryEmbedding  # RoPE 位置编码
    attn: Attention              # FlashAttention

class Qwen3MLP:
    gate_up_proj: MergedColumnParallelLinear  # gate+up 合并 (2*intermediate_size)
    down_proj: RowParallelLinear
    act_fn: SiluAndMul  # SwiGLU 激活
```

**Tensor Parallelism**:
- `ColumnParallelLinear`: 按列切分,输出维度分片
- `RowParallelLinear`: 按行切分,输入维度分片,需要 all_reduce
- `QKVParallelLinear`: 特殊处理 Q/K/V 的分片加载

**9. `nanovllm/layers/attention.py`** - Attention 实现 (核心!)
```python
class Attention:
    def forward(q, k, v):
        # 1. 写入 KV Cache
        store_kvcache(k, v, k_cache, v_cache, slot_mapping)
        
        # 2. 执行 Attention
        if is_prefill:
            if has_prefix_cache:
                k, v = k_cache, v_cache  # 使用完整缓存
                flash_attn_varlen_func(q, k, v, block_table=...)
            else:
                flash_attn_varlen_func(q, k, v, cu_seqlens=...)
        else:  # decode
            flash_attn_with_kvcache(q, k_cache, v_cache, block_table=...)
```

**Triton Kernel**: `store_kvcache_kernel` 将 K/V 写入分页的 KV Cache
```python
@triton.jit
def store_kvcache_kernel(key, k_cache, slot_mapping, D):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping + idx)
    if slot == -1: return  # padding token
    
    key = tl.load(key + idx * D + tl.arange(0, D))
    tl.store(k_cache + slot * D + tl.arange(0, D), key)
```

**10. `nanovllm/layers/activation.py`** - SwiGLU 激活
```python
class SiluAndMul:
    def forward(x):
        # x shape: (N, 2*intermediate_size)
        x, y = x.chunk(2, -1)  # 拆分为 gate 和 up
        return F.silu(x) * y   # SwiGLU
```

**11. `nanovllm/layers/rotary_embedding.py`** - RoPE 位置编码
```python
class RotaryEmbedding:
    # 预计算 cos/sin 缓存
    cos_sin_cache: (max_position, head_size)
    
    def forward(positions, q, k):
        cos, sin = cos_sin_cache[positions]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
```

**12. `nanovllm/layers/sampler.py`** - 采样策略
```python
class Sampler:
    def forward(logits, temperatures):
        # 1. Temperature scaling
        logits = logits / temperatures
        
        # 2. Softmax → 概率
        probs = softmax(logits)
        
        # 3. Gumbel-Max 采样
        # -log(-log(U)) 生成 Gumbel 噪声
        # argmax(log_probs + gumbel_noise) 等价于采样
        gumbel_noise = -log(-log(uniform(0,1)))
        tokens = argmax(log(probs) + gumbel_noise)
```

---

### 第四阶段: 辅助模块 (20分钟)

**13. `nanovllm/utils/loader.py`** - 权重加载
- 支持 safetensors 格式
- 处理 packed modules (QKV、gate_up 合并加载)
- 支持 tensor parallel 的分片加载

**14. `nanovllm/utils/context.py`** - 上下文管理
- 线程本地存储: `is_prefill`, `cu_seqlens`, `slot_mapping`, `block_tables`
- 在 forward 过程中传递元信息

**15. `nanovllm/layers/layernorm.py`** - RMSNorm
- 标准 RMSNorm 实现
- 支持 fused add + norm (减少显存访问)

**16. `nanovllm/layers/embed_head.py`** - Embedding & LM Head
- `VocabParallelEmbedding`: 词表并行,每个 GPU 存储部分词表
- `ParallelLMHead`: LM Head 并行,使用 all_reduce 聚合 logits

---

## 🔑 核心概念详解

### 1. PagedAttention (分页注意力)

**问题**: 传统 KV Cache 需要连续内存,导致碎片和浪费

**解决方案**: 
- 将 KV Cache 分成固定大小的 block (256 tokens)
- 使用 `block_table` 映射逻辑位置 → 物理位置
- 类似操作系统的虚拟内存机制

```python
# 逻辑视图
Sequence: [block_0, block_1, block_2]  # 连续逻辑块

# 物理视图
KV Cache: [block_7, block_3, block_15]  # 分散物理块
block_table = [7, 3, 15]
```

### 2. Prefix Caching (前缀缓存)

**场景**: 多个请求共享相同前缀 (如 system prompt)

**实现**:
```python
# 计算 block hash
hash = xxhash(token_ids + parent_hash)

# 查找缓存
if hash in hash_to_block:
    reuse_cached_block()
else:
    allocate_new_block()
```

**收益**: 
- 相同前缀只需计算一次
- 大幅减少 prefill 时间

### 3. Tensor Parallelism (张量并行)

**切分策略**:
```
QKV Projection (Column Parallel):
  输入: (N, hidden_size)
  GPU0: W_q[0:16], W_k[0:8], W_v[0:8]  → (N, 4096)
  GPU1: W_q[16:32], W_k[8:16], W_v[8:16] → (N, 4096)

Output Projection (Row Parallel):
  GPU0: W_o[0:4096] → (N, hidden_size)  ┐
  GPU1: W_o[4096:8192] → (N, hidden_size) ┘ all_reduce
```

### 4. Continuous Batching (连续批处理)

**传统批处理**: 固定 batch,短序列浪费计算
**连续批处理**: 动态添加/移除序列,最大化吞吐

```python
# Step 1: prefill seq_0, seq_1
batch = [seq_0(prompt), seq_1(prompt)]

# Step 2: decode seq_0, seq_1; prefill seq_2
batch = [seq_0(decode), seq_1(decode), seq_2(prefill)]

# Step 3: seq_0 finished, decode seq_1, seq_2
batch = [seq_1(decode), seq_2(decode)]
```

---

## 📊 性能优化技术

| 技术 | 文件 | 作用 |
|------|------|------|
| PagedAttention | `block_manager.py`, `attention.py` | 消除显存碎片 |
| Prefix Caching | `block_manager.py` | 复用前缀计算 |
| FlashAttention | `attention.py` | 高效 attention kernel |
| Triton Kernel | `attention.py` | 自定义 KV Cache 写入 |
| CUDA Graph | `model_runner.py` | 减少 kernel launch 开销 |
| torch.compile | 多个文件 | JIT 优化算子 |
| Tensor Parallel | `linear.py`, `embed_head.py` | 多 GPU 并行 |
| Fused Operators | `layernorm.py`, `activation.py` | 减少显存访问 |

---

## 🎯 关键代码片段

### 1. KV Cache 分配 (`model_runner.py`)
```python
def allocate_kv_cache():
    # 计算可用显存
    free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    
    # 计算 block 数量
    block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
    num_blocks = (total * 0.9 - used - peak) // block_bytes
    
    # 分配 KV Cache: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
    self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
```

### 2. Prefill 准备 (`model_runner.py`)
```python
def prepare_prefill(seqs):
    slot_mapping = []
    for seq in seqs:
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            block_id = seq.block_table[i]
            start = block_id * block_size
            end = start + block_size
            slot_mapping.extend(range(start, end))
    
    # slot_mapping: 每个 token → KV Cache 槽位
    return slot_mapping
```

### 3. Attention 执行 (`attention.py`)
```python
def forward(q, k, v):
    # 写入 KV Cache
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)
    
    if is_prefill:
        if has_prefix_cache:
            # 使用完整 KV Cache (包含缓存部分)
            flash_attn_varlen_func(q, k_cache, v_cache, block_table=block_tables)
        else:
            # 只使用当前计算的 K/V
            flash_attn_varlen_func(q, k, v, cu_seqlens=cu_seqlens)
    else:  # decode
        # 每个序列只有 1 个新 token
        flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, block_table=block_tables)
```

---

## 🔍 调试技巧

### 1. 打印关键张量形状
```python
# 在 attention.py 中添加
print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
print(f"k_cache: {k_cache.shape}, slot_mapping: {slot_mapping.shape}")
print(f"cu_seqlens_q: {cu_seqlens_q}, cu_seqlens_k: {cu_seqlens_k}")
```

### 2. 可视化 Block 分配
```python
# 在 scheduler.py 中添加
def print_block_status():
    print(f"Free blocks: {len(block_manager.free_blocks)}")
    print(f"Used blocks: {len(block_manager.used_blocks)}")
    for seq in running:
        print(f"Seq {seq.id}: {seq.block_table}")
```

### 3. 对比 vLLM 输出
```python
# 验证正确性
from vllm import LLM as VLLM
nano_llm = LLM(model_path)
vllm = VLLM(model_path)
assert nano_llm.generate(prompts) == vllm.generate(prompts)
```

---

## 📚 延伸阅读

1. **vLLM 论文**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. **FlashAttention**: "Fast and Memory-Efficient Exact Attention with IO-Awareness"
3. **Tensor Parallelism**: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
4. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"

---

## ❓ 常见问题

**Q1: 为什么 prefill 阶段也需要 Attention?**
A: Prefill 需要处理完整 prompt,每个 token 需要 attend 到前面的所有 token,生成第一个输出 token。

**Q2: block_size 为什么是 256?**
A: 平衡显存利用率和碎片率。太小导致管理开销大,太大导致碎片严重。256 是经验值。

**Q3: 什么时候触发 preemption?**
A: 当 running 队列中的序列需要新 block,但 free_blocks 不足时,会抢占低优先级序列。

**Q4: Tensor Parallel 的通信开销?**
A: Row Parallel 需要 all_reduce,Column Parallel 不需要。合理切分可以最小化通信。

**Q5: 如何支持其他模型?**
A: 参考 `qwen3.py`,实现对应的 model class,注册到 `models/` 目录,修改 `model_runner.py` 的模型选择逻辑。

---

## 🎓 学习检查点

完成阅读后,你应该能回答:

- [ ] 一个请求从输入到输出的完整流程是什么?
- [ ] PagedAttention 如何解决显存碎片问题?
- [ ] Prefix Caching 如何检测和复用缓存?
- [ ] Prefill 和 Decode 的区别是什么?
- [ ] Tensor Parallel 如何切分和聚合?
- [ ] KV Cache 的大小如何计算和分配?
- [ ] Scheduler 如何决定 batch 组成?
- [ ] RoPE 位置编码如何应用?
- [ ] SwiGLU 激活函数的实现?
- [ ] Gumbel-Max 采样的原理?

---

## 📝 代码统计

```
nanovllm/
├── engine/
│   ├── block_manager.py    # 143 lines - 内存管理
│   ├── llm_engine.py       # 107 lines - 主循环
│   ├── model_runner.py     # 252 lines - 执行引擎
│   ├── scheduler.py        # 71 lines  - 调度策略
│   └── sequence.py         # 65 lines  - 序列抽象
├── layers/
│   ├── activation.py       # 12 lines  - SwiGLU
│   ├── attention.py        # 112 lines - Attention
│   ├── embed_head.py       # 66 lines  - Embedding
│   ├── layernorm.py        # 32 lines  - RMSNorm
│   ├── linear.py           # 153 lines - 并行线性层
│   ├── rotary_embedding.py # 61 lines  - RoPE
│   └── sampler.py          # 15 lines  - 采样
├── models/
│   └── qwen3.py            # 180 lines - Qwen3 模型
├── utils/
│   ├── context.py          # 30 lines  - 上下文
│   └── loader.py           # 35 lines  - 权重加载
├── config.py               # 42 lines  - 配置
├── llm.py                  # 5 lines   - 入口
└── sampling_params.py      # 15 lines  - 采样参数

Total: ~1,200 lines
```

---

**Happy Reading! 🚀**
