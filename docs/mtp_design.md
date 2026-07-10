# MTP (Multi-Token Prediction) 设计文档

本文档记录 nano-vllm 中 MTP (Multi-Token Prediction) 特性的设计与模拟实现。

## 1. 背景

**MTP** 是 DeepSeek-V3 提出的推理加速技术：模型自带 D 个 MTP module，可以一次前向预测未来 D+1 个 token，用于替代传统 speculative decoding（外部 draft model 猜测 + target 验证）。

真实 MTP 结构：

```
                    target model (shared trunk)
                              ↓
                       hidden state h_t
                              ↓
     ┌────────────────────────┼────────────────────────┐
     ↓                        ↓                        ↓
 main head (LM head)     MTP module 1              MTP module 2
     ↓                   input: (h_t, embed(t+1))  input: (h'_{t+1}, embed(t+2))
   token t+1              → h'_{t+1}                → h'_{t+2}
                          → token t+2               → token t+3
```

一次 forward：main head 出 t+1，MTP module chain 出 t+2..t+D+1。target 再跑一次 verify（读 D+1 个位置）判断接受。

## 2. 与传统 Speculative Decoding 的差异

| 维度 | Speculative Decoding | MTP |
|---|---|---|
| draft 来源 | 独立小模型（如 Qwen3-0.6B） | target 自带的 D 个 head |
| trunk 共享 | ❌ 完全独立 | ✅ 共享 target 主体 |
| KV cache | draft 独立维护 | 依托 target hidden，无需独立 KV |
| 参数量开销 | draft model 全参 | 每 head 一层 transformer |
| 与 target 分布契合 | 训练不共同，接受率较低 | 联合训练，接受率高 |

## 3. 本项目模拟策略

### 3.1 硬件约束

4× NVIDIA L20 (46GB × 4 = 184GB)。原生支持 MTP 的开源模型有：
- **DeepSeek-V3**：685GB（671B main + 14B MTP module），显存不够
- **DeepSeek-V2-Lite**：无 MTP module
- 其他小模型：普遍无 MTP

**结论**：4×L20 上无原生 MTP 模型可跑。

### 3.2 模拟方案

用一个小模型（默认 Qwen3-0.6B）作为 D 个 MTP module 的 **stand-in**，通过接口封装暴露 MTP 语义：

- **MTPPredictor**：语义等价于"target 的 chain of MTP modules 逐个预测下一 token"
  - 真实实现：调用 target 内置 head chain（未来实现）
  - 模拟实现：调用小模型 D 次串行 decode（当前）
- **MTPVerifier**：语义等价于"target 再跑一次读 D+1 个位置验证"
  - 复用 `ModelRunner._target_verify` + `_verify_and_accept`

### 3.3 与 spec 复用底层

`MTPPredictor.predict_chain` 内部直接调用 `ModelRunner._draft_step_and_append` 循环 D 次；`MTPVerifier` 复用 `_target_verify` + `_verify_and_accept`。因此 **MTP 与 spec 的输出完全一致**（bit-exact），差异仅在：
- Config 字段（`mtp_module` vs `speculative_model`）
- 接口封装（`MTPPredictor` / `MTPVerifier` 存在与否）
- 语义命名（`run_mtp` vs `run_spec`）

## 4. 代码结构

```
nanovllm/
├── config.py                  # 新增 mtp_module / mtp_num_heads / mtp_hf_config
├── engine/
│   ├── mtp.py                # 新增: MTPPredictor + MTPVerifier
│   ├── model_runner.py       # 新增: _init_mtp_module, run_mtp
│   └── llm_engine.py         # step() 里加 mtp 分支 (与 spec 共享路径)
tests/
└── test_mtp.py               # end-to-end 测试 (baseline vs spec vs mtp 三方对比)
```

## 5. 未来切换为真实 MTP

当有真实 MTP 模型（如 DeepSeek-V3 量化版）可用时，切换路径：

1. **Config**：`mtp_module` 改为指向 target 模型自身（target 加载时同时加载 MTP module 权重）
2. **`_init_mtp_module`**：改为从 target 模型抽取 MTP module 层，无需独立加载
3. **`MTPPredictor.predict_chain`**：内部改为
   - 输入 target 最后一层 hidden（需要 ModelRunner 保留 target forward 的 hidden 输出）
   - chain 调用 target 的 D 个 MTP module 各产一个 hidden + token
   - 输出 D 个候选 token
4. **KV cache**：MTP module 本身可能需要独立的小 KV cache（每 head 一层），复用现有 `draft_kv_cache` 存储路径

`MTPVerifier` / `run_mtp` / `LLMEngine.step` 里的 mtp 分支**不需改动**，因为它们是 MTP 无关的通用逻辑。

## 6. 验证结果

`tests/test_mtp.py` (4 prompts, k=D=3, max_tokens=30, Qwen3-8B target + Qwen3-0.6B MTP module)：

| 模式 | tok/s | accept rate | 与 baseline 一致性 |
|---|---|---|---|
| baseline | 68.8 | N/A | — |
| spec (Qwen3-0.6B draft) | 33.0 | 62.7% | 4/4 完全一致 |
| **mtp (Qwen3-0.6B module)** | 33.3 | 62.7% | 4/4 完全一致 |

MTP 与 spec bit-exact 一致（预期，共用底层实现）。速率上暂低于 baseline（draft launch overhead 主导，Phase B 已诊断）。

## 7. 限制与已知问题

- **仅模拟**：不使用 target 内置 MTP head，无法体现真实 MTP 的 trunk 共享收益
- **不支持 tp>1**：MTP module 加载路径与 draft 一致，仅支持 tp=1
- **不支持 CUDA Graph**：spec/mtp 走 eager mode 路径
- **速率暂未提升**：draft 每 step k 次 forward 各 bs=N 小 batch，GPU 利用率极低（Phase B 分析）
