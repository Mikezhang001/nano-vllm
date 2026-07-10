"""Multi-Token Prediction (MTP) 抽象接口 + 模拟实现.

真实 MTP (DeepSeek-V3 风格) 结构:

    target model
        └─ trunk (共享层, 输出 hidden_state h_t)
              ├─ main LM head            → next_token t+1
              ├─ MTP module 1  input=(h_t, embed(t+1))   → h'_{t+1} → t+2
              ├─ MTP module 2  input=(h'_{t+1}, embed(t+2)) → h'_{t+2} → t+3
              └─ ...  (chain, 一共 D 个 MTP module 一次预测 t+1..t+D+1)

    verify: target 再跑一次 forward 读 D+1 个位置, 与 MTP module 输出比对.

由于 4xL20 上无原生 MTP 开源模型 (DeepSeek-V3 需 >600GB), 本文件用一个小模型
(通常复用 spec 的 draft model) 作为 D 个 MTP module 的 stand-in, 通过以下方式
模拟 MTP 的语义:

  - MTPPredictor.predict_chain(seqs, num_predict=D)
       → 内部串行 decode D 次, 追加到 seq.token_ids
       → 相当于 chain 里 D 个 MTP module 各出一个 candidate
  - MTPVerifier 复用 ModelRunner._target_verify / _verify_and_accept.
       → verify 的第一个位置(q0) 逻辑上就是 "target main head 的 next token",
         对应真 MTP 中 target 直接产的 t+1; 我们的模拟把它归入 verify 阶段, 与
         MTP module 的输出统一比对.
  - 与 spec 的实现区别: **纯接口/config/log 层** — spec 强调 "外部 draft model 猜",
    MTP 强调 "target 自带的 head chain 猜". 未来把 predict_chain 内部换成 target
    真实的 MTP heads forward 即可.

Config 关键字段:
  Config.mtp_module: str        —— 模拟用的小模型路径 (通常与 draft 相同)
  Config.mtp_num_heads: int     —— chain 长度 D
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence


class MTPPredictor:
    """MTP 预测器 (模拟实现).

    真实 MTP: 从 target hidden_state 起, 用 D 个内置 head chain 出 D 个 candidate.
    模拟版: 用一个小模型 (ModelRunner.draft_model, 由 config.mtp_module 指定加载)
             做 D 次 1-token decode.

    调用约定:
      - 由 ModelRunner 持有 (每个 rank 一个)
      - predict_chain 会追加 D 个 token 到 seq.token_ids 与 seq.draft_token_ids
      - 也会写 MTP module 的 KV cache (通过 draft_model + draft_kv_cache)
      - 与 spec 复用同一份 kv_cache / block_table 状态
    """

    def __init__(self, runner: "ModelRunner", num_heads: int):
        self.runner = runner
        self.num_heads = num_heads

    def predict_chain(self, seqs: list["Sequence"]) -> None:
        """对每条 seq 追加 num_heads 个 MTP 候选 token.

        真实 MTP: 每次调用相当于跑完整 chain (D 个 module 串行, 各写一次 KV,
                  各出一个 token).
        模拟版: 复用 spec 的 _draft_step_and_append 循环 D 次.

        副作用:
            - seq.token_ids 追加 D 个 (预留供 verify)
            - seq.draft_token_ids 追加 D 个
            - MTP module KV cache 更新
        """
        for _ in range(self.num_heads):
            self.runner._draft_step_and_append(seqs)


class MTPVerifier:
    """MTP verify + accept 封装.

    真实 MTP: target 一次 forward 读 D+1 个位置, 每位置 next-token 逐个比对 chain
              输出, 找到第一个不一致的位置截断并用 target 修正.
    模拟版: 复用 ModelRunner._target_verify + _verify_and_accept, 语义完全等价.
    """

    def __init__(self, runner: "ModelRunner", num_heads: int):
        self.runner = runner
        self.num_heads = num_heads

    def verify_and_accept(
        self,
        seqs: list["Sequence"],
        original_num_tokens: list[int],
    ) -> list[list[int]]:
        """执行 target verify + 按位接受.

        Args:
            seqs: 已经追加了 num_heads 个 MTP 候选 token 的 seq 列表
            original_num_tokens: 进入本 step 前每 seq 的 num_tokens

        Returns:
            每 seq 本 step 接受的 token id 列表 (长度 1..D+1)
        """
        D = self.num_heads
        target_logits = self.runner._target_verify(seqs, D)  # (N*(D+1), vocab)
        temperatures = self.runner.prepare_sample(seqs)
        temperatures = temperatures.repeat_interleave(D + 1)
        target_tokens = self.runner.sampler(target_logits, temperatures).tolist()
        return self.runner._verify_and_accept(seqs, target_tokens, D, original_num_tokens)
