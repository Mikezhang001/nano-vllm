import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 贪婪采样: temperatures==0 时直接 argmax (数值确定性)
        greedy_tokens = logits.argmax(dim=-1)
        # 随机采样 (Gumbel-max trick)
        # 保护: temperatures 里的 0 值参与 div 会 NaN, 用 max(t, 1e-6) 兜底
        safe_t = temperatures.clamp_min(1e-6).unsqueeze(dim=1)
        scaled_logits = logits.float().div(safe_t)
        probs = torch.softmax(scaled_logits, dim=-1)
        random_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        # 按 temperature 选一: t<=0 走 greedy, 否则 random
        is_greedy = temperatures <= 0
        return torch.where(is_greedy, greedy_tokens, random_tokens)
