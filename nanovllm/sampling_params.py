from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        # temperature == 0 表示贪婪采样 (argmax, 确定性)
        # temperature > 0 走 Gumbel-max 随机采样
        assert self.temperature >= 0, "temperature must be non-negative"
