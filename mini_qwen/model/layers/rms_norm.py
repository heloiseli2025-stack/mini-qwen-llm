import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 与 HF 保持一致：fp32 归一化，cast 回原 dtype 后再乘 weight
        # 注意：先 cast 再乘 weight（BF16×BF16），而非全程 fp32 最后 cast
        # 全程 fp32 方案会与 HF 在 BF16 下产生 1 ULP 差异，经过多层放大后导致路由不同
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_fp32 * rms).to(input_dtype)
