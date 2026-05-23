import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match HF behavior: normalize in fp32, cast back to original dtype, then multiply by weight
        # Note: cast first then multiply weight (BF16 x BF16), rather than keeping full fp32 and casting at the end
        # The all-fp32 approach produces a 1-ULP difference vs HF in BF16, which amplifies across layers and causes routing divergence
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_fp32 * rms).to(input_dtype)
