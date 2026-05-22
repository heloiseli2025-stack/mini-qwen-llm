"""Triton Fused RMSNorm Kernel（可选优化，PyTorch 实现已够用时跳过）。"""
import torch


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    raise NotImplementedError("按需在 M2 之后实现")
