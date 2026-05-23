"""Triton Fused RMSNorm Kernel (optional optimization; skip if the PyTorch implementation is sufficient)."""
import torch


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    raise NotImplementedError("Implement on demand after M2")
