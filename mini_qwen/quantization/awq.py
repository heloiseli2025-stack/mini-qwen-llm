"""AWQ quantization (M3.1 implementation).

Actual implementation: per-group min-max INT4 quantization (not full AWQ).
  - Full AWQ inversely scales weights by activation magnitude to protect salient channels, yielding better PPL.
  - This module implements per-group min-max for kernel correctness testing; the kernel structure is identical to full AWQ.
  - Full AWQ requires running calibration forward passes on GPU; quantize_awq is a placeholder interface.
"""
import torch
from torch import Tensor

from mini_qwen.quantization.packing import pack_int4


def _per_group_minmax_quantize(
    w: Tensor,
    group_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-group min-max INT4 quantization (no calibration data needed; used for kernel correctness testing).

    Args:
        w:          [K, N] float32, K = in_features, N = out_features
        group_size: Group size along the K dimension, default 128

    Returns:
        qweight: [K//8, N]          int32, packed int4, packed along K dimension
        scales:  [K//group_size, N] bfloat16, per-group scale
        qzeros:  [K//group_size, N//8] int32, packed int4 zeros, packed along N dimension
    """
    K, N = w.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert N % 8 == 0, f"N={N} must be a multiple of 8 (qzeros packed along N dimension)"

    w_g  = w.float().reshape(K // group_size, group_size, N)   # [G, gs, N]
    w_max = w_g.amax(dim=1)                                     # [G, N]
    w_min = w_g.amin(dim=1)                                     # [G, N]

    scales = (w_max - w_min) / 15.0                            # [G, N]
    scales = scales.clamp(min=1e-8)

    # zero point (int4 range 0~15) so that 0.0 is represented exactly
    zeros = (-w_min / scales).round().clamp(0, 15).to(torch.int32)  # [G, N]

    # quantize
    w_int = (
        (w_g / scales[:, None, :] + zeros[:, None, :].float())
        .round()
        .clamp(0, 15)
        .to(torch.int32)
        .reshape(K, N)
    )

    # pack qweight along K dimension: [K, N] -> [K//8, N]
    qweight = pack_int4(w_int.T).T   # T->[N,K], pack->[N,K//8], T->[K//8,N]

    # pack qzeros along N dimension: [G, N] -> [G, N//8]
    qzeros = pack_int4(zeros)

    return qweight, scales.to(torch.bfloat16), qzeros


def quantize_awq(model, calibration_data, group_size: int = 128):
    """Full AWQ quantization algorithm (placeholder interface).

    Full implementation requires:
    1. Register hooks on every Linear layer to collect input activations from calibration data
    2. Compute per-channel activation scale: abs_mean^alpha
    3. Apply activation-aware rescaling to weights to protect salient channels
    4. Apply per-group min-max quantization to rescaled weights
    5. Replace quantized results into LinearW4A16

    Reference: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression"
    """
    raise NotImplementedError(
        "Full AWQ requires GPU calibration (forward pass on calibration_data). "
        "Use _per_group_minmax_quantize for kernel correctness testing, "
        "or use the autoawq library to offline-quantize real model weights and load quantization parameters."
    )
