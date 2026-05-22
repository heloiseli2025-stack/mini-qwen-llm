"""AWQ 量化（M3.1 实现）。

实际实现：per-group min-max INT4 量化（不是完整 AWQ）。
  - 完整 AWQ 通过 activation magnitude 反向缩放 weight 保护 salient channels，PPL 更好。
  - 本模块实现 per-group min-max，用于 kernel 正确性测试；kernel 结构和完整 AWQ 完全相同。
  - 完整 AWQ 需要在 GPU 上运行 calibration 前向传播，留 quantize_awq 接口占位。
"""
import torch
from torch import Tensor

from mini_qwen.quantization.packing import pack_int4


def _per_group_minmax_quantize(
    w: Tensor,
    group_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-group min-max INT4 量化（无需 calibration data，用于 kernel 正确性测试）。

    Args:
        w:          [K, N] float32，K = in_features，N = out_features
        group_size: K 维分组大小，默认 128

    Returns:
        qweight: [K//8, N]   int32，packed int4，沿 K 维打包
        scales:  [K//group_size, N] bfloat16，per-group scale
        qzeros:  [K//group_size, N//8] int32，packed int4 zero，沿 N 维打包
    """
    K, N = w.shape
    assert K % group_size == 0, f"K={K} 必须整除 group_size={group_size}"
    assert N % 8 == 0, f"N={N} 必须是 8 的倍数（qzeros 沿 N 维打包）"

    w_g  = w.float().reshape(K // group_size, group_size, N)   # [G, gs, N]
    w_max = w_g.amax(dim=1)                                     # [G, N]
    w_min = w_g.amin(dim=1)                                     # [G, N]

    scales = (w_max - w_min) / 15.0                            # [G, N]
    scales = scales.clamp(min=1e-8)

    # zero point（int4 范围 0~15）使 0.0 精确表示
    zeros = (-w_min / scales).round().clamp(0, 15).to(torch.int32)  # [G, N]

    # 量化
    w_int = (
        (w_g / scales[:, None, :] + zeros[:, None, :].float())
        .round()
        .clamp(0, 15)
        .to(torch.int32)
        .reshape(K, N)
    )

    # pack qweight 沿 K 维：[K, N] -> [K//8, N]
    qweight = pack_int4(w_int.T).T   # T→[N,K], pack→[N,K//8], T→[K//8,N]

    # pack qzeros 沿 N 维：[G, N] -> [G, N//8]
    qzeros = pack_int4(zeros)

    return qweight, scales.to(torch.bfloat16), qzeros


def quantize_awq(model, calibration_data, group_size: int = 128):
    """完整 AWQ 量化算法（接口占位）。

    完整实现需要：
    1. 对每一个 Linear 层注册 hook，收集 calibration data 的输入 activation
    2. 计算 per-channel activation scale：abs_mean^alpha
    3. 对 weight 做 activation-aware rescaling，保护 salient channels
    4. 对 rescaled weight 做 per-group min-max 量化
    5. 将量化结果替换进 LinearW4A16

    参考：Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression"
    """
    raise NotImplementedError(
        "完整 AWQ 需要 GPU calibration（calibration_data 前向传播）。"
        "请用 _per_group_minmax_quantize 做 kernel 正确性测试，"
        "或用 autoawq 库离线量化真实模型权重后加载量化参数。"
    )
