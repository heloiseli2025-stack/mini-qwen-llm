"""Int4 Packing / Unpacking 工具（M3.1 实现）。

约定（pack 和 unpack 必须一致，否则 silent error）：
  packed[i] = w[8i+0] | w[8i+1]<<4 | ... | w[8i+7]<<28
  nibble 0 = 最低 4 bit = 最低 K index（little-endian nibble 顺序）
"""
import torch
from torch import Tensor


def pack_int4(x: Tensor) -> Tensor:
    """将 int4 tensor 打包为 int32（8 个 int4 → 1 个 int32）。

    Args:
        x: [..., K] int32，值域 0~15（uint4）

    Returns:
        [..., K//8] int32
    """
    assert x.shape[-1] % 8 == 0, f"最后一维必须是 8 的倍数，got {x.shape[-1]}"
    x = x.reshape(*x.shape[:-1], -1, 8).to(torch.int32)
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                          dtype=torch.int32, device=x.device)
    return (x << shifts).sum(dim=-1).to(torch.int32)


def unpack_int4(packed: Tensor) -> Tensor:
    """将 int32 解包为 int4 tensor。方向与 pack_int4 完全一致。

    Args:
        packed: [..., K//8] int32

    Returns:
        [..., K] int32，值域 0~15（uint4）
    """
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                          dtype=torch.int32, device=packed.device)
    x = (packed.unsqueeze(-1) >> shifts) & 0xF
    return x.reshape(*packed.shape[:-1], -1).to(torch.int32)
