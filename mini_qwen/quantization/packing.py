"""Int4 Packing / Unpacking utilities (M3.1 implementation).

Convention (pack and unpack must be consistent, otherwise silent errors):
  packed[i] = w[8i+0] | w[8i+1]<<4 | ... | w[8i+7]<<28
  nibble 0 = lowest 4 bits = lowest K index (little-endian nibble order)
"""
import torch
from torch import Tensor


def pack_int4(x: Tensor) -> Tensor:
    """Pack an int4 tensor into int32 (8 int4 values -> 1 int32).

    Args:
        x: [..., K] int32, values in range 0~15 (uint4)

    Returns:
        [..., K//8] int32
    """
    assert x.shape[-1] % 8 == 0, f"Last dimension must be a multiple of 8, got {x.shape[-1]}"
    x = x.reshape(*x.shape[:-1], -1, 8).to(torch.int32)
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                          dtype=torch.int32, device=x.device)
    return (x << shifts).sum(dim=-1).to(torch.int32)


def unpack_int4(packed: Tensor) -> Tensor:
    """Unpack int32 to an int4 tensor. Direction is exactly consistent with pack_int4.

    Args:
        packed: [..., K//8] int32

    Returns:
        [..., K] int32, values in range 0~15 (uint4)
    """
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                          dtype=torch.int32, device=packed.device)
    x = (packed.unsqueeze(-1) >> shifts) & 0xF
    return x.reshape(*packed.shape[:-1], -1).to(torch.int32)
