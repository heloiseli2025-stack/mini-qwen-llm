"""W4A16 quantized Linear layer (M3 implementation)."""
import torch
import torch.nn as nn

from mini_qwen.kernels.w4a16_gemm import w4a16_gemm


class LinearW4A16(nn.Module):
    """Quantized Linear with packed int4 weights and bf16 activations.

    Args:
        in_features:  Input dimension K
        out_features: Output dimension N
        group_size:   Quantization group size, default 128
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size

        K, N, gs = in_features, out_features, group_size
        G = K // gs

        # packed int4 weights; shape convention matches the w4a16_gemm signature
        self.register_buffer("qweight", torch.zeros(K // 8, N,     dtype=torch.int32))
        self.register_buffer("scales",  torch.zeros(G,      N,     dtype=torch.bfloat16))
        self.register_buffer("qzeros",  torch.zeros(G,      N // 8, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, K] bf16 → [*, N] bf16。"""
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        out = w4a16_gemm(x2d, self.qweight, self.scales, self.qzeros, self.group_size)
        return out.reshape(*orig_shape[:-1], self.out_features)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
    ) -> "LinearW4A16":
        """Quantize a bf16/fp32 nn.Linear to W4A16.

        Args:
            linear:     Original nn.Linear (weight [N, K])
            group_size: Quantization group size

        Returns:
            LinearW4A16 instance with weight replaced by packed int4
        """
        from mini_qwen.quantization.awq import _per_group_minmax_quantize

        w = linear.weight.data.float()    # [N, K] -> transposed to [K, N]
        w = w.T.contiguous()              # [K, N] float32

        K, N = w.shape
        assert K % group_size == 0,  f"K={K} is not divisible by group_size={group_size}"
        assert N % 8 == 0,           f"N={N} is not a multiple of 8"

        qweight, scales, qzeros = _per_group_minmax_quantize(w, group_size)

        layer = cls(K, N, group_size)
        layer.qweight.copy_(qweight)
        layer.scales.copy_(scales)
        layer.qzeros.copy_(qzeros)
        return layer

    @classmethod
    def from_gptq(
        cls,
        qweight: torch.Tensor,   # [K//8, N] int32, packed along K (consistent with this class)
        qzeros: torch.Tensor,    # [G, N//8] int32, packed along N
        scales: torch.Tensor,    # [G, N]
        group_size: int = 128,
        zero_plus_one: bool = True,
    ) -> "LinearW4A16":
        """Construct a LinearW4A16 from pre-quantized GPTQ buffers.

        GPTQ (checkpoint_format=gptq, v1) zero-point convention:
        Dequantization: (q - (z_stored + 1)) * scale, i.e. z_stored is 1 less than the true zero.
        This class kernel uses (q - z) * scale, so at load time unpacked zeros must be incremented by 1 and re-packed.

        Args:
            qweight/qzeros/scales: Tensors from the GPTQ checkpoint (packing format matches this class)
            zero_plus_one: True -> apply the GPTQ +1 zero correction (required for v1 format)
        """
        from mini_qwen.quantization.packing import pack_int4, unpack_int4

        K = qweight.shape[0] * 8
        N = qweight.shape[1]
        layer = cls(K, N, group_size)
        layer.qweight.copy_(qweight.to(torch.int32))
        layer.scales.copy_(scales.to(torch.bfloat16))
        if zero_plus_one:
            z = unpack_int4(qzeros.to(torch.int32)) + 1   # [G, N], effective zero
            z = z.clamp(0, 15)
            layer.qzeros.copy_(pack_int4(z))
        else:
            layer.qzeros.copy_(qzeros.to(torch.int32))
        return layer

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group_size={self.group_size}")
