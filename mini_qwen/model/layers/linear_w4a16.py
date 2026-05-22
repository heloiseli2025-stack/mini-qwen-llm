"""W4A16 量化 Linear 层（M3 实现）。"""
import torch
import torch.nn as nn

from mini_qwen.kernels.w4a16_gemm import w4a16_gemm


class LinearW4A16(nn.Module):
    """packed int4 权重 + bf16 激活的量化 Linear。

    Args:
        in_features:  输入维度 K
        out_features: 输出维度 N
        group_size:   量化 group size，默认 128
    """

    def __init__(self, in_features: int, out_features: int, group_size: int = 128):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size

        K, N, gs = in_features, out_features, group_size
        G = K // gs

        # packed int4 权重；shape 约定与 w4a16_gemm 签名一致
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
        """将 bf16/fp32 nn.Linear 量化为 W4A16。

        Args:
            linear:     原始 nn.Linear（weight [N, K]）
            group_size: 量化 group size

        Returns:
            LinearW4A16 实例，weight 已替换为 packed int4
        """
        from mini_qwen.quantization.awq import _per_group_minmax_quantize

        w = linear.weight.data.float()    # [N, K] → 转置后 [K, N]
        w = w.T.contiguous()              # [K, N] float32

        K, N = w.shape
        assert K % group_size == 0,  f"K={K} 不能被 group_size={group_size} 整除"
        assert N % 8 == 0,           f"N={N} 不是 8 的倍数"

        qweight, scales, qzeros = _per_group_minmax_quantize(w, group_size)

        layer = cls(K, N, group_size)
        layer.qweight.copy_(qweight)
        layer.scales.copy_(scales)
        layer.qzeros.copy_(qzeros)
        return layer

    @classmethod
    def from_gptq(
        cls,
        qweight: torch.Tensor,   # [K//8, N] int32，沿 K 打包（与本类一致）
        qzeros: torch.Tensor,    # [G, N//8] int32，沿 N 打包
        scales: torch.Tensor,    # [G, N]
        group_size: int = 128,
        zero_plus_one: bool = True,
    ) -> "LinearW4A16":
        """从预量化的 GPTQ buffer 构造 LinearW4A16。

        GPTQ（checkpoint_format=gptq, v1）的 zero-point 约定：
        反量化为 (q - (z_stored + 1)) * scale，即 z_stored 比真实 zero 小 1。
        本类 kernel 用 (q - z) * scale，故加载时需把 unpacked zero +1 再重新打包。

        Args:
            qweight/qzeros/scales: GPTQ checkpoint 里的张量（打包格式与本类一致）
            zero_plus_one: True → 应用 GPTQ 的 +1 zero 修正（v1 格式需要）
        """
        from mini_qwen.quantization.packing import pack_int4, unpack_int4

        K = qweight.shape[0] * 8
        N = qweight.shape[1]
        layer = cls(K, N, group_size)
        layer.qweight.copy_(qweight.to(torch.int32))
        layer.scales.copy_(scales.to(torch.bfloat16))
        if zero_plus_one:
            z = unpack_int4(qzeros.to(torch.int32)) + 1   # [G, N]，effective zero
            z = z.clamp(0, 15)
            layer.qzeros.copy_(pack_int4(z))
        else:
            layer.qzeros.copy_(qzeros.to(torch.int32))
        return layer

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group_size={self.group_size}")
