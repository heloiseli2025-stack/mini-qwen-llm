import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from mini_qwen.model.layers.rms_norm import RMSNorm
from mini_qwen.model.layers.rope import apply_rotary_emb, rotate_half


class Qwen3Attention(nn.Module):
    """Qwen3 GQA + QK-Norm attention.

    Difference from Llama: Q/K each pass through a per-head RMSNorm before RoPE (QK-Norm).
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, S, hidden_size]
        cos: torch.Tensor,              # [S, head_dim]
        sin: torch.Tensor,              # [S, head_dim]
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

    def _is_quantized(self) -> bool:
        from mini_qwen.model.layers.linear_w4a16 import LinearW4A16
        return isinstance(self.q_proj, LinearW4A16)

    def paged_forward(
        self,
        hidden_states: torch.Tensor,    # [B, S, hidden_size]
        cos: torch.Tensor,              # prefill: [S, D]; decode: [B, D] per-seq
        sin: torch.Tensor,
        k_cache: torch.Tensor,          # [num_blocks, block_size, H_kv, head_dim]
        v_cache: torch.Tensor,
        block_table: torch.Tensor,      # [B, max_blocks_per_seq] int32
        seq_info: torch.Tensor,         # prefill: cu_seqlens [B+1]; decode: seq_lens [B]
        max_seqlen: int,
        mode: str = "prefill",
    ) -> torch.Tensor:
        """Paged attention forward pass.

        BF16 model: fused_qkv_rope handles GEMM + QK-Norm + RoPE in one kernel.
        W4A16 model: W4A16 GEMM + unfused QK-Norm + RoPE (fused kernel requires fp16 weights).

        prefill: hidden_states [B, S, H], calls paged_attn_prefill
        decode:  hidden_states [B, 1, H], calls paged_attn_decode
        """
        from mini_qwen.kernels.paged_attn_prefill import paged_attn_prefill, write_kv_decode
        from mini_qwen.kernels.paged_attn_decode import paged_attn_decode

        B, S, _ = hidden_states.shape
        quantized = self._is_quantized()

        if mode == "prefill":
            if not quantized:
                # BF16: fused QKV GEMM + QK-Norm + RoPE (single Triton kernel)
                from mini_qwen.kernels.fused_qkv_rope import fused_qkv_rope
                q, k, v = fused_qkv_rope(
                    hidden_states,
                    self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                    self.q_norm.weight, self.k_norm.weight,
                    cos, sin,
                )
            else:
                # W4A16: separate quantized GEMMs + unfused QK-Norm + RoPE
                q = self.q_proj(hidden_states).view(B, S, self.num_heads,    self.head_dim)
                k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
                v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
                q = self.q_norm(q)
                k = self.k_norm(k)
                q, k = apply_rotary_emb(q, k, cos, sin)

            total = B * S
            q_packed = q.reshape(total, self.num_heads,    self.head_dim).contiguous()
            k_packed = k.reshape(total, self.num_kv_heads, self.head_dim).contiguous()
            v_packed = v.reshape(total, self.num_kv_heads, self.head_dim).contiguous()
            out = paged_attn_prefill(
                q_packed, k_packed, v_packed,
                k_cache, v_cache, block_table, seq_info, max_seqlen,
            )
            out = out.view(B, S, self.num_heads * self.head_dim)

        else:  # decode
            write_positions = (seq_info - 1).to(torch.int32)

            if not quantized:
                # BF16 decode: fused kernel with per-sequence positions.
                # cos/sin: [B, D] pre-indexed per sequence; positions=arange(B) selects row b.
                from mini_qwen.kernels.fused_qkv_rope import fused_qkv_rope
                positions_fused = torch.arange(B, device=hidden_states.device, dtype=torch.int32)
                q, k, v = fused_qkv_rope(
                    hidden_states,
                    self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                    self.q_norm.weight, self.k_norm.weight,
                    cos, sin,
                    positions=positions_fused,
                )
            else:
                # W4A16 decode: separate quantized GEMMs + unfused QK-Norm + RoPE
                # cos/sin: [B, D] broadcast over head dim
                raw_q = self.q_proj(hidden_states).view(B, 1, self.num_heads,    self.head_dim)
                raw_k = self.k_proj(hidden_states).view(B, 1, self.num_kv_heads, self.head_dim)
                raw_v = self.v_proj(hidden_states).view(B, 1, self.num_kv_heads, self.head_dim)
                raw_q = self.q_norm(raw_q)
                raw_k = self.k_norm(raw_k)
                cos_b = cos.unsqueeze(1).unsqueeze(1)
                sin_b = sin.unsqueeze(1).unsqueeze(1)
                q = raw_q * cos_b + rotate_half(raw_q) * sin_b
                k = raw_k * cos_b + rotate_half(raw_k) * sin_b
                v = raw_v

            write_kv_decode(
                k.squeeze(1).contiguous(), v.squeeze(1).contiguous(),
                k_cache, v_cache, block_table, write_positions,
            )
            out = paged_attn_decode(
                q.squeeze(1).contiguous(), k_cache, v_cache, block_table, seq_info,
            )
            out = out.view(B, 1, self.num_heads * self.head_dim)

        # cast to the input dtype expected by o_proj:
        # nn.Linear stores weight; LinearW4A16 uses bf16 activations
        o_dtype = self.o_proj.weight.dtype if isinstance(self.o_proj, nn.Linear) else torch.bfloat16
        return self.o_proj(out.to(o_dtype))
