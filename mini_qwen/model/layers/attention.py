import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from mini_qwen.model.layers.rms_norm import RMSNorm
from mini_qwen.model.layers.rope import apply_rotary_emb, rotate_half


class Qwen3Attention(nn.Module):
    """Qwen3 GQA + QK-Norm 注意力。

    与 Llama 的区别：Q/K 在 RoPE 之前各过一遍 per-head RMSNorm（QK-Norm）。
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        # Qwen3 无 QKV bias
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)

        # Qwen3 独有：per-head RMSNorm，作用于 head_dim 维度
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

        # QK-Norm（per-head，沿 head_dim 归一化）
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        # GQA：每个 KV head 服务 num_kv_groups 个 Q head
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # SDPA 期望 [B, heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # [B, heads, S, head_dim] -> [B, S, heads * head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

    def paged_forward(
        self,
        hidden_states: torch.Tensor,    # [B, S, hidden_size]
        cos: torch.Tensor,              # [S, head_dim]
        sin: torch.Tensor,              # [S, head_dim]
        k_cache: torch.Tensor,          # [num_blocks, block_size, H_kv, head_dim]
        v_cache: torch.Tensor,
        block_table: torch.Tensor,      # [B, max_blocks_per_seq] int32
        seq_info: torch.Tensor,         # prefill: cu_seqlens [B+1]; decode: seq_lens [B]
        max_seqlen: int,
        mode: str = "prefill",          # "prefill" 或 "decode"
    ) -> torch.Tensor:
        """使用 fused_qkv_rope + paged attention 的推理路径。

        prefill：hidden_states shape [B, S, H]，调用 paged_attn_prefill
        decode ：hidden_states shape [B, 1, H]，调用 paged_attn_decode
        """
        from mini_qwen.kernels.fused_qkv_rope import fused_qkv_rope
        from mini_qwen.kernels.paged_attn_prefill import paged_attn_prefill, write_kv_decode
        from mini_qwen.kernels.paged_attn_decode import paged_attn_decode

        B, S, _ = hidden_states.shape

        if mode == "prefill":
            # ① QKV 投影 + QK-Norm + RoPE（fused，prefill single-seq 正确）
            q, k, v = fused_qkv_rope(
                hidden_states,
                self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                self.q_norm.weight, self.k_norm.weight,
                cos, sin,
            )
            # q: [B, S, H_q, D]  k/v: [B, S, H_kv, D]
            # paged_attn_prefill 期望 packed [total_tokens, heads, D]
            total = B * S
            q_packed = q.reshape(total, self.num_heads,    self.head_dim)
            k_packed = k.reshape(total, self.num_kv_heads, self.head_dim)
            v_packed = v.reshape(total, self.num_kv_heads, self.head_dim)
            out = paged_attn_prefill(
                q_packed, k_packed, v_packed,
                k_cache, v_cache, block_table, seq_info, max_seqlen,
            )                                          # [total, H_q, D]
            out = out.view(B, S, self.num_heads * self.head_dim)
        else:
            # decode：每条序列位置不同，不能用 fused_qkv_rope
            # cos/sin: [B, D]（由 ForCausalLM 按各序列位置 index 后传入）
            raw_q = self.q_proj(hidden_states).view(B, 1, self.num_heads,    self.head_dim)
            raw_k = self.k_proj(hidden_states).view(B, 1, self.num_kv_heads, self.head_dim)
            raw_v = self.v_proj(hidden_states).view(B, 1, self.num_kv_heads, self.head_dim)
            raw_q = self.q_norm(raw_q)
            raw_k = self.k_norm(raw_k)
            # cos/sin [B, D] → [B, 1, 1, D] 广播到 head 维
            cos_b = cos.unsqueeze(1).unsqueeze(1)
            sin_b = sin.unsqueeze(1).unsqueeze(1)
            q = raw_q * cos_b + rotate_half(raw_q) * sin_b   # [B, 1, H_q,  D]
            k = raw_k * cos_b + rotate_half(raw_k) * sin_b   # [B, 1, H_kv, D]
            v = raw_v                                          # [B, 1, H_kv, D]
            # seq_info = seq_lens_new [B]，新 token 位置 = seq_lens_new - 1
            positions = (seq_info - 1).to(torch.int32)
            write_kv_decode(
                k.squeeze(1).contiguous(), v.squeeze(1).contiguous(),
                k_cache, v_cache, block_table, positions,
            )
            out = paged_attn_decode(
                q.squeeze(1).contiguous(), k_cache, v_cache, block_table, seq_info,
            )                                          # [B, H_q, D]
            out = out.view(B, 1, self.num_heads * self.head_dim)

        return self.o_proj(out.to(self.o_proj.weight.dtype))
