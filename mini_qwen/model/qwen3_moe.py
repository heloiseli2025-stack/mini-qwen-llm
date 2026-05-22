"""Qwen3 MoE 推理模型（M4 实现）。

Qwen3-30B-A3B：128 expert，top-8 路由，无 shared expert。
模块命名与 HF Qwen3MoEForCausalLM 保持一致，方便 load_state_dict。
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from mini_qwen.config import Qwen3MoEConfig
from mini_qwen.model.layers.attention import Qwen3Attention
from mini_qwen.model.layers.moe import Qwen3MoEBlock
from mini_qwen.model.layers.rms_norm import RMSNorm
from mini_qwen.model.layers.rope import RotaryEmbedding

if TYPE_CHECKING:
    from mini_qwen.cache.kv_cache import KVCache


class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.input_layernorm         = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn               = Qwen3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp                     = Qwen3MoEBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states

    def paged_forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_info: torch.Tensor,
        max_seqlen: int,
        mode: str,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn.paged_forward(
            self.input_layernorm(hidden_states), cos, sin,
            k_cache, v_cache, block_table, seq_info, max_seqlen, mode,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class Qwen3MoEModel(nn.Module):
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoEDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        seq_len = input_ids.shape[1]
        cos, sin = self.rotary_emb(seq_len)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        return self.norm(hidden_states)


class Qwen3MoEForCausalLM(nn.Module):
    def __init__(self, config: Qwen3MoEConfig):
        super().__init__()
        self.config = config
        self.model  = Qwen3MoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """返回 logits，shape [batch, seq_len, vocab_size]。"""
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(generated)[:, -1, :]
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def quantize_experts_to_w4a16(self, group_size: int = 128) -> None:
        """将所有层的 expert 权重量化为 W4A16。E2E 前必须调用（30B BF16 超出 24GB）。"""
        for layer in self.model.layers:
            layer.mlp.quantize_to_w4a16(group_size=group_size)

    def paged_forward_single_prefill(
        self,
        input_ids: torch.Tensor,       # [1, S]
        kv_caches: "list[KVCache]",
        block_table: torch.Tensor,     # [1, max_blocks] int32
    ) -> torch.Tensor:                 # [1, S, vocab_size]
        """单条序列 prefill，写入 KV cache，返回全序列 logits。"""
        S = input_ids.shape[1]
        hidden = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(S)
        cos = cos.to(hidden.dtype)
        sin = sin.to(hidden.dtype)
        cu_seqlens = torch.tensor([0, S], dtype=torch.int32, device=input_ids.device)
        for i, layer in enumerate(self.model.layers):
            hidden = layer.paged_forward(
                hidden, cos, sin,
                kv_caches[i].k_cache, kv_caches[i].v_cache,
                block_table, cu_seqlens, S, "prefill",
            )
        hidden = self.model.norm(hidden)
        return self.lm_head(hidden)

    def paged_forward_decode(
        self,
        input_ids: torch.Tensor,       # [B]
        kv_caches: "list[KVCache]",
        block_table: torch.Tensor,     # [B, max_blocks] int32
        seq_lens_new: torch.Tensor,    # [B] int32，含新 token 的总长
    ) -> torch.Tensor:                 # [B, vocab_size]
        """批量 decode 一步，返回各序列 next-token logits。"""
        B = input_ids.shape[0]
        hidden = self.model.embed_tokens(input_ids.unsqueeze(1))   # [B, 1, H]
        positions = (seq_lens_new - 1).to(torch.long)
        cos = self.model.rotary_emb.cos_cached[positions].to(hidden.dtype)  # [B, D]
        sin = self.model.rotary_emb.sin_cached[positions].to(hidden.dtype)
        for i, layer in enumerate(self.model.layers):
            hidden = layer.paged_forward(
                hidden, cos, sin,
                kv_caches[i].k_cache, kv_caches[i].v_cache,
                block_table, seq_lens_new, 0, "decode",
            )
        hidden = self.model.norm(hidden)
        return self.lm_head(hidden).squeeze(1)   # [B, vocab_size]
