from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_hf_config(cls, hf_config) -> "Qwen3Config":
        # transformers 4.51+ 把 rope_theta 放进了 rope_parameters / rope_scaling 字典
        rope_params = getattr(hf_config, "rope_parameters", None) or getattr(
            hf_config, "rope_scaling", {}
        )
        rope_theta = (
            rope_params.get("rope_theta", 1000000.0)
            if isinstance(rope_params, dict)
            else 1000000.0
        )

        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=getattr(
                hf_config, "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            ),
            intermediate_size=hf_config.intermediate_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", True),
        )


@dataclass
class Qwen3MoEConfig(Qwen3Config):
    """Qwen3 MoE 扩展配置（Qwen3-30B-A3B 等）。"""
    num_experts: int = 128
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True   # Qwen3-30B-A3B 默认 True：topk weights 除以其 sum 归一化

    @classmethod
    def from_hf_config(cls, hf_config) -> "Qwen3MoEConfig":
        base = Qwen3Config.from_hf_config(hf_config)
        # MoE 模型用 moe_intermediate_size（per-expert），而非 intermediate_size（dense FFN）
        moe_intermediate = getattr(hf_config, "moe_intermediate_size", None)
        if moe_intermediate is not None:
            base.intermediate_size = moe_intermediate
        return cls(
            **vars(base),
            num_experts=getattr(hf_config, "num_experts", 128),
            num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 8),
            norm_topk_prob=getattr(hf_config, "norm_topk_prob", True),
        )
