"""Qwen3 MoE FFN block (M4 implementation).

Qwen3 MoE characteristics: 128 experts, top-8 routing, no shared experts.
Module names exactly match HF Qwen3MoEForCausalLM for direct load_state_dict compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_qwen.kernels.moe_router import moe_router
from mini_qwen.kernels.moe_permute import moe_permute
from mini_qwen.kernels.moe_unpermute import moe_unpermute


class _ExpertMLP(nn.Module):
    """SwiGLU MLP for a single expert. Names match HF experts[e].gate_proj / up_proj / down_proj."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoEBlock(nn.Module):
    """Qwen3 MoE FFN block.

    Forward flow:
      1. moe_router -> topk_ids [T, K], topk_weights [T, K]
      2. moe_permute -> permuted_hidden [T*K, H], expert_offsets [E+1]
      3. per-expert SwiGLU (supports nn.Linear BF16 and LinearW4A16)
      4. moe_unpermute -> weighted reduction [T, H]
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts         = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob      = getattr(config, "norm_topk_prob", True)

        # router: name matches HF mlp.gate.weight
        self.gate    = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # experts: names match HF mlp.experts.{e}.*
        self.experts = nn.ModuleList([_ExpertMLP(config) for _ in range(config.num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, x.shape[-1])          # [T, H]，T = B*S
        T, H = x2d.shape

        topk_ids, topk_weights = moe_router(x2d, self.gate.weight, self.num_experts_per_tok)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        permuted, expert_offsets = moe_permute(x2d, topk_ids, self.num_experts)

        out_perm = torch.zeros_like(permuted)
        for e in range(self.num_experts):
            s = expert_offsets[e].item()
            t = expert_offsets[e + 1].item()
            if s == t:
                continue
            out_perm[s:t] = self.experts[e](permuted[s:t])

        out = moe_unpermute(out_perm, topk_weights, topk_ids, T)
        return out.reshape(orig_shape)

    def quantize_to_w4a16(self, group_size: int = 128) -> None:
        """Replace all expert Linears with LinearW4A16 in-place to reduce GPU memory usage."""
        from mini_qwen.model.layers.linear_w4a16 import LinearW4A16
        # LinearW4A16.from_float creates new modules on CPU by default; must explicitly move back to original device
        device = next(self.parameters()).device
        for expert in self.experts:
            expert.gate_proj = LinearW4A16.from_float(expert.gate_proj, group_size).to(device)
            expert.up_proj   = LinearW4A16.from_float(expert.up_proj,   group_size).to(device)
            expert.down_proj = LinearW4A16.from_float(expert.down_proj, group_size).to(device)
