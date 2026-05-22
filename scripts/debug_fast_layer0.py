#!/usr/bin/env python3
"""
快速版：只从 safetensors 加载 layer 0 + embed 权重（~2GB），
手动逐步计算 layer 0 的 forward，同时用「HF 风格」和「Our 风格」计算，
对比每个中间值，定位 bug 所在。

用法：
  PYTHONPATH=/root/autodl-tmp/mini-qwen-llm \
    python scripts/debug_fast_layer0.py --model /root/autodl-tmp/Qwen3-30B-A3B
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F


# ─── 辅助函数（都在 bfloat16，除非注释说明） ──────────────────────────────
def rms_norm_hf(x, w, eps=1e-6):
    """HF 风格：float32 归一化，cast 回 input dtype 后再乘 weight（weight 是 BF16）。"""
    input_dtype = x.dtype
    x_fp32 = x.float()
    v = x_fp32.pow(2).mean(-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(v + eps)
    return w * x_norm.to(input_dtype)          # BF16 × BF16 → BF16


def rms_norm_ours(x, w, eps=1e-6):
    """Our 风格：全程 float32，最后 cast。"""
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp32 * rms * w.float()).to(x.dtype)   # float32 → BF16


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q, k, cos, sin):
    """q,k: [B, S, H, D];  cos,sin: [S, D]"""
    cos = cos.unsqueeze(0).unsqueeze(2)   # [1, S, 1, D]
    sin = sin.unsqueeze(0).unsqueeze(2)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def cmp(a, b, name, tol=0.05):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    flag = " ← DIFF" if d.max() > tol else ""
    print(f"  {name:45s}  max={d.max():.4f}  mean={d.mean():.6f}{flag}")
    return d.max().item()


# ─── 加载 layer 0 权重 ─────────────────────────────────────────────────────
def load_layer0_weights(model_path):
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)["weight_map"]

    # 找出 layer 0 + embed + final_norm 涉及的分片文件
    needed_keys = {k for k in index if
                   k.startswith("model.layers.0.") or
                   k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")}
    needed_files = {index[k] for k in needed_keys}

    print(f"需要加载 {len(needed_files)} 个分片文件（共 {len(set(index.values()))} 个）...")
    sd = {}
    for fname in sorted(needed_files):
        fpath = os.path.join(model_path, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in needed_keys:
                    sd[key] = f.get_tensor(key)
        print(f"  {fname}  已加载 {sum(1 for k in sd if index[k]==fname)} 个张量")

    print(f"共加载 {len(sd)} 个张量")
    return sd


# ─── Layer 0 forward ──────────────────────────────────────────────────────
def forward_layer0(sd, input_ids, config, norm_fn, label):
    """用给定的 norm_fn 计算 layer 0 full forward，打印每步中间值。"""
    p = "model.layers.0"
    eps = config["rms_norm_eps"]
    B, S = input_ids.shape
    num_heads    = config["num_attention_heads"]   # 32
    num_kv_heads = config["num_key_value_heads"]   # 4
    head_dim     = config["head_dim"]              # 128
    num_kv_groups = num_heads // num_kv_heads      # 8
    num_experts  = config["num_experts"]           # 128
    num_experts_per_tok = config["num_experts_per_tok"]  # 8
    moe_int      = config["moe_intermediate_size"] # 768

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Embed
    x = sd["model.embed_tokens.weight"][input_ids].to(torch.bfloat16)  # [B, S, H]
    print(f"  embed[:5] = {x[0, 0, :5].tolist()}")

    # ── Attention ────────────────────────────────────────────────
    residual = x
    norm_x = norm_fn(x, sd[f"{p}.input_layernorm.weight"], eps)
    print(f"  input_norm[:5] = {norm_x[0,0,:5].tolist()}")

    q = F.linear(norm_x, sd[f"{p}.self_attn.q_proj.weight"])  # [B, S, 4096]
    k = F.linear(norm_x, sd[f"{p}.self_attn.k_proj.weight"])  # [B, S, 512]
    v = F.linear(norm_x, sd[f"{p}.self_attn.v_proj.weight"])  # [B, S, 512]

    q = q.view(B, S, num_heads, head_dim)
    k = k.view(B, S, num_kv_heads, head_dim)
    v = v.view(B, S, num_kv_heads, head_dim)

    # QK-Norm
    q = norm_fn(q, sd[f"{p}.self_attn.q_norm.weight"], eps)
    k = norm_fn(k, sd[f"{p}.self_attn.k_norm.weight"], eps)
    print(f"  q_after_qknorm[0,0,0,:5] = {q[0,0,0,:5].tolist()}")

    # RoPE（位置 0 → cos=1, sin=0，旋转是恒等）
    inv_freq = (1.0 / (config["rope_theta"] **
                       (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)))
    freqs = torch.outer(torch.arange(S, dtype=torch.float32), inv_freq)
    emb   = torch.cat([freqs, freqs], dim=-1)
    cos   = emb.cos().to(torch.bfloat16)   # [S, head_dim]
    sin   = emb.sin().to(torch.bfloat16)
    q, k  = apply_rope(q, k, cos, sin)
    print(f"  q_after_rope[0,0,0,:5] = {q[0,0,0,:5].tolist()}")

    # GQA expand
    k = k.repeat_interleave(num_kv_groups, dim=2)
    v = v.repeat_interleave(num_kv_groups, dim=2)

    # SDPA
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
    attn_out = F.linear(attn_out, sd[f"{p}.self_attn.o_proj.weight"])
    print(f"  attn_out[:5] = {attn_out[0,0,:5].tolist()}")

    x = residual + attn_out
    print(f"  after_attn_residual[:5] = {x[0,0,:5].tolist()}")

    # ── MoE ──────────────────────────────────────────────────────
    residual = x
    norm_x2  = norm_fn(x, sd[f"{p}.post_attention_layernorm.weight"], eps)
    print(f"  post_attn_norm[:5] = {norm_x2[0,0,:5].tolist()}")

    # Router（fp32）
    x2d  = norm_x2.reshape(-1, norm_x2.shape[-1])     # [T, H]
    T, H = x2d.shape
    gate_logits = F.linear(x2d.float(), sd[f"{p}.mlp.gate.weight"].float())  # [T, 128]
    gate_scores = F.softmax(gate_logits, dim=-1)
    topk_w, topk_ids = torch.topk(gate_scores, num_experts_per_tok, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)   # norm_topk_prob
    print(f"  selected experts (token 0) = {topk_ids[0].tolist()}")
    print(f"  topk_weights (token 0)     = {topk_w[0].tolist()}")

    # Expert loop（HF 风格：逐 token 逐 expert，native BF16）
    moe_out = torch.zeros(T, H, dtype=torch.bfloat16)
    # 从 HF 格式的 gate_up_proj 读取
    gu = sd[f"{p}.mlp.experts.gate_up_proj"]   # [128, 2*768, 2048]
    dw = sd[f"{p}.mlp.experts.down_proj"]      # [128, 2048, 768]
    for t_idx in range(T):
        for k_idx in range(num_experts_per_tok):
            e   = topk_ids[t_idx, k_idx].item()
            w   = topk_w[t_idx, k_idx].item()
            x_e = x2d[t_idx:t_idx+1]                      # [1, 2048]
            gu_e = gu[e]                                    # [2*768, 2048]
            gu_out = F.linear(x_e, gu_e)                   # [1, 1536]
            gate_e, up_e = gu_out.chunk(2, dim=-1)          # [1,768] each
            y_e  = F.linear(F.silu(gate_e) * up_e, dw[e])  # [1, 2048]
            moe_out[t_idx] += w * y_e[0]
    print(f"  moe_out[:5] = {moe_out[0,:5].tolist()}")

    x = residual + moe_out.reshape_as(residual)
    print(f"  layer0_out[:5] = {x[0,0,:5].tolist()}")
    return x


# ─── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen3-30B-A3B")
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoConfig
    from mini_qwen.config import Qwen3MoEConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    print(f"Input ids: {input_ids}  ({tokenizer.convert_ids_to_tokens(input_ids[0].tolist())})")

    hf_config   = AutoConfig.from_pretrained(args.model)
    our_config  = Qwen3MoEConfig.from_hf_config(hf_config)
    config = {
        "rms_norm_eps":        our_config.rms_norm_eps,
        "rope_theta":          our_config.rope_theta,
        "num_attention_heads": our_config.num_attention_heads,
        "num_key_value_heads": our_config.num_key_value_heads,
        "head_dim":            our_config.head_dim,
        "num_experts":         our_config.num_experts,
        "num_experts_per_tok": our_config.num_experts_per_tok,
        "moe_intermediate_size": our_config.intermediate_size,
    }
    print(f"Config: {config}")

    # 加载 layer 0 权重（~2GB, 几十秒）
    sd = load_layer0_weights(args.model)

    # HF 风格 forward
    out_hf  = forward_layer0(sd, input_ids, config, rms_norm_hf,  "HF  风格（weight 乘法在 BF16）")
    # Our 风格 forward
    out_our = forward_layer0(sd, input_ids, config, rms_norm_ours, "Our 风格（全程 float32，最后 cast）")

    print("\n=== 最终比较 ===")
    cmp(out_hf, out_our, "layer0 output", tol=0.01)

    diff = (out_hf.float() - out_our.float()).abs()
    if diff.max() < 0.01:
        print("-> layer 0 两种风格一致（误差 < 0.01）")
        print("-> bug 可能在 layer 1+ 或 lm_head 的精度积累")
    else:
        print("-> layer 0 已出现差异，RMSNorm 精度是原因")
        print("   但这不足以解释 logit 误差 ~13，需要继续排查")


if __name__ == "__main__":
    main()
