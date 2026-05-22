#!/usr/bin/env python3
"""
流式逐层比较：每次只加载一层权重，找到第一个 HF vs Our 出现差异的层。
峰值内存 ~3GB，几分钟内完成（不需要加载完整 58GB 模型）。

用法：
  PYTHONPATH=/root/autodl-tmp/mini-qwen-llm \
    python scripts/debug_stream_layers.py --model /root/autodl-tmp/Qwen3-30B-A3B --prompt "Hello, how are you?"
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F


# ─── RMSNorm：两种风格都试 ────────────────────────────────────────────────────
def rms_norm_hf(x, w, eps=1e-6):
    """HF 风格：float32 norm，cast 后乘 weight（BF16 × BF16）。"""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * x32.to(x.dtype)

def rms_norm_ours(x, w, eps=1e-6):
    """Our 风格：全程 float32，最后 cast。"""
    x32 = x.float()
    r   = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * r * w.float()).to(x.dtype)


# ─── RoPE ────────────────────────────────────────────────────────────────────
def build_rope_cache(seq_len, head_dim, rope_theta, dtype):
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t    = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb   = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)   # [S, head_dim]

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def apply_rope(q, k, cos, sin):
    """q,k: [B, S, H, D]; cos,sin: [S, D]"""
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


# ─── safetensors 加载工具 ──────────────────────────────────────────────────────
def load_tensors(index, model_path, key_filter):
    """加载满足 key_filter 的张量，返回 dict。"""
    from safetensors import safe_open
    needed_keys  = {k for k in index if key_filter(k)}
    needed_files = {index[k] for k in needed_keys}
    sd = {}
    for fname in sorted(needed_files):
        fpath = os.path.join(model_path, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in needed_keys:
                    sd[key] = f.get_tensor(key)
    return sd


# ─── 单层 forward（HF 风格 + Our 风格同时运行）───────────────────────────────
def forward_layer(sd_layer, x, cos, sin, cfg, layer_idx, norm_fn):
    """用 sd_layer（单层权重）计算 Qwen3MoE layer forward。
    HF 格式：experts.gate_up_proj / experts.down_proj（批量）。
    Our 格式：experts.{e}.gate_proj / up_proj / down_proj（拆分）。
    norm_fn 控制 RMSNorm 风格。
    """
    p    = f"model.layers.{layer_idx}"
    eps  = cfg["rms_norm_eps"]
    B, S, H = x.shape
    nh   = cfg["num_attention_heads"]
    nkv  = cfg["num_key_value_heads"]
    hd   = cfg["head_dim"]
    ngrp = nh // nkv
    E    = cfg["num_experts"]
    K    = cfg["num_experts_per_tok"]
    D    = cfg["moe_intermediate_size"]

    # ── Attention ──
    residual = x
    nx = norm_fn(x, sd_layer[f"{p}.input_layernorm.weight"], eps)

    q = F.linear(nx, sd_layer[f"{p}.self_attn.q_proj.weight"]).view(B, S, nh,  hd)
    k = F.linear(nx, sd_layer[f"{p}.self_attn.k_proj.weight"]).view(B, S, nkv, hd)
    v = F.linear(nx, sd_layer[f"{p}.self_attn.v_proj.weight"]).view(B, S, nkv, hd)

    q = norm_fn(q, sd_layer[f"{p}.self_attn.q_norm.weight"], eps)
    k = norm_fn(k, sd_layer[f"{p}.self_attn.k_norm.weight"], eps)

    q, k = apply_rope(q, k, cos, sin)

    k = k.repeat_interleave(ngrp, dim=2)
    v = v.repeat_interleave(ngrp, dim=2)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ao = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ao = ao.transpose(1, 2).contiguous().view(B, S, -1)
    ao = F.linear(ao, sd_layer[f"{p}.self_attn.o_proj.weight"])
    x  = residual + ao

    # ── MoE ──
    residual = x
    nx2  = norm_fn(x, sd_layer[f"{p}.post_attention_layernorm.weight"], eps)
    x2d  = nx2.reshape(-1, H)
    T    = x2d.shape[0]

    # Router（fp32）
    logits  = F.linear(x2d.float(), sd_layer[f"{p}.mlp.gate.weight"].float())
    scores  = F.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(scores, K, dim=-1)
    topk_w  = topk_w / topk_w.sum(dim=-1, keepdim=True)

    # Expert（HF 格式）
    gu_key  = f"{p}.mlp.experts.gate_up_proj"
    dw_key  = f"{p}.mlp.experts.down_proj"
    use_hf_fmt = gu_key in sd_layer

    moe_out = torch.zeros(T, H, dtype=x.dtype)
    if use_hf_fmt:
        gu = sd_layer[gu_key]   # [E, 2*D, H]
        dw = sd_layer[dw_key]   # [E, H, D]  -- wait, HF is [E, H, D]? Let me check
        # Actually HF stores down_proj as [E, H, D] where H=hidden, D=inter
        # F.linear(x, dw[e]) = x @ dw[e].T = [1,D] @ [D,H] = [1,H]  ✓
    else:
        # 已拆分的 expert 权重（ours 格式）
        gu, dw = None, None

    for t in range(T):
        acc = torch.zeros(H, dtype=torch.float32)
        for k_idx in range(K):
            e = topk_ids[t, k_idx].item()
            w = topk_w[t, k_idx].item()
            xe = x2d[t:t+1]
            if use_hf_fmt:
                gu_out = F.linear(xe, gu[e])               # [1, 2D]
                g, u   = gu_out.chunk(2, dim=-1)
                ye     = F.linear(F.silu(g) * u, dw[e])    # [1, H]
            else:
                ep     = f"{p}.mlp.experts.{e}"
                g  = F.silu(F.linear(xe, sd_layer[f"{ep}.gate_proj.weight"]))
                u  = F.linear(xe, sd_layer[f"{ep}.up_proj.weight"])
                ye = F.linear(g * u, sd_layer[f"{ep}.down_proj.weight"])
            acc += w * ye[0].float()
        moe_out[t] = acc.to(x.dtype)

    x = residual + moe_out.view_as(residual)
    return x


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="/root/autodl-tmp/Qwen3-30B-A3B")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max-layers", type=int, default=48)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoConfig
    from mini_qwen.config import Qwen3MoEConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    print(f"Prompt: '{args.prompt}'")
    print(f"Tokens: {input_ids.tolist()}")

    hf_config  = AutoConfig.from_pretrained(args.model)
    our_config = Qwen3MoEConfig.from_hf_config(hf_config)
    cfg = {
        "rms_norm_eps":        our_config.rms_norm_eps,
        "rope_theta":          our_config.rope_theta,
        "num_attention_heads": our_config.num_attention_heads,
        "num_key_value_heads": our_config.num_key_value_heads,
        "head_dim":            our_config.head_dim,
        "num_experts":         our_config.num_experts,
        "num_experts_per_tok": our_config.num_experts_per_tok,
        "moe_intermediate_size": our_config.intermediate_size,
    }
    print(f"Config: {cfg}")

    # 加载 safetensors index
    index_path = os.path.join(args.model, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # ── Embed ──────────────────────────────────────────────────────────────────
    print("\n加载 embed weights...", flush=True)
    sd_embed = load_tensors(weight_map, args.model,
                            lambda k: k == "model.embed_tokens.weight")
    x = sd_embed["model.embed_tokens.weight"][input_ids].to(torch.bfloat16)
    del sd_embed
    print(f"embed[:5] = {x[0, 0, :5].tolist()}")

    # RoPE cache
    S   = input_ids.shape[1]
    cos, sin = build_rope_cache(S, cfg["head_dim"], cfg["rope_theta"], torch.bfloat16)

    # ── 逐层 forward ───────────────────────────────────────────────────────────
    x_hf  = x.clone()
    x_our = x.clone()

    for layer_idx in range(min(args.max_layers, our_config.num_hidden_layers)):
        print(f"\n--- 加载 layer {layer_idx} ---", flush=True)
        # 加载该层权重
        sd_layer = load_tensors(
            weight_map, args.model,
            lambda k, i=layer_idx: k.startswith(f"model.layers.{i}.")
        )
        print(f"  张量数: {len(sd_layer)}", flush=True)

        # HF 风格 forward
        x_hf_new  = forward_layer(sd_layer, x_hf,  cos, sin, cfg, layer_idx, rms_norm_hf)
        # Our 风格 forward
        x_our_new = forward_layer(sd_layer, x_our, cos, sin, cfg, layer_idx, rms_norm_ours)

        del sd_layer

        diff_hf_our = (x_hf_new.float() - x_our_new.float()).abs()
        print(f"  HF  out[:5] = {x_hf_new[0, 0, :5].tolist()}")
        print(f"  Our out[:5] = {x_our_new[0, 0, :5].tolist()}")
        print(f"  diff(HF vs Our): max={diff_hf_our.max():.5f}  mean={diff_hf_our.mean():.7f}")

        if diff_hf_our.max() > 0.1:
            print(f"\n!!! Layer {layer_idx}: HF vs Our 差异超出阈值 !!!")
            print(f"  差异集中在 idx={diff_hf_our.argmax().item()}")
            break

        x_hf  = x_hf_new
        x_our = x_our_new

    print("\n=== 完成 ===")
    print(f"两模型同步点（最后一致层）之后出现差异")
    print(f"HF  最终[:5] = {x_hf[0, 0, :5].tolist()}")
    print(f"Our 最终[:5] = {x_our[0, 0, :5].tolist()}")


if __name__ == "__main__":
    main()
