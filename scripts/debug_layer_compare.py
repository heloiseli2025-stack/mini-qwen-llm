#!/usr/bin/env python3
"""
逐层比较 mini-qwen MoE 与 HF Qwen3MoE 的中间值，定位 forward 分叉点。

用法（云端 GPU 服务器，CPU 模式）：
  python scripts/debug_layer_compare.py --model /root/autodl-tmp/Qwen3-30B-A3B

两步运行：先加载 HF 收集逐层激活，删除 HF 释放内存，再加载 ours 收集，最后比较。
峰值内存 ~58GB（两次各自独立），不超过 120GB cgroup 限制。
"""
import sys, os, argparse, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F


# ─── CPU fallback for torch._grouped_mm ───────────────────────────────────────
def _install_grouped_mm_cpu_fallback():
    """
    torch._grouped_mm 是 CUDA-only op，CPU forward 需要 fallback。
    覆盖为 Python for-loop 实现，保证数值正确性。
    """
    def _fallback(A, B, offs=None, output_dtype=None):
        if offs is None:
            out = torch.bmm(A.float(), B.float().transpose(-1, -2))
            return out.to(output_dtype) if output_dtype else out

        # HF 传入的 weight 已 transpose(-2,-1)，B 是 [E, K, N]（K=Hin, N=Dout）。
        # 真实 torch._grouped_mm 计算 input[s:t] @ weight[e]，不再转置。
        T, K_in = A.shape
        E = B.shape[0]
        N = B.shape[2]

        out = torch.zeros(T, N, dtype=output_dtype or A.dtype, device=A.device)

        if len(offs) == E + 1:   # [E+1] prefix-sum format
            starts, ends = offs[:-1], offs[1:]
        else:                    # [E] end-index format
            starts = torch.cat([offs.new_zeros(1), offs[:-1]])
            ends = offs

        for e in range(E):
            s, t = starts[e].item(), ends[e].item()
            if s >= t:
                continue
            seg = A[s:t].float() @ B[e].float()   # [M,K] @ [K,N] = [M,N]
            out[s:t] = seg.to(output_dtype) if output_dtype else seg.to(A.dtype)
        return out

    torch._grouped_mm = _fallback
    print("[patch] torch._grouped_mm → CPU for-loop fallback")


_install_grouped_mm_cpu_fallback()


# ─── hook utilities ────────────────────────────────────────────────────────────
def collect_layer_hooks(model_layers, store: dict, prefix: str):
    handles = []
    for i, layer in enumerate(model_layers):
        def make_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                store[f"{prefix}_layer_{idx}"] = out.detach().cpu().clone()
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ─── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/Qwen3-30B-A3B")
    parser.add_argument("--prompt", default="Hello")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from mini_qwen.model.loader import load_moe_from_hf

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    print(f"Input ids: {input_ids}  tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")

    hf_acts  = {}
    our_acts = {}

    # ── Step 1: HF ─────────────────────────────────────────────────────────────
    print("\n=== 加载 HF 模型（CPU, BF16）===")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    hf_model.eval()

    # embed hook
    def hf_embed_hook(module, inp, out):
        hf_acts["embed"] = out.detach().cpu().clone()
    hf_model.model.embed_tokens.register_forward_hook(hf_embed_hook)

    # layer hooks
    hf_handles = collect_layer_hooks(hf_model.model.layers, hf_acts, "hf")

    # final norm hook
    def hf_norm_hook(module, inp, out):
        hf_acts["final_norm"] = out.detach().cpu().clone()
    hf_model.model.norm.register_forward_hook(hf_norm_hook)

    print("运行 HF forward...")
    with torch.no_grad():
        hf_out = hf_model(input_ids)
    hf_logits = hf_out.logits.detach().cpu().clone()

    remove_hooks(hf_handles)
    del hf_model, hf_out
    gc.collect()
    print(f"HF forward 完成，采集 {len(hf_acts)} 个激活")

    # ── Step 2: Our model ──────────────────────────────────────────────────────
    print("\n=== 加载 Our 模型（CPU, BF16, 无量化）===")
    our_model = load_moe_from_hf(args.model, dtype=torch.bfloat16, quantize_w4a16=False)
    our_model.eval()

    def our_embed_hook(module, inp, out):
        our_acts["embed"] = out.detach().cpu().clone()
    our_model.model.embed_tokens.register_forward_hook(our_embed_hook)

    our_handles = collect_layer_hooks(our_model.model.layers, our_acts, "our")

    def our_norm_hook(module, inp, out):
        our_acts["final_norm"] = out.detach().cpu().clone()
    our_model.model.norm.register_forward_hook(our_norm_hook)

    print("运行 Our forward...")
    with torch.no_grad():
        our_logits = our_model(input_ids)
    our_logits = our_logits.detach().cpu().clone()

    remove_hooks(our_handles)
    del our_model
    gc.collect()
    print(f"Our forward 完成，采集 {len(our_acts)} 个激活")

    # ── Step 3: 逐层比较 ────────────────────────────────────────────────────────
    print("\n=== 逐层比较结果 ===")
    num_layers = len([k for k in hf_acts if k.startswith("hf_layer_")])

    def cmp(a, b, name):
        a, b = a.float(), b.float()
        diff = (a - b).abs()
        print(f"  {name:40s}  max={diff.max():.6f}  mean={diff.mean():.8f}"
              f"  a[:4]={a.flatten()[:4].tolist()}")

    cmp(hf_acts["embed"], our_acts["embed"], "embed")

    first_bad = None
    for i in range(num_layers):
        hf_v  = hf_acts.get(f"hf_layer_{i}")
        our_v = our_acts.get(f"our_layer_{i}")
        if hf_v is None or our_v is None:
            print(f"  layer_{i}: MISSING")
            continue
        diff_max = (hf_v.float() - our_v.float()).abs().max().item()
        marker = "  ← 首次超出阈值" if (first_bad is None and diff_max > 0.1) else ""
        print(f"  layer_{i:3d}: max_err={diff_max:.6f}{marker}")
        if first_bad is None and diff_max > 0.1:
            first_bad = i

    print()
    cmp(hf_acts["final_norm"], our_acts["final_norm"], "final_norm")

    # 最终 logits 比较
    logit_diff = (hf_logits.float() - our_logits.float()).abs()
    print(f"\n  {'logits':40s}  max={logit_diff.max():.6f}  mean={logit_diff.mean():.8f}")

    # HF / Our top-5 tokens
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    hf_top5  = hf_logits[0, -1].topk(5)
    our_top5 = our_logits[0, -1].topk(5)
    print(f"\nHF  top-5: {tok.convert_ids_to_tokens(hf_top5.indices.tolist())}")
    print(f"Our top-5: {tok.convert_ids_to_tokens(our_top5.indices.tolist())}")

    # 第一个出问题层的详细分析
    if first_bad is not None:
        print(f"\n=== 第 {first_bad} 层详细分析 ===")
        hf_v  = hf_acts[f"hf_layer_{first_bad}"].float()
        our_v = our_acts[f"our_layer_{first_bad}"].float()
        diff = (hf_v - our_v).abs()
        print(f"  shape: {hf_v.shape}")
        print(f"  HF [:8] = {hf_v.flatten()[:8].tolist()}")
        print(f"  Our[:8] = {our_v.flatten()[:8].tolist()}")
        print(f"  max at idx {diff.argmax().item()}: hf={hf_v.flatten()[diff.argmax()].item():.4f}"
              f"  our={our_v.flatten()[diff.argmax()].item():.4f}")

        # 看前一层是否正常
        if first_bad > 0:
            hf_prev  = hf_acts[f"hf_layer_{first_bad-1}"].float()
            our_prev = our_acts[f"our_layer_{first_bad-1}"].float()
            prev_diff = (hf_prev - our_prev).abs()
            print(f"  前一层 {first_bad-1} max_err={prev_diff.max():.6f}  (应 < 0.1)")

    print("\n=== 分析结束 ===")


if __name__ == "__main__":
    main()
