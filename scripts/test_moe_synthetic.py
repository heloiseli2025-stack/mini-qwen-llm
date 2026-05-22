#!/usr/bin/env python3
"""
MoE kernel の合成テスト（モデル不要、数秒で終わる）。

permute + expert loop + unpermute が naive for-loop oracle と一致するか検証。
失敗 → kernel に bug。合格 → bug は attention か model 構造にある。

用法:
  PYTHONPATH=/root/autodl-tmp/mini-qwen-llm python scripts/test_moe_synthetic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from mini_qwen.kernels.moe_router    import moe_router
from mini_qwen.kernels.moe_permute   import moe_permute
from mini_qwen.kernels.moe_unpermute import moe_unpermute


def oracle_moe_forward(x, gate_w, expert_weights, top_k, norm_topk_prob):
    """Naive for-loop oracle（正解計算，ベクトル化なし）。"""
    T, H = x.shape
    E = gate_w.shape[0]
    D = expert_weights['gate'][0].shape[0]

    # Router（fp32）
    logits = F.linear(x.float(), gate_w.float())
    scores = F.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(scores, top_k, dim=-1)
    if norm_topk_prob:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    # 用 float32 累加，与 moe_unpermute 的 (topk_w.unsqueeze(-1) * expert_h).sum() 保持一致
    out_fp32 = torch.zeros(T, H, dtype=torch.float32)
    for t in range(T):
        for k in range(top_k):
            e = topk_ids[t, k].item()
            w = topk_w[t, k].float().item()
            x_e = x[t:t+1]   # [1, H]
            gate_e = F.silu(F.linear(x_e, expert_weights['gate'][e]))
            up_e   = F.linear(x_e, expert_weights['up'][e])
            y_e    = F.linear(gate_e * up_e, expert_weights['down'][e])
            out_fp32[t] += w * y_e[0].float()
    return out_fp32.to(x.dtype), topk_ids, topk_w


def our_moe_forward(x, gate_w, expert_weights, top_k, norm_topk_prob, E):
    """我们的实现（使用 permute/unpermute kernel）。"""
    topk_ids, topk_w = moe_router(x, gate_w, top_k)
    if norm_topk_prob:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    permuted, offsets = moe_permute(x, topk_ids, E)

    T, H = x.shape
    D = expert_weights['gate'][0].shape[0]
    out_perm = torch.zeros_like(permuted)

    for e in range(E):
        s = offsets[e].item()
        t = offsets[e+1].item()
        if s == t:
            continue
        x_e = permuted[s:t]
        gate_e = F.silu(F.linear(x_e, expert_weights['gate'][e]))
        up_e   = F.linear(x_e, expert_weights['up'][e])
        y_e    = F.linear(gate_e * up_e, expert_weights['down'][e])
        out_perm[s:t] = y_e

    out = moe_unpermute(out_perm, topk_w, topk_ids, T)
    return out, topk_ids, topk_w


def run_test(T, E, K, H, D, seed, dtype, norm_topk_prob, label):
    torch.manual_seed(seed)
    device = "cpu"

    x        = torch.randn(T, H, dtype=dtype, device=device)
    gate_w   = torch.randn(E, H, dtype=dtype, device=device)
    gate_ws  = [torch.randn(D, H, dtype=dtype) for _ in range(E)]
    up_ws    = [torch.randn(D, H, dtype=dtype) for _ in range(E)]
    down_ws  = [torch.randn(H, D, dtype=dtype) for _ in range(E)]
    expert_weights = {'gate': gate_ws, 'up': up_ws, 'down': down_ws}

    ref_out,  ref_ids, ref_w  = oracle_moe_forward(x, gate_w, expert_weights, K, norm_topk_prob)
    our_out,  our_ids, our_w  = our_moe_forward(   x, gate_w, expert_weights, K, norm_topk_prob, E)

    # 路由必须完全一致（相同输入，相同 softmax+topk）
    ids_match = (ref_ids == our_ids).all().item()
    w_diff    = (ref_w.float() - our_w.float()).abs().max().item()
    out_diff  = (ref_out.float() - our_out.float()).abs().max().item()

    ok = ids_match and (w_diff < 1e-4) and (out_diff < 1e-3)
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"[{label}] {status}  ids_match={ids_match}  w_diff={w_diff:.2e}  out_diff={out_diff:.2e}")
    if not ok:
        print(f"       ref_out[:4] = {ref_out[0, :4].tolist()}")
        print(f"       our_out[:4] = {our_out[0, :4].tolist()}")
        print(f"       ref_ids[0]  = {ref_ids[0].tolist()}")
        print(f"       our_ids[0]  = {our_ids[0].tolist()}")
    return ok


def main():
    print("=== MoE 合成测试 ===\n")
    all_ok = True

    # 基础：T=1 token（单 token 推理）
    all_ok &= run_test(T=1, E=4, K=2, H=8, D=4, seed=0,
                       dtype=torch.float32, norm_topk_prob=True,
                       label="T=1,E=4,K=2,fp32,norm")

    # 多 token
    all_ok &= run_test(T=4, E=8, K=3, H=16, D=8, seed=1,
                       dtype=torch.float32, norm_topk_prob=True,
                       label="T=4,E=8,K=3,fp32,norm")

    # BF16（Qwen3-30B 实际使用的精度）
    all_ok &= run_test(T=2, E=16, K=4, H=32, D=16, seed=2,
                       dtype=torch.bfloat16, norm_topk_prob=True,
                       label="T=2,E=16,K=4,bf16,norm")

    # norm_topk_prob=False
    all_ok &= run_test(T=2, E=8, K=2, H=16, D=8, seed=3,
                       dtype=torch.float32, norm_topk_prob=False,
                       label="T=2,E=8,K=2,fp32,no-norm")

    # 接近真实规模（小 H/D 但 E=128, K=8）
    all_ok &= run_test(T=1, E=128, K=8, H=64, D=32, seed=42,
                       dtype=torch.bfloat16, norm_topk_prob=True,
                       label="T=1,E=128,K=8,bf16 (真实规模 E/K)")

    print()
    if all_ok:
        print("=== 所有测试通过 → MoE kernel 无 bug，问题在 attention 或 model 结构 ===")
    else:
        print("=== 有测试失败 → MoE kernel 存在 bug，修复后重试 ===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
