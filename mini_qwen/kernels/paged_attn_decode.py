"""Paged Attention Decode Kernel（M1.1–M1.4 实现）。

FROZEN SIGNATURE（M1 完成后冻结，改动须 owner 批准）：
  paged_attn_decode(
      q:           [batch, num_q_heads, head_dim], bf16
      k_cache:     [num_blocks, block_size, num_kv_heads, head_dim], bf16
      v_cache:     [num_blocks, block_size, num_kv_heads, head_dim], bf16
      block_table: [batch, max_blocks_per_seq], int32
      seq_lens:    [batch], int32
  ) -> [batch, num_q_heads, head_dim], bf16
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _check_inputs(q, k_cache, v_cache, block_table, seq_lens):
    assert q.dim() == 3 and k_cache.dim() == 4, "shape mismatch"
    assert q.is_cuda, "inputs must be on CUDA"
    assert block_table.dtype == torch.int32, "block_table must be int32"


# ══════════════════════════════════════════════════════════════════════════════
# M1.1 — Naive two-pass decode（fp32，无 GQA）
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _paged_decode_v1_kernel(
    Q, K_cache, V_cache,
    Block_table, Seq_lens, Out,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_tb, stride_tp,
    stride_ob, stride_oh, stride_od,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """两遍扫描：第一遍找 max score，第二遍 softmax + 加权 V。"""
    batch_id = tl.program_id(0)
    head_id  = tl.program_id(1)

    seq_len   = tl.load(Seq_lens + batch_id).to(tl.int32)
    num_pages = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale     = 1.0 / tl.math.sqrt(float(HEAD_DIM))

    d_range    = tl.arange(0, HEAD_DIM)
    q          = tl.load(Q + batch_id * stride_qb + head_id * stride_qh + d_range).to(tl.float32)
    kv_head_id = head_id  # M1.1: no GQA

    # Pass 1: find max score
    m_i = float('-inf')
    for page_idx in range(num_pages):
        phys = tl.load(Block_table + batch_id * stride_tb + page_idx * stride_tp)
        for slot in tl.static_range(BLOCK_SIZE):
            token_idx = page_idx * BLOCK_SIZE + slot
            valid     = token_idx < seq_len
            k = tl.load(
                K_cache + phys * stride_kb + slot * stride_ks + kv_head_id * stride_kh + d_range,
                mask=valid, other=0.0,
            ).to(tl.float32)
            score = tl.where(valid, tl.sum(q * k) * scale, float('-inf'))
            m_i   = tl.maximum(m_i, score)

    # Pass 2: softmax weights + weighted V sum
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], tl.float32)
    for page_idx in range(num_pages):
        phys = tl.load(Block_table + batch_id * stride_tb + page_idx * stride_tp)
        for slot in tl.static_range(BLOCK_SIZE):
            token_idx = page_idx * BLOCK_SIZE + slot
            valid     = token_idx < seq_len
            k = tl.load(
                K_cache + phys * stride_kb + slot * stride_ks + kv_head_id * stride_kh + d_range,
                mask=valid, other=0.0,
            ).to(tl.float32)
            score = tl.where(valid, tl.sum(q * k) * scale, float('-inf'))
            p     = tl.where(valid, tl.exp(score - m_i), 0.0)
            l_i  += p
            v = tl.load(
                V_cache + phys * stride_vb + slot * stride_vs + kv_head_id * stride_vh + d_range,
                mask=valid, other=0.0,
            ).to(tl.float32)
            acc = acc + p * v

    acc = acc / tl.maximum(l_i, 1e-9)
    tl.store(Out + batch_id * stride_ob + head_id * stride_oh + d_range, acc)


def paged_attn_decode_v1(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """M1.1 naive two-pass decode（fp32, no GQA）。"""
    _check_inputs(q, k_cache, v_cache, block_table, seq_lens)
    B, H_q, D = q.shape
    block_size = k_cache.shape[1]
    q_fp = q.float().contiguous()
    out  = torch.empty(B, H_q, D, dtype=torch.float32, device=q.device)

    _paged_decode_v1_kernel[(B, H_q)](
        q_fp, k_cache, v_cache,
        block_table, seq_lens, out,
        *q_fp.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_table.stride(),
        *out.stride(),
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# M1.2 — Two-pass + GQA（kv_head = q_head // num_kv_groups）
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _paged_decode_v2_kernel(
    Q, K_cache, V_cache,
    Block_table, Seq_lens, Out,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_tb, stride_tp,
    stride_ob, stride_oh, stride_od,
    NUM_KV_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_id   = tl.program_id(0)
    head_id    = tl.program_id(1)
    kv_head_id = head_id // NUM_KV_GROUPS  # GQA

    seq_len   = tl.load(Seq_lens + batch_id).to(tl.int32)
    num_pages = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale     = 1.0 / tl.math.sqrt(float(HEAD_DIM))

    d_range = tl.arange(0, HEAD_DIM)
    q       = tl.load(Q + batch_id * stride_qb + head_id * stride_qh + d_range).to(tl.float32)

    m_i = float('-inf')
    for page_idx in range(num_pages):
        phys = tl.load(Block_table + batch_id * stride_tb + page_idx * stride_tp)
        for slot in tl.static_range(BLOCK_SIZE):
            token_idx = page_idx * BLOCK_SIZE + slot
            valid     = token_idx < seq_len
            k = tl.load(
                K_cache + phys * stride_kb + slot * stride_ks + kv_head_id * stride_kh + d_range,
                mask=valid, other=0.0,
            ).to(tl.float32)
            score = tl.where(valid, tl.sum(q * k) * scale, float('-inf'))
            m_i   = tl.maximum(m_i, score)

    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], tl.float32)
    for page_idx in range(num_pages):
        phys = tl.load(Block_table + batch_id * stride_tb + page_idx * stride_tp)
        for slot in tl.static_range(BLOCK_SIZE):
            token_idx = page_idx * BLOCK_SIZE + slot
            valid     = token_idx < seq_len
            k = tl.load(
                K_cache + phys * stride_kb + slot * stride_ks + kv_head_id * stride_kh + d_range,
                mask=valid, other=0.0,
            ).to(tl.float32)
            score = tl.where(valid, tl.sum(q * k) * scale, float('-inf'))
            p     = tl.where(valid, tl.exp(score - m_i), 0.0)
            l_i  += p
            v = tl.load(
                V_cache + phys * stride_vb + slot * stride_vs + kv_head_id * stride_vh + d_range,
                mask=valid, other=0.0,
            ).to(tl.float32)
            acc = acc + p * v

    acc = acc / tl.maximum(l_i, 1e-9)
    tl.store(Out + batch_id * stride_ob + head_id * stride_oh + d_range, acc)


def paged_attn_decode_v2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_groups: int,
) -> torch.Tensor:
    """M1.2 two-pass decode with GQA。"""
    _check_inputs(q, k_cache, v_cache, block_table, seq_lens)
    B, H_q, D = q.shape
    block_size = k_cache.shape[1]
    q_fp = q.float().contiguous()
    out  = torch.empty(B, H_q, D, dtype=torch.float32, device=q.device)

    _paged_decode_v2_kernel[(B, H_q)](
        q_fp, k_cache, v_cache,
        block_table, seq_lens, out,
        *q_fp.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_table.stride(),
        *out.stride(),
        NUM_KV_GROUPS=num_kv_groups,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# M1.3 — Online softmax（单遍，FlashAttention v2 风格）
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _paged_decode_v3_kernel(
    Q, K_cache, V_cache,
    Block_table, Seq_lens, Out,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_tb, stride_tp,
    stride_ob, stride_oh, stride_od,
    NUM_KV_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """单遍 online softmax：running max m_i + running sum l_i + acc。"""
    batch_id   = tl.program_id(0)
    head_id    = tl.program_id(1)
    kv_head_id = head_id // NUM_KV_GROUPS

    seq_len   = tl.load(Seq_lens + batch_id).to(tl.int32)
    num_pages = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale     = 1.0 / tl.math.sqrt(float(HEAD_DIM))

    d_range = tl.arange(0, HEAD_DIM)
    q       = tl.load(Q + batch_id * stride_qb + head_id * stride_qh + d_range).to(tl.float32)

    m_i = float('-inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], tl.float32)

    for page_idx in range(num_pages):
        phys = tl.load(Block_table + batch_id * stride_tb + page_idx * stride_tp)
        for slot in tl.static_range(BLOCK_SIZE):
            token_idx = page_idx * BLOCK_SIZE + slot
            valid     = token_idx < seq_len
            k = tl.load(
                K_cache + phys * stride_kb + slot * stride_ks + kv_head_id * stride_kh + d_range,
                mask=valid, other=0.0,
            ).to(tl.float32)
            score = tl.sum(q * k) * scale
            score = tl.where(valid, score, float('-inf'))

            # 安全的 online softmax 更新（valid=False 时 m_i 不变，alpha=1, beta=0）
            m_new = tl.where(valid, tl.maximum(m_i, score), m_i)
            alpha = tl.exp(m_i - m_new)
            beta  = tl.where(valid, tl.exp(score - m_new), 0.0)
            l_i   = alpha * l_i + beta

            v = tl.load(
                V_cache + phys * stride_vb + slot * stride_vs + kv_head_id * stride_vh + d_range,
                mask=valid, other=0.0,
            ).to(tl.float32)
            acc = alpha * acc + beta * v
            m_i = m_new

    acc = acc / tl.maximum(l_i, 1e-9)
    tl.store(Out + batch_id * stride_ob + head_id * stride_oh + d_range, acc)


def paged_attn_decode_v3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_groups: int,
) -> torch.Tensor:
    """M1.3 online softmax decode with GQA（输入可以是 bf16）。"""
    _check_inputs(q, k_cache, v_cache, block_table, seq_lens)
    B, H_q, D = q.shape
    block_size = k_cache.shape[1]
    out = torch.empty(B, H_q, D, dtype=torch.float32, device=q.device)

    _paged_decode_v3_kernel[(B, H_q)](
        q.float().contiguous(), k_cache, v_cache,
        block_table, seq_lens, out,
        *q.float().stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_table.stride(),
        *out.stride(),
        NUM_KV_GROUPS=num_kv_groups,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# M1.4 — 向量化（整页加载）+ autotune
# ══════════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["HEAD_DIM", "NUM_KV_GROUPS"],
)
@triton.jit
def _paged_decode_v4_kernel(
    Q, K_cache, V_cache,
    Block_table, Seq_lens, Out,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_tb, stride_tp,
    stride_ob, stride_oh, stride_od,
    NUM_KV_GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """向量化版本：整页一次性 load K/V，block-wise online softmax。"""
    batch_id   = tl.program_id(0)
    head_id    = tl.program_id(1)
    kv_head_id = head_id // NUM_KV_GROUPS

    seq_len   = tl.load(Seq_lens + batch_id).to(tl.int32)
    num_pages = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale     = 1.0 / tl.math.sqrt(float(HEAD_DIM))

    d_range = tl.arange(0, HEAD_DIM)   # [D]
    s_range = tl.arange(0, BLOCK_SIZE) # [BS]
    q       = tl.load(Q + batch_id * stride_qb + head_id * stride_qh + d_range).to(tl.float32)  # [D]

    m_i = float('-inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], tl.float32)

    for page_idx in range(num_pages):
        phys       = tl.load(Block_table + batch_id * stride_tb + page_idx * stride_tp)
        token_offs = page_idx * BLOCK_SIZE + s_range  # [BS] 每个 slot 的全局 token 下标
        mask       = token_offs < seq_len             # [BS]

        # 整页 load K/V: [BLOCK_SIZE, HEAD_DIM]
        k_base = K_cache + phys * stride_kb + kv_head_id * stride_kh
        k = tl.load(
            k_base + s_range[:, None] * stride_ks + d_range[None, :],
            mask=mask[:, None], other=0.0,
        ).to(tl.float32)  # [BS, D]

        # 向量化 score: [BS]
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(mask, scores, float('-inf'))

        # Block-wise online softmax（安全处理全无效 page）
        m_block = tl.max(scores, 0)
        m_new   = tl.maximum(m_i, m_block)
        # 两者都是 -inf 时避免 NaN：alpha=1（l_i=0 acc=0，结果不变）
        alpha   = tl.where(m_new != float('-inf'), tl.exp(m_i - m_new), 1.0)
        p       = tl.exp(scores - m_new)  # [BS]
        p       = tl.where(mask, p, 0.0)
        l_i     = alpha * l_i + tl.sum(p, 0)

        v_base = V_cache + phys * stride_vb + kv_head_id * stride_vh
        v = tl.load(
            v_base + s_range[:, None] * stride_vs + d_range[None, :],
            mask=mask[:, None], other=0.0,
        ).to(tl.float32)  # [BS, D]

        # acc = alpha * acc + sum_{s} p[s] * v[s, :]
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    acc = acc / tl.maximum(l_i, 1e-9)
    tl.store(Out + batch_id * stride_ob + head_id * stride_oh + d_range, acc)


def paged_attn_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """M1.4 vectorized paged attention decode（公共接口，返回 bf16）。

    q:           [batch, num_q_heads, head_dim], bf16
    k_cache:     [num_blocks, block_size, num_kv_heads, head_dim], bf16
    v_cache:     [num_blocks, block_size, num_kv_heads, head_dim], bf16
    block_table: [batch, max_blocks_per_seq], int32
    seq_lens:    [batch], int32
    returns:     [batch, num_q_heads, head_dim], bf16
    """
    _check_inputs(q, k_cache, v_cache, block_table, seq_lens)
    B, H_q, D   = q.shape
    H_kv        = k_cache.shape[2]
    block_size  = k_cache.shape[1]
    num_kv_groups = H_q // H_kv
    out = torch.empty(B, H_q, D, dtype=torch.float32, device=q.device)

    _paged_decode_v4_kernel[(B, H_q)](
        q.float().contiguous(), k_cache, v_cache,
        block_table, seq_lens, out,
        *q.float().stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_table.stride(),
        *out.stride(),
        NUM_KV_GROUPS=num_kv_groups,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
    )
    return out.to(torch.bfloat16)
