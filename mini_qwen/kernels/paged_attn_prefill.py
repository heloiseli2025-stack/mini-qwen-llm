"""Paged Attention Prefill Kernel (M1.5 implementation).

FROZEN SIGNATURE (frozen after M1.5 is complete):
  paged_attn_prefill(
      q:           [total_tokens, num_q_heads, head_dim], bf16
      k:           [total_tokens, num_kv_heads, head_dim], bf16
      v:           [total_tokens, num_kv_heads, head_dim], bf16
      k_cache:     [num_blocks, block_size, num_kv_heads, head_dim], bf16
      v_cache:     [num_blocks, block_size, num_kv_heads, head_dim], bf16
      block_table: [batch, max_blocks_per_seq], int32
      cu_seqlens:  [batch+1], int32   # (0, S0, S0+S1, ...)
      max_seqlen:  int
  ) -> [total_tokens, num_q_heads, head_dim], bf16
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════════════
# helper: write K/V to paged cache
# Grid: (total_tokens, H_kv)
# ══════════════════════════════════════════════════════════════════════════════

@triton.jit
def _write_kv_cache_kernel(
    K_in, V_in,           # [total, H_kv, D]
    K_cache, V_cache,     # [num_blocks, page_size, H_kv, D]
    Block_table,          # [batch, max_blocks]
    Batch_ids,            # [total] int32
    Positions,            # [total] int32  (position within its sequence)
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_kc_b, stride_kc_s, stride_kc_h, stride_kc_d,
    stride_vc_b, stride_vc_s, stride_vc_h, stride_vc_d,
    stride_bt_b, stride_bt_p,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    tok_id  = tl.program_id(0)
    head_id = tl.program_id(1)

    batch_id = tl.load(Batch_ids + tok_id).to(tl.int32)
    pos      = tl.load(Positions  + tok_id).to(tl.int32)
    page_idx = pos // PAGE_SIZE
    slot     = pos  % PAGE_SIZE
    phys     = tl.load(Block_table + batch_id * stride_bt_b + page_idx * stride_bt_p)

    d_range = tl.arange(0, HEAD_DIM)
    k = tl.load(K_in + tok_id * stride_kt + head_id * stride_kh + d_range)
    tl.store(K_cache + phys * stride_kc_b + slot * stride_kc_s + head_id * stride_kc_h + d_range, k)

    v = tl.load(V_in + tok_id * stride_vt + head_id * stride_vh + d_range)
    tl.store(V_cache + phys * stride_vc_b + slot * stride_vc_s + head_id * stride_vc_h + d_range, v)


# ══════════════════════════════════════════════════════════════════════════════
# M1.5 — FlashAttention v2 style prefill kernel (bf16 input, fp32 accumulation)
# Grid: (ceil(max_seqlen / BLOCK_Q), batch, H_q)
# ══════════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 16, 'BLOCK_KV': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_KV': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
    ],
    key=['HEAD_DIM', 'NUM_KV_GROUPS'],
)
@triton.jit
def _paged_prefill_attn_kernel(
    Q, K, V,              # [total, H_q / H_kv, D]  bf16
    Cu_seqlens,           # [batch+1] int32
    Out,                  # [total, H_q, D]  fp32 (caller converts to bf16)
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    q_tile_id  = tl.program_id(0)
    batch_id   = tl.program_id(1)
    head_id    = tl.program_id(2)
    kv_head_id = head_id // NUM_KV_GROUPS

    seq_start = tl.load(Cu_seqlens + batch_id).to(tl.int32)
    seq_end   = tl.load(Cu_seqlens + batch_id + 1).to(tl.int32)
    seq_len   = seq_end - seq_start

    q_start = q_tile_id * BLOCK_Q
    if q_start >= seq_len:
        return

    d_range = tl.arange(0, HEAD_DIM)
    q_range = q_start + tl.arange(0, BLOCK_Q)
    q_mask  = q_range < seq_len

    # Load Q as bf16 (keep original dtype for Tensor Core usage)
    q_ptrs = (Q + (seq_start + q_range)[:, None] * stride_qt
               + head_id * stride_qh
               + d_range[None, :])
    q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.bfloat16)  # [BQ, D]

    scale = 1.0 / tl.math.sqrt(float(HEAD_DIM))

    m_i = tl.full([BLOCK_Q], float('-inf'), tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, HEAD_DIM], tl.float32)

    kv_upper = q_start + BLOCK_Q

    for kv_start in range(0, kv_upper, BLOCK_KV):
        kv_range = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask  = kv_range < seq_len
        causal    = q_range[:, None] >= kv_range[None, :]
        full_mask = causal & kv_mask[None, :] & q_mask[:, None]

        # Load K as bf16
        k_ptrs = (K + (seq_start + kv_range)[:, None] * stride_kt
                   + kv_head_id * stride_kh
                   + d_range[None, :])
        k_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.bfloat16)  # [BKV, D]

        # Tensor Core: bf16 @ bf16.T → fp32 accumulation by default
        scores = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * scale
        scores = tl.where(full_mask, scores, float('-inf'))

        m_block = tl.max(scores, 1)
        m_new   = tl.maximum(m_i, m_block)
        alpha   = tl.where(m_new != float('-inf'), tl.exp(m_i - m_new), 1.0)

        p = tl.exp(scores - m_new[:, None])
        p = tl.where(full_mask, p, 0.0)
        l_i = alpha * l_i + tl.sum(p, 1)

        # Load V as bf16
        v_ptrs = (V + (seq_start + kv_range)[:, None] * stride_vt
                   + kv_head_id * stride_vh
                   + d_range[None, :])
        v_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.bfloat16)

        # Tensor Core: bf16 p @ bf16 V → fp32
        acc = alpha[:, None] * acc + tl.dot(p.to(tl.bfloat16), v_tile).to(tl.float32)
        m_i = m_new

    acc = acc / tl.maximum(l_i[:, None], 1e-9)
    out_ptrs = (Out + (seq_start + q_range)[:, None] * stride_ot
                 + head_id * stride_oh
                 + d_range[None, :])
    tl.store(out_ptrs, acc.to(tl.float32), mask=q_mask[:, None])


# ══════════════════════════════════════════════════════════════════════════════
# public interface
# ══════════════════════════════════════════════════════════════════════════════

def paged_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """M1.5 FlashAttention-style causal prefill + KV cache write (returns bf16).

    q/k/v:       [total_tokens, H_q/H_kv, head_dim], bf16
    k_cache/v_cache: [num_blocks, block_size, H_kv, head_dim], bf16
    block_table: [batch, max_blocks], int32
    cu_seqlens:  [batch+1], int32  cumulative seq lens starting at 0
    max_seqlen:  int
    """
    assert q.is_cuda, "inputs must be on CUDA"
    assert block_table.dtype == torch.int32

    B               = cu_seqlens.shape[0] - 1
    total, H_q, D   = q.shape
    H_kv            = k.shape[1]
    num_kv_groups   = H_q // H_kv
    page_size       = k_cache.shape[1]
    device          = q.device

    # ── 1. batch_ids / positions (all CUDA, no Python loop) ──────────────────
    # batch_ids[i] = which sequence this token belongs to
    batch_ids = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int32, device=device),
        (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64),
    )  # [total]
    # positions[i] = offset of this token within its sequence
    token_indices = torch.arange(total, dtype=torch.int32, device=device)
    seq_starts    = cu_seqlens[batch_ids.to(torch.int64)]        # [total]
    positions     = token_indices - seq_starts                    # [total]

    # ── 2. K/V scatter → paged cache ─────────────────────────────────────
    _write_kv_cache_kernel[(total, H_kv)](
        k, v, k_cache, v_cache,
        block_table, batch_ids, positions,
        *k.stride(),
        *v.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *block_table.stride(),
        PAGE_SIZE=page_size,
        HEAD_DIM=D,
    )

    # ── 3. FlashAttention prefill (bf16 input, Tensor Core) ──────────────────
    out  = torch.empty(total, H_q, D, dtype=torch.float32, device=device)
    grid = lambda meta: (triton.cdiv(max_seqlen, meta['BLOCK_Q']), B, H_q)

    _paged_prefill_attn_kernel[grid](
        q.contiguous(), k, v,        # keep bf16, no fp32 conversion
        cu_seqlens, out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        NUM_KV_GROUPS=num_kv_groups,
        HEAD_DIM=D,
    )
    return out.to(torch.bfloat16)


def write_kv_decode(
    k_new: torch.Tensor,        # [B, H_kv, head_dim] bf16
    v_new: torch.Tensor,        # [B, H_kv, head_dim] bf16
    k_cache: torch.Tensor,      # [num_blocks, block_size, H_kv, head_dim]
    v_cache: torch.Tensor,
    block_table: torch.Tensor,  # [B, max_blocks] int32
    positions: torch.Tensor,    # [B] int32 — global position of the new token in each sequence
) -> None:
    """Write new K/V from the decode step (1 token per sequence) into the paged cache."""
    assert k_new.is_cuda
    B, H_kv, D = k_new.shape
    page_size = k_cache.shape[1]
    batch_ids = torch.arange(B, dtype=torch.int32, device=k_new.device)
    k_c = k_new.contiguous()
    v_c = v_new.contiguous()
    _write_kv_cache_kernel[(B, H_kv)](
        k_c, v_c, k_cache, v_cache,
        block_table, batch_ids, positions,
        *k_c.stride(), *v_c.stride(),
        *k_cache.stride(), *v_cache.stride(),
        *block_table.stride(),
        PAGE_SIZE=page_size, HEAD_DIM=D,
    )
