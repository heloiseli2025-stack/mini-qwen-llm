# CLAUDE.md

Project: mini-qwen-llm — Triton+PyTorch Qwen3 inference engine.

## Stack

- Python 3.10–3.12, PyTorch ≥ 2.4, Triton ≥ 3.0, CUDA ≥ 12.1
- Test runner: `pytest tests/`
- Lint: `ruff check mini_qwen/ tests/`
- Language: code in English; comments and docstrings in Chinese (project convention)

## Key rules

- Match existing style. Do not reformat unrelated code.
- Every Triton kernel needs a PyTorch oracle in the test file; correctness threshold is `atol=1e-2` for bf16.
- Frozen interfaces (KVCacheConfig, block_table convention, Sequence fields, kernel signatures) must not be changed without updating all callers. See ARCHITECTURE.md §2.
- Do not commit model weights — they go in `weights/` (gitignored).
- Block pre-allocation for decode happens in `Scheduler.step()`, not inside `ModelRunner.run_decode()`.
- The fused QKV kernel (`fused_qkv_rope`) is only valid when all sequences in the batch have the same length. The decode path uses an unfused fallback with per-sequence RoPE positions.
