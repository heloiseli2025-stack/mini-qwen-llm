# Contributing

## Development setup

```bash
git clone https://github.com/heloiseli2025-stack/mini-qwen-llm
cd mini-qwen-llm
pip install -e ".[dev]"
```

Run the test suite:
```bash
pytest tests/ -v
```

Lint:
```bash
ruff check mini_qwen/ tests/
```

## Coding principles

**Think before coding.** State assumptions explicitly. If multiple valid interpretations exist, surface the tradeoff rather than picking one silently. If a simpler approach exists, say so.

**Simplicity first.** Minimum code that solves the stated problem. No speculative features, no abstractions for single-use code, no error handling for scenarios that cannot occur.

**Surgical changes.** Touch only what the task requires. Don't improve adjacent code, reformat unrelated sections, or refactor things that aren't broken. Match the existing style.

**Define success before starting.** Turn tasks into verifiable goals — a test that passes, a benchmark number that's reached — before writing implementation code.

**Read before writing.** Before adding a function, check whether one already exists nearby. Duplicated logic that diverges silently is a common source of bugs.

**Tests verify behavior, not shape.** A test that checks `output.shape == (4, 128)` without checking values is not a correctness test. Assertions should tie to the actual behavior being claimed.

**Fail visibly.** Partial failures, skipped records, or truncated output should surface as explicit errors, not silent success.

## Kernel development conventions

Every Triton kernel must:

1. Have a docstring at the top of the file with the frozen signature (input shapes, dtypes, output shape)
2. Have a corresponding PyTorch reference implementation in the test file — this is the numerical oracle
3. Pass correctness tests at `rtol=1e-2, atol=1e-2` against the oracle in bf16
4. Be tested at both small shapes (fast CI) and shapes close to real inference dimensions

When writing pointer arithmetic involving strides, add a short comment showing the expected shape alongside the formula. This is where the majority of silent errors occur.

## Numerical tolerances

| dtype | rtol | atol |
|-------|------|------|
| fp32  | 1e-5 | 1e-5 |
| bf16  | 1e-2 | 1e-2 |

## Frozen interfaces

The following are stable and must not be changed without updating all consumers:

- `KVCacheConfig` fields and the KV cache tensor layout
- `block_table` shape and -1 convention for unallocated slots
- `Sequence` field names and append-only semantics for `output_token_ids` and `block_ids`
- Public kernel signatures (documented in each `mini_qwen/kernels/*.py` file header)

## Commit style

```
feat(paged-attn): implement decode kernel with GQA broadcast
fix(w4a16): correct int4 unpack bit order
perf(moe-gemm): reduce register pressure via shared memory tiling
test(scheduler): add block OOM test
docs(m3): add W4A16 benchmark results
```

One logical change per commit. The main branch should always be runnable.

## Benchmark results

Before reporting a benchmark number:

- Warmup ≥ 20 iterations
- Measure 100 iterations, report the **median**
- Synchronize CUDA before and after each timed block
- Use the same prompt length, batch size, and dtype as any comparison baseline

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full benchmark protocol.

## Model weights

Do not commit model weights. Download them with:

```bash
python scripts/download_model.py --model Qwen/Qwen3-8B
```

Quantized weights go in `weights/` (gitignored).
