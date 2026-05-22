"""性能分析工具：计时、显存统计。"""
import time
from contextlib import contextmanager
import torch


@contextmanager
def cuda_timer(name: str = ""):
    """CUDA event 计时，精度到微秒。CPU fallback 用 time.perf_counter。"""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        print(f"[timer] {name}: {ms:.3f} ms")
    else:
        t0 = time.perf_counter()
        yield
        ms = (time.perf_counter() - t0) * 1000
        print(f"[timer] {name}: {ms:.3f} ms (CPU)")


def peak_memory_gb() -> float:
    """返回 GPU 显存峰值（GB）。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0
