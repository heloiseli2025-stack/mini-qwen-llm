"""Performance analysis utilities: timing and GPU memory statistics."""
import time
from contextlib import contextmanager
import torch


@contextmanager
def cuda_timer(name: str = ""):
    """CUDA event timer with microsecond precision. Falls back to time.perf_counter on CPU."""
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
    """Return peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0
