import torch
from contextlib import contextmanager


@contextmanager
def cuda_memory_tracker():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        print(f"Start memory: {start_mem / 1e6:.2f} MB")

    yield

    if torch.cuda.is_available():
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"End memory: {end_mem / 1e6:.2f} MB")
        print(f"Peak memory: {peak_mem / 1e6:.2f} MB")
        print(f"Delta: {(end_mem - start_mem) / 1e6:.2f} MB")


# Usage
with cuda_memory_tracker():
    x = torch.randn(10000, 10000, device="cuda")
    y = x @ x
