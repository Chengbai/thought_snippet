import torch
import torch.distributed as dist

from contextlib import contextmanager


@contextmanager
def distributed_context(backend="nccl", **kwargs):
    """
    Context manager for distributed training.
    """
    try:
        # === SETUP PHASE (runs on entering 'with') ===
        dist.init_process_group(backend=backend, **kwargs)
        rank = dist.get_rank()
        print(f"Rank {rank}: Process group initialized")

        # === YIELD (pause here, run user code) ===
        yield  # Control goes to code inside 'with' block

        # === CLEANUP PHASE (runs on exiting 'with') ===
        # This runs AFTER the with block finishes

    finally:
        # finally ensures this ALWAYS runs, even if:
        # - Exception occurs in user code
        # - Exception occurs in setup
        # - User does early return
        if dist.is_initialized():
            rank = dist.get_rank()
            print(f"Rank {rank}: Cleaning up process group")
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed")


with distributed_context(backend="nccl"):
    # ↑ Execution enters here
    # 1. dist.init_process_group() is called
    # 2. yield is reached
    # ↓ Control transfers to this block

    tensor = torch.randn(100, device="cuda")
    dist.all_reduce(tensor)
    # Your code runs here

    # ↓ Block finishes
    # 3. Execution returns to after yield
    # 4. dist.destroy_process_group() is called
# ↓ Context manager exits
