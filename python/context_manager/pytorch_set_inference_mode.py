import torch
import torch.nn as nn

from contextlib import contextmanager

model = nn.Module()  # Dummy model


@contextmanager
def inference_mode():
    """Disable gradient computation for inference"""
    was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(was_enabled)


# Usage
with inference_mode():
    output = model(input)  # No gradients computed
# Gradient state restored here
