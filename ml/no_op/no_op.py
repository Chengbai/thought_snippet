import torch.nn as nn


class NoOpLayer(nn.Module):
    def forward(self, x):
        return x


# Or simply use nn.Identity()
no_op = nn.Identity()
