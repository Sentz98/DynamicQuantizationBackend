from torch import nn
from quant_fn import Qtensor

class QuantizedFlatten(nn.Module):
    def __init__(self, flatten_layer):
        super().__init__()
        self.start_dim = flatten_layer.start_dim
        self.end_dim = flatten_layer.end_dim

    def forward(self, x):
        if not isinstance(x, Qtensor):
            raise TypeError("Input tensor must be a Qtensor with .tensor, .scale, and .zero_point attributes.")
        # Apply flatten to the underlying tensor
        x.tensor = x.tensor.flatten(start_dim=self.start_dim, end_dim=self.end_dim)
        return x