from torch import nn
from quant_fn import Qtensor, quantize_tensor

class QuantizedReLU(nn.Module):
    def __init__(self, symmetric):
        super().__init__() 
        self.symmetric = symmetric

    def forward(self, x):
        if not isinstance(x, Qtensor):
            raise TypeError("Input tensor must be a Qtensor with .tensor, .scale, and .zero_point attributes.")
        if not self.symmetric: 
            zp = x.zero_point[(...,) + (None,) * (x.tensor.ndim - x.zero_point.ndim)]
            x.tensor = x.tensor - zp

        x.tensor = nn.functional.relu(x.tensor)
        return x