from torch import nn
from quant_fn import Qtensor, quantize_tensor

class QuantizedRelu(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, x):
        if not isinstance(x, Qtensor):
            raise TypeError("Input tensor must be a Qtensor with .tensor, .scale, and .zero_point attributes.")
        x.tensor = nn.functional.relu(x.tensor)
        return x