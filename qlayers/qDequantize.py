from torch import nn
from quant_fn import Qtensor, dequantize_tensor

class DequantizeTensor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        # Store quantization arguments
        self.dequant_args = kwargs

    def forward(self, x):
        if not isinstance(x, Qtensor):
            raise TypeError("Input tensor must be a Qtensor with .tensor, .scale, and .zero_point attributes.")
        return dequantize_tensor(x, **self.dequant_args)