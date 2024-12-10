from torch import nn
from quant_fn import quantize_tensor

class QuantizeTensor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__() 
        # Store quantization arguments
        self.quant_args = kwargs

    def forward(self, x):
        return quantize_tensor(x, **self.quant_args)