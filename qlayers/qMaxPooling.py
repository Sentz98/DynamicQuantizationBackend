from torch import nn
from quant_fn import Qtensor, quantize_tensor

class QuantizedMaxPooling(nn.Module):
    def __init__(self, maxp_layer):
        super().__init__()
        self.kernel_size = maxp_layer.kernel_size
        self.stride = maxp_layer.stride
        self.padding = maxp_layer.padding
        self.dilation = maxp_layer.dilation
        self.ceil_mode = maxp_layer.ceil_mode

    def forward(self, x):
        if not isinstance(x, Qtensor):
            raise TypeError("Input tensor must be a Qtensor with .tensor, .scale, and .zero_point attributes.")
        # Apply max pooling to the underlying tensor
        x.tensor = nn.functional.max_pool2d(
            x.tensor,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode
        )
        return x