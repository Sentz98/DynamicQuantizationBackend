import torch
from torch import nn
from quant_fn import Qtensor, quantize_tensor

class QuantizedConv2d(nn.Module):
    def __init__(self, conv_layer, symmetric=True, per_channel=False, estimate=False, alpha=0.1, verbose = False):
        """
        Converts a trained nn.Conv2d layer into a quantized version.

        Args:
            conv_layer (nn.Conv2d): Trained Conv2d layer to quantize.
            *args: Additional arguments passed to the quantization function.
                - symmetric (bool): Whether to use symmetric quantization.
                - per_channel (bool): Whether to apply per-channel quantization.
        """
        super(QuantizedConv2d, self).__init__()

        # Extract parameters from the original Conv2d layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.bias = conv_layer.bias

        # Store quantization arguments
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.estimate = estimate
        self.alpha = alpha

        # Logging
        self.verbose = verbose

        # Quantize weights
        self.qweight = quantize_tensor(
                            conv_layer.weight, 
                            channel_dim=0, 
                            symmetric=True,
                            per_channel=self.per_channel
                        )
        
        if self.estimate and (per_channel or not symmetric):
            raise NotImplementedError("HEHE volevi")
    
    def estimateConvOutputRange(self, input, weight):        
        positive_w_sum = torch.sum(weight[weight > 0])
        negative_w_sum = torch.sum(weight[weight < 0])

        input_min = input.amin(dim=[1,2,3], keepdim=False)
        input_max = input.amax(dim=[1,2,3], keepdim=False)

        if self.verbose:
            print(f"Input range [{input_min}, {input_max}]")

        # Calculate q_min and q_max as tensors
        q_min = (input_max * negative_w_sum + input_min * positive_w_sum) * self.alpha
        q_max = (input_min * negative_w_sum + input_max * positive_w_sum) * self.alpha

        if self.verbose:
            print(f"Estimated output range [{q_min}, {q_max}]")
        
        return q_min , q_max

    def forward(self, x):
        if not isinstance(x, Qtensor):
            raise TypeError("Input tensor must be a Qtensor with .tensor, .scale, and .zero_point attributes.")
        
        # # manage zero point
        sc = x.scale[(...,) + (None,) * (x.tensor.ndim - x.scale.ndim)]
        zp = x.zero_point[(...,) + (None,) * (x.tensor.ndim - x.zero_point.ndim)]
        x.tensor = x.tensor - zp

        # Perform convolution using the quantized weights and input
        conv_result = nn.functional.conv2d(
            x.tensor,
            self.qweight.tensor,
            bias= None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        # The bias is managed outside of conv operation bc need to bee shifted accordingly to input scale
        # so wont work with torch convolutional logic
        bias = self.bias.unsqueeze(0)
        bias = bias[(...,) + (None,) * (conv_result.ndim - bias.ndim)]
        qbias = torch.round( bias / (self.qweight.scale.item() * sc))
        conv_result += qbias

        if self.estimate:
            q_min, q_max = self.estimateConvOutputRange(x.tensor, self.qweight.tensor)
            if self.verbose:
                print(f"Theoretical output range [{conv_result.amin(dim=[1,2,3], keepdim=False)}, {conv_result.amax(dim=[1,2,3], keepdim=False)}]")
            quantized_result = quantize_tensor(conv_result, q_min, q_max, symmetric=True, per_channel=False)
        else: 
            # Quantize the output
            quantized_result = quantize_tensor(conv_result, symmetric=self.symmetric, per_channel=self.per_channel)

        # Update scale
        quantized_result.scale = quantized_result.scale * x.scale * self.qweight.scale
        return quantized_result

