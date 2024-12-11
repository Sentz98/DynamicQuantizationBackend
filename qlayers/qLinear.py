import torch
from torch import nn
from quant_fn import Qtensor, quantize_tensor


class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer, symmetric=True, per_channel=False, estimate=False):
        """
        Converts a trained nn.Linear layer into a quantized version.

        Args:
            linear_layer (nn.Linear): Trained Linear layer to quantize.
            quantize_fn (function): Function to quantize tensors,
                                    should return (quantized_tensor, scale, zero_point).
        """
        super(QuantizedLinear, self).__init__()

        # Extract parameters from the original layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.bias = linear_layer.bias is not None

        # Store quantization arguments
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.estimate = estimate

        # Quantize weights
        self.qweight = quantize_tensor(
                            linear_layer.weight, 
                            channel_dim=0, 
                            symmetric=True,
                            per_channel=self.per_channel
                        )
        self.fp_bias = linear_layer.bias


    def forward(self, x):
        assert isinstance(x, Qtensor), "Input tensor must be quantized"

        sc = x.scale[(...,) + (None,) * (x.tensor.ndim - x.scale.ndim)]
        zp = x.zero_point[(...,) + (None,) * (x.tensor.ndim - x.zero_point.ndim)]
        x.tensor = x.tensor - zp

        # Perform linear using the quantized weights and input
        linear_result = nn.functional.linear(
            x.tensor,
            self.qweight.tensor,
            None
        )

        # The bias is managed outside of conv operation bc need to bee shifted accordingly to input scale
        # so wont work with torch convolutional logic
        if self.bias:
            bias = self.fp_bias.unsqueeze(0)
            bias = bias[(...,) + (None,) * (linear_result.ndim - bias.ndim)]
            qbias = torch.round(bias / (self.qweight.scale.item() * sc))
            linear_result += qbias

        if self.estimate:
            q_min, q_max = estimateOutputRange(x.tensor, self.qweight.tensor, .5, self.per_channel)
            quantized_result = quantize_tensor(linear_result, q_min, q_max, symmetric=True, per_channel=False)
        else: 
            # Quantize the output
            quantized_result = quantize_tensor(linear_result, symmetric=self.symmetric, per_channel=self.per_channel)
        
        # Update scale
        quantized_result.scale = quantized_result.scale * x.scale * self.qweight.scale
        return quantized_result
    
def estimateOutputRange(input, weight, alpha, per_channel):
    if per_channel:
        raise NotImplementedError("HEHE volevi")
    
    positive_w_sum = torch.sum(weight[weight > 0])
    negative_w_sum = torch.sum(weight[weight < 0])

    input_min = input.amin(dim=[1], keepdim=False)
    input_max = input.amax(dim=[1], keepdim=False)

    # Calculate q_min and q_max as tensors
    q_min = (input_max * negative_w_sum + input_min * positive_w_sum) * alpha
    q_max = (input_min * negative_w_sum + input_max * positive_w_sum) * alpha
    
    return q_min , q_max

