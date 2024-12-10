import torch
from torch import nn
from quant_fn import Qtensor, quantize_tensor

class QuantizedBatchNorm2d(nn.Module):
    def __init__(self, batchnorm_layer, quantize_fn):
        """
        Converts a trained nn.BatchNorm2d layer into a quantized version.

        Args:
            batchnorm_layer (nn.BatchNorm2d): Trained BatchNorm2d layer to quantize.
            quantize_fn (function): Function to quantize tensors,
                                    should return (quantized_tensor, scale, zero_point).
        """
        super(QuantizedBatchNorm2d, self).__init__()

        # Extract parameters from the original layer
        self.num_features = batchnorm_layer.num_features
        self.eps = batchnorm_layer.eps
        self.momentum = batchnorm_layer.momentum
        self.affine = batchnorm_layer.affine
        self.track_running_stats = batchnorm_layer.track_running_stats

        # Quantize weights (gamma) and bias (beta) if affine=True
        if self.affine:
            self.qweight = quantize_fn(batchnorm_layer.weight.data)
            self.qbias = quantize_fn(batchnorm_layer.bias.data)
        else:
            self.qweight = self.qbias = None

        # Running mean and variance (not quantized here, but could be)
        if self.track_running_stats:
            self.running_mean = batchnorm_layer.running_mean.clone()
            self.running_var = batchnorm_layer.running_var.clone()

    def forward(self, x):
        assert isinstance(x, Qtensor), "Input tensor must be quantized"

        weight = self.qweight.tensor if self.affine else None
        bias = self.qbias.tensor if self.affine else None

        # Perform batch normalization using the quantized parameters
        resi32 = nn.functional.batch_norm(
            x,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            weight,
            bias,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps
        )

        resi8 = self.quantize_fn(resi32)
        return resi8