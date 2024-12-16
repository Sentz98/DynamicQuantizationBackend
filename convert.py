import torch.nn as nn
import copy
from qlayers.qLinear import QuantizedLinear
from qlayers.qConv2d import QuantizedConv2d
from qlayers.qQuantize import QuantizeTensor
from qlayers.qDequantize import DequantizeTensor
from qlayers.qRelu import QuantizedReLU
from qlayers.qMaxPooling import QuantizedMaxPooling
from qlayers.qFlatten import QuantizedFlatten

def substitute_quant_layers(module: nn.Module, symmetric=True, per_channel=False, estimate=False, alpha=0.01):
    """
    Recursively substitutes nn.Linear and nn.Conv2d layers in a network with their quantized counterparts.

    Args:
        module (nn.Module): The neural network or submodule to modify.
        symmetric (bool): Whether to use symmetric quantization.
        per_channel (bool): Whether to apply per-channel quantization.
        estimate (bool): Whether to estimate additional parameters for quantization.
        alpha (float): Scaling factor for certain quantization methods.

    Returns:
        nn.Module: The modified network with quantized layers.
    """
    for child_name, child_module in module.named_children():
        # Recursively apply to child modules
        module.add_module(child_name, substitute_quant_layers(child_module, symmetric, per_channel, estimate, alpha))

        # Replace Linear layers with QuantizedLinear
        if isinstance(child_module, nn.Linear):
            quantized_linear = QuantizedLinear(child_module, symmetric=symmetric, per_channel=per_channel)
            module.add_module(child_name, quantized_linear)

        # Replace Conv2d layers with QuantizedConv2d
        elif isinstance(child_module, nn.Conv2d):
            quantized_conv = QuantizedConv2d(child_module, symmetric=symmetric, per_channel=per_channel, estimate=estimate, alpha=alpha)
            module.add_module(child_name, quantized_conv)

        # Replace ReLU layers with QuantizedReLU
        elif isinstance(child_module, nn.ReLU):
            quantized_relu = QuantizedReLU(symmetric)
            module.add_module(child_name, quantized_relu)

        # Replace MaxPool2d layers with QuantizedMaxPooling
        elif isinstance(child_module, nn.MaxPool2d):
            quantized_max_pool = QuantizedMaxPooling(child_module)
            module.add_module(child_name, quantized_max_pool)

        # Replace Flatten layers with QuantizedFlatten
        elif isinstance(child_module, nn.Flatten):
            quantized_flatten = QuantizedFlatten(child_module)
            module.add_module(child_name, quantized_flatten)

    return module

class QuantizedNetwork(nn.Module):
    def __init__(self, base_network: nn.Module, symmetric: bool, per_channel: bool, estimate: bool, alpha: float = None):
        """
        A wrapper for quantizing a neural network.

        Args:
            base_network (nn.Module): The original network to quantize.
            symmetric (bool): Whether to use symmetric quantization.
            per_channel (bool): Whether to apply per-channel quantization.
            estimate (bool): Whether to estimate additional parameters for quantization.
            alpha (float, optional): Scaling factor for certain quantization methods.
        """
        super(QuantizedNetwork, self).__init__()
        self.quantizer = QuantizeTensor(symmetric=symmetric, per_channel=per_channel)
        self.dequantizer = DequantizeTensor(per_channel=per_channel)

        self.network = copy.deepcopy(base_network)
        self.network = substitute_quant_layers(self.network, symmetric, per_channel, estimate, alpha)

    def forward(self, x, collect_activations=False):
        """
        Forward pass for the quantized network.

        Args:
            x (Tensor): Input tensor.
            collect_activations (bool): Whether to collect and return layer activations.

        Returns:
            Tensor or tuple: The output tensor, and optionally a dictionary of activations.
        """
        x = self.quantizer(x)

        activations = {} if collect_activations else None
        for layer in self.network:
            x = layer(x)
            if collect_activations:
                # Use the layer's class name and ID as the key
                layer_name = f"{layer.__class__.__name__}_{id(layer)}"
                activations[layer_name] = x

        x = self.dequantizer(x)

        return (x, activations) if collect_activations else x
