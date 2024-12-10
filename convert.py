import torch.nn as nn
import copy
from qlayers.qLinear import QuantizedLinear
from qlayers.qConv2d import QuantizedConv2d
from qlayers.qQuantize import QuantizeTensor
from qlayers.qDequantize import DequantizeTensor
from qlayers.qRelu import QuantizedRelu
from qlayers.qMaxPooling import QuantizedMaxPooling
from qlayers.qFlatten import QuantizedFlatten

def substitute_quant_layers(module:nn.Sequential, symmetric=True, per_channel=False, estimate=False):
    """
    Recursively substitutes nn.Linear and nn.Conv2d layers in a network with their quantized counterparts.
    
    Args:
        module (nn.Module): The neural network or submodule to modify.
        symmetric (bool): Whether to use symmetric quantization.
        per_channel (bool): Whether to apply per-channel quantization.
    
    Returns:
        nn.Module: The modified network with quantized layers.
    """
    for name, child in module.named_children():
        # Recursively apply to child modules
        module.add_module(name, substitute_quant_layers(child, symmetric, per_channel, estimate))
        
        # Replace Linear layers with QuantizedLinear
        if isinstance(child, nn.Linear):
            quantized_linear = QuantizedLinear(child, symmetric=symmetric, per_channel=per_channel)
            module.add_module(name, quantized_linear)
        
        # Replace Conv2d layers with QuantizedConv2d
        elif isinstance(child, nn.Conv2d):
            quantized_conv = QuantizedConv2d(child, symmetric=symmetric, per_channel=per_channel, estimate=estimate)
            module.add_module(name, quantized_conv)

        elif isinstance(child, nn.ReLU):
            quantized_relu = QuantizedRelu()
            module.add_module(name, quantized_relu)

        elif isinstance(child, nn.MaxPool2d):
            quantized_mp = QuantizedMaxPooling(child)
            module.add_module(name, quantized_mp)
        
        elif isinstance(child, nn.Flatten):
            quantized_f = QuantizedFlatten(child)
            module.add_module(name, quantized_f)
    
    return module

class QuantizedNetwork(nn.Module):
    def __init__(self, network:nn.Sequential, symmetric:bool, per_channel:bool, estimate:bool):
        super(QuantizedNetwork, self).__init__()
        self.quant = QuantizeTensor(symmetric=symmetric, per_channel=per_channel)
        self.network = copy.deepcopy(network)
        self.network = substitute_quant_layers(self.network, symmetric, per_channel, estimate)
        self.dequant = DequantizeTensor(per_channel=per_channel)

    def forward(self, x, collect_activations=False):  
        x = self.quant(x)

        activations = []
        for layer in self.network:
            x = layer(x)
            if collect_activations:
                activations.append(x)
                
        x = self.dequant(x)
        # breakpoint()
        return (x, activations) if collect_activations else x

