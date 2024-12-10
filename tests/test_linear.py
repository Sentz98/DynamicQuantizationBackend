import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import argparse
import torch
import torch.nn as nn
from qlayers.qLinear import QuantizedLinear  # Assuming QuantizedLinear exists
from quant_fn import quantize_tensor, dequantize_tensor  # Existing quantization functions

class TestQuantizedLinear(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Test Quantized Linear Layer")
        parser.add_argument("--tolerance", type=float, default=5e-4, help="Tolerable MSE threshold")
        parser.add_argument("--size", type=int, nargs=2, default=[10, 1280], help="Size of the input tensor (N, Features)")
        args, _ = parser.parse_known_args()

        # Parse arguments and set class variables
        cls.tolerance = args.tolerance
        cls.size = args.size

    def setUp(self):
        # Create and initialize the original Linear layer
        self.linear_layer = nn.Linear(in_features=self.size[1], out_features=64)
        nn.init.normal_(self.linear_layer.weight, mean=0.0, std=0.02)  # Random initialization for weights

        # Define a sample input tensor
        self.input_tensor = torch.randn(size=self.size)  # Random input tensor

        # Calculate original output for comparison
        self.original_output = self.linear_layer(self.input_tensor)

        # Define a loss function for comparison
        self.loss_fn = nn.MSELoss()

    def test_symmetric_per_tensor(self):
        # Wrap the Linear layer with the QuantizedLinear class
        quantized_linear = QuantizedLinear(self.linear_layer, symmetric=True, per_channel=False)

        # Quantize input and run forward pass
        quantized_input = quantize_tensor(self.input_tensor, symmetric=True, per_channel=False)
        q_output = quantized_linear(quantized_input)

        # Dequantize output and compute loss
        deq_output = dequantize_tensor(q_output, per_channel=False)
        loss = self.loss_fn(self.original_output, deq_output)
        self.assertLess(loss.item(), self.tolerance, f"Loss too high for symmetric per-tensor quantization: {loss.item()}")
        print(f"Symmetric Tensor loss is {loss.item()}")
    
    def test_asymmetric_per_tensor(self):
        # Wrap the Linear layer with the QuantizedLinear class
        quantized_linear = QuantizedLinear(self.linear_layer, symmetric=False, per_channel=False)

        # Quantize input and run forward pass
        quantized_input = quantize_tensor(self.input_tensor, symmetric=False, per_channel=False)
        q_output = quantized_linear(quantized_input)

        # Dequantize output and compute loss
        deq_output = dequantize_tensor(q_output, per_channel=False)
        loss = self.loss_fn(self.original_output, deq_output)
        self.assertLess(loss.item(), self.tolerance, f"Loss too high for asymmetric per-tensor quantization: {loss.item()}")
        print(f"Asymmetric Tensor loss is {loss.item()}")
    
    def test_ESTIMATE_symmetric_per_tensor(self):
        # Wrap the Conv2d layer with the QuantizedLinear class
        quantized_linear = QuantizedLinear(self.linear_layer, symmetric=True, per_channel=False, estimate=True)

        # Quantize input and run forward pass
        quantized_input = quantize_tensor(self.input_tensor, symmetric=True, per_channel=False)
        q_output = quantized_linear(quantized_input)

        # Dequantize output and compute loss
        deq_output = dequantize_tensor(q_output, per_channel=False)
        loss = self.loss_fn(self.original_output, deq_output)
        self.assertLess(loss.item(), 1, f"ciao: {loss.item()}")
        print(f"Symmetric ESTIMATED Tensor loss is {loss.item()}")


    # def test_symmetric_per_channel(self):
    #     # Wrap the Linear layer with the QuantizedLinear class
    #     quantized_linear = QuantizedLinear(self.linear_layer, symmetric=True, per_channel=True)

    #     # Quantize input and run forward pass
    #     quantized_input = quantize_tensor(self.input_tensor, symmetric=True, per_channel=True)
    #     q_output = quantized_linear(quantized_input)

    #     # Dequantize output and compute loss
    #     deq_output = dequantize_tensor(q_output, per_channel=True)
    #     loss = self.loss_fn(self.original_output, deq_output)
    #     self.assertLess(loss.item(), self.tolerance, f"Loss too high for symmetric per-channel quantization: {loss.item()}")
    #     print(f"Symmetric Per-Channel loss is {loss.item()}")
    
    # def test_asymmetric_per_channel(self):
    #     # Wrap the Linear layer with the QuantizedLinear class
    #     quantized_linear = QuantizedLinear(self.linear_layer, symmetric=False, per_channel=True)

    #     # Quantize input and run forward pass
    #     quantized_input = quantize_tensor(self.input_tensor, symmetric=False, per_channel=True)
    #     q_output = quantized_linear(quantized_input)

    #     # Dequantize output and compute loss
    #     deq_output = dequantize_tensor(q_output, per_channel=True)
    #     loss = self.loss_fn(self.original_output, deq_output)
    #     self.assertLess(loss.item(), self.tolerance, f"Loss too high for asymmetric per-channel quantization: {loss.item()}")
    #     print(f"Asymmetric Per-Channel loss is {loss.item()}")

if __name__ == "__main__":
    unittest.main()
