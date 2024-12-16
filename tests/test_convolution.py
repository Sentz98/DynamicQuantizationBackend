import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import argparse
import torch
import torch.nn as nn
from qlayers.qConv2d import QuantizedConv2d
from quant_fn import quantize_tensor, dequantize_tensor

class TestQuantizedConv2d(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Test Quantization")
        parser.add_argument("--tolerance", type=float, default=5e-4, help="Tolerable MSE threshold")
        parser.add_argument("--size", type=int, nargs=4, default=[1, 3, 32, 32], help="Size of the input tensor (N, C, H, W)")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        args, _ = parser.parse_known_args()

        # Parse arguments and set class variables
        cls.tolerance = args.tolerance
        cls.size = args.size
        cls.verbose = args.verbose

    def setUp(self):
        # Create and initialize the original Conv2d layer
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv_layer.weight, mean=0.0, std=0.2)  # Random initialization for weights

        # Define a sample input tensor
        min_val = -1.0   # Minimum value of the range
        max_val = 1.0    # Maximum value of the range

        # Generate a tensor with values uniformly distributed within the range
        self.input_tensor = torch.rand(self.size) * (max_val - min_val) + min_val
        # self.input_tensor = torch.randn(size=self.size)  # Random input tensor

        # Calculate original output for comparison
        self.original_output = self.conv_layer(self.input_tensor)

        # if self.verbose:
        #     print("original")
        #     print(self.original_output)

        # Define a loss function for comparison
        self.loss_fn = nn.MSELoss()

    def test_symmetric_per_tensor(self):
        # Wrap the Conv2d layer with the QuantizedConv2d class
        quantized_conv = QuantizedConv2d(self.conv_layer, symmetric=True, per_channel=False)

        # Quantize input and run forward pass
        quantized_input = quantize_tensor(self.input_tensor, symmetric=True, per_channel=False)
        q_output = quantized_conv(quantized_input)

        # Dequantize output and compute loss
        deq_output = dequantize_tensor(q_output, per_channel=False)
        loss = self.loss_fn(self.original_output, deq_output)
        self.assertLess(loss.item(), self.tolerance, f"Loss too high for symmetric per-tensor quantization: {loss.item()}")
        print(f"Symmetric Tensor loss is {loss.item()}")
        print(q_output.tensor.min())
        # if self.verbose:
        #     print("------quant------")
        #     print(q_output)
        #     print("-----dequant-----")
        #     print(deq_output)
        #     print("\n")
    
    def test_asymmetric_per_tensor(self):
        # Wrap the Conv2d layer with the QuantizedConv2d class
        quantized_conv = QuantizedConv2d(self.conv_layer, symmetric=False, per_channel=False)

        # Quantize input and run forward pass
        quantized_input = quantize_tensor(self.input_tensor, symmetric=False, per_channel=False)
        q_output = quantized_conv(quantized_input)

        # Dequantize output and compute loss
        deq_output = dequantize_tensor(q_output, per_channel=False)
        loss = self.loss_fn(self.original_output, deq_output)
        self.assertLess(loss.item(), self.tolerance, f"Loss too high for asymmetric per-tensor quantization: {loss.item()}")
        print(f"Asymmetric Tensor loss is {loss.item()}")
        # if self.verbose:
        #     print("------quant------")
        #     print(q_output)
        #     print("-----dequant----")
        #     print(deq_output)
        #     print("\n")

    def test_ESTIMATE_symmetric_per_tensor(self):
        # Wrap the Conv2d layer with the QuantizedConv2d class
        quantized_conv = QuantizedConv2d(self.conv_layer, symmetric=True, per_channel=False, estimate=True, alpha=0.00001, verbose=self.verbose)

        # Quantize input and run forward pass
        quantized_input = quantize_tensor(self.input_tensor, symmetric=True, per_channel=False)
        q_output = quantized_conv(quantized_input)

        # Dequantize output and compute loss
        deq_output = dequantize_tensor(q_output, per_channel=False)
        loss = self.loss_fn(self.original_output, deq_output)
        self.assertLess(loss.item(), 1, f"ciao: {loss.item()}")
        print(f"Symmetric ESTIMATED Tensor loss is {loss.item()}")
        # if self.verbose:
        #     print("------quant------")
        #     print(q_output)
        #     print("-----dequant----")
        #     print(deq_output)
        #     print("\n")

    # def test_symmetric_per_channel(self):
    #     # Wrap the Conv2d layer with the QuantizedConv2d class
    #     quantized_conv = QuantizedConv2d(self.conv_layer, symmetric=True, per_channel=True)

    #     # Quantize input and run forward pass
    #     quantized_input = quantize_tensor(self.input_tensor, symmetric=True, per_channel=True)
    #     q_output = quantized_conv(quantized_input)

    #     # Dequantize output and compute loss
    #     deq_output = dequantize_tensor(q_output, per_channel=True)
    #     loss = self.loss_fn(self.original_output, deq_output)
    #     self.assertLess(loss.item(), self.tolerance, f"Loss too high for symmetric per-channel quantization: {loss.item()}")

    

    # def test_asymmetric_per_channel(self):
    #     # Wrap the Conv2d layer with the QuantizedConv2d class
    #     quantized_conv = QuantizedConv2d(self.conv_layer, symmetric=False, per_channel=True)

    #     # Quantize input and run forward pass
    #     quantized_input = quantize_tensor(self.input_tensor, symmetric=False, per_channel=True)
    #     q_output = quantized_conv(quantized_input)

    #     # Dequantize output and compute loss
    #     deq_output = dequantize_tensor(q_output, per_channel=True)
    #     loss = self.loss_fn(self.original_output, deq_output)
    #     self.assertLess(loss.item(), self.tolerance, f"Loss too high for asymmetric per-channel quantization: {loss.item()}")

if __name__ == "__main__":
    unittest.main()
