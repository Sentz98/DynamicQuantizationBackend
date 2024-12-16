import unittest
import torch
import torch.nn as nn
import sys
import os
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant_fn import quantize_tensor, dequantize_tensor


class TestQuantization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Test Quantization")
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        parser.add_argument("--tolerance", "-t", type=float, default=1e-4, help="Tolerable MSE threshold")
        parser.add_argument("--size", type=int, nargs=4, default=[10, 3, 32, 32], help="Size of the input tensor (N, C, H, W)")
        args, _ = parser.parse_known_args()

        # Parse arguments and set class variables
        cls.tolerance = args.tolerance
        cls.size = args.size
        cls.verbose = args.verbose

    def setUp(self):
        # Generate random input tensor
        # Define a sample input tensor
        min_val = -0.4    # Minimum value of the range
        max_val = 2.0    # Maximum value of the range

        # Generate a tensor with values uniformly distributed within the range
        self.input_tensor = torch.rand(self.size) * (max_val - min_val) + min_val
        self.loss_fn = nn.MSELoss()

    def test_dymensions_and_types(self):
        # Per-Tensor Quantization Check
        quant_tensor = quantize_tensor(self.input_tensor, per_channel=False, symmetric=True, cast=False)
        self.assertEqual(quant_tensor.tensor.ndim, self.input_tensor.ndim,
                         "Quantized tensor dimensions do not match input tensor")
        self.assertEqual(quant_tensor.scale.shape, quant_tensor.zero_point.shape,
                         "Mismatch btw scale and zeropoint shape")

        # Per-Channel Quantization Check
        quant_channel = quantize_tensor(self.input_tensor, per_channel=True, symmetric=True, cast=False)
        self.assertEqual(quant_channel.tensor.size(0), self.input_tensor.size(0),
                         "Quantized per-channel tensor batch size mismatch")

    def test_symmetric_per_tensor(self):
        quant_st_pt = quantize_tensor(self.input_tensor, per_channel=False, symmetric=True, cast=False)
        deq_st_pt = dequantize_tensor(quant_st_pt)
        mse_st_pt = self.loss_fn(self.input_tensor, deq_st_pt)
        self.assertLess(mse_st_pt, self.tolerance, f"Symmetric, Per-Tensor MSE too high: {mse_st_pt}")
        if self.verbose:
            print(f"Symmetric loss per_tensor = {mse_st_pt}")

    def test_symmetric_per_channel(self):
        quant_st_pc = quantize_tensor(self.input_tensor, per_channel=True, symmetric=True, cast=False)
        deq_st_pc = dequantize_tensor(quant_st_pc)
        mse_st_pc = self.loss_fn(self.input_tensor, deq_st_pc)
        self.assertLess(mse_st_pc, self.tolerance, f"Symmetric, Per-Channel MSE too high: {mse_st_pc}")
        if self.verbose:
            print(f"Symmetric loss per_channel = {mse_st_pc}")

    def test_asymmetric_per_tensor(self):
        quant_as_pt = quantize_tensor(self.input_tensor, per_channel=False, symmetric=False, cast=False)
        deq_as_pt = dequantize_tensor(quant_as_pt)
        mse_as_pt = self.loss_fn(self.input_tensor, deq_as_pt)
        self.assertLess(mse_as_pt, self.tolerance, f"Asymmetric, Per-Tensor MSE too high: {mse_as_pt}")
        if self.verbose:
            print(f"Asymmetric loss per_tensor = {mse_as_pt}")

    def test_asymmetric_per_channel(self):
        quant_as_pc = quantize_tensor(self.input_tensor, per_channel=True, symmetric=False, cast=False)
        deq_as_pc = dequantize_tensor(quant_as_pc)
        mse_as_pc = self.loss_fn(self.input_tensor, deq_as_pc)
        self.assertLess(mse_as_pc, self.tolerance, f"Asymmetric, Per-Channel MSE too high: {mse_as_pc}")
        if self.verbose:
            print(f"Asymmetric loss per_channel = {mse_as_pc}")
    
    def test_ESTIMATE_per_tensor(self):

        min = self.input_tensor.amin(dim=[1,2,3], keepdim=False)
        max = self.input_tensor.amax(dim=[1,2,3], keepdim=False)
        external_param = quantize_tensor(self.input_tensor, min, max, per_channel=False, symmetric=True, cast=False)
        internal_param = quantize_tensor(self.input_tensor, per_channel=False, symmetric=True, cast=False)
        torch.testing.assert_close(external_param.tensor, internal_param.tensor)
        external_param = quantize_tensor(self.input_tensor, min, max, per_channel=False, symmetric=False, cast=False)
        internal_param = quantize_tensor(self.input_tensor, per_channel=False, symmetric=False, cast=False)
        torch.testing.assert_close(external_param.tensor, internal_param.tensor)
        min = self.input_tensor.amin(dim=[2,3], keepdim=False)
        max = self.input_tensor.amax(dim=[2,3], keepdim=False)
        external_param = quantize_tensor(self.input_tensor, min, max, per_channel=True, symmetric=True, cast=False)
        internal_param = quantize_tensor(self.input_tensor, per_channel=True, symmetric=True, cast=False)
        torch.testing.assert_close(external_param.tensor, internal_param.tensor)
        
if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])  # Prevent unittest from interpreting custom args
