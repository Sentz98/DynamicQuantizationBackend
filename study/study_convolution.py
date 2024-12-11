import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from qlayers.qConv2d import QuantizedConv2d
from quant_fn import quantize_tensor, dequantize_tensor

class QuantizationStudy:
    def __init__(self, tolerance=5e-4, verbose=False):
        self.tolerance = tolerance
        self.verbose = verbose
        self.loss_fn = nn.MSELoss()

    def run_experiment(self, output_channels, image_size, repetitions=10):
        results = {
            "symmetric_per_tensor": [],
            "asymmetric_per_tensor": [],
            "estimate_symmetric": []
        }

        for _ in range(repetitions):
            # Initialize Conv2d layer
            conv_layer = nn.Conv2d(in_channels=3, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(conv_layer.weight, mean=0.0, std=0.02)

            # Generate random input tensor
            input_tensor = torch.randn((1, 3, *image_size))

            # Compute original output
            original_output = conv_layer(input_tensor)

            # Symmetric per-tensor quantization
            loss = self._test_quantization(conv_layer, input_tensor, original_output, symmetric=True, per_channel=False)
            results["symmetric_per_tensor"].append(loss)

            # Asymmetric per-tensor quantization
            loss = self._test_quantization(conv_layer, input_tensor, original_output, symmetric=False, per_channel=False)
            results["asymmetric_per_tensor"].append(loss)

            # Estimated symmetric per-tensor quantization
            loss = self._test_quantization(conv_layer, input_tensor, original_output, symmetric=True, per_channel=False, estimate=True)
            results["estimate_symmetric"].append(loss)

        return results

    def _test_quantization(self, conv_layer, input_tensor, original_output, symmetric, per_channel, estimate=False):
        # Wrap the Conv2d layer with the QuantizedConv2d class
        quantized_conv = QuantizedConv2d(conv_layer, symmetric=symmetric, per_channel=per_channel, estimate=estimate)

        # Quantize input and run forward pass
        quantized_input = quantize_tensor(input_tensor, symmetric=symmetric, per_channel=per_channel)
        q_output = quantized_conv(quantized_input)

        # Dequantize output and compute loss
        deq_output = dequantize_tensor(q_output, per_channel=per_channel)
        loss = self.loss_fn(original_output, deq_output).item()

        if self.verbose:
            print(f"Loss: {loss}")

        return loss

    def save_results(self, results, filename="results.json"):
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)

    def save_and_plot(self, results, output_file="results.png"):
        # Aggregate results
        means = {key: np.mean(values) for key, values in results.items()}
        stds = {key: np.std(values) for key, values in results.items()}

        # Plot results
        labels = list(means.keys())
        x = np.arange(len(labels))
        means = [means[key] for key in labels]
        stds = [stds[key] for key in labels]

        plt.bar(x, means, yerr=stds, capsize=5)
        plt.xticks(x, labels, rotation=45)
        plt.ylabel("Loss")
        plt.title("Quantization Loss Comparison")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
    
    def plot_error_trends(self, results):
        approaches = ["symmetric_per_tensor", "asymmetric_per_tensor", "estimate_symmetric"]

        # Parse the keys to get channels and sizes
        parsed_keys = [(key.split("_size_"), key) for key in results.keys()]
        channels = sorted({int(k[0][0].split("_")[1]) for k in parsed_keys})
        sizes = sorted({int(k[0][1].split("x")[0]) for k in parsed_keys})

        for approach in approaches:
            plt.figure(figsize=(10, 6))
            for channel in channels:
                errors = []
                for size in sizes:
                    key = f"channels_{channel}_size_{size}x{size}"
                    if key in results:
                        errors.append(np.mean(results[key][approach]))
                    else:
                        errors.append(None)

                plt.plot(sizes, errors, label=f"{channel} channels", marker="o")

            plt.xlabel("Image Size")
            plt.ylabel("Loss")
            plt.title(f"Quantization Loss Trend: {approach.replace('_', ' ').title()}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"study/{approach}_trend.png")
            plt.show()
    
    def plot_approaches_for_channels(self, results, channels):
        sizes = sorted({int(key.split("_size_")[1].split("x")[0]) for key in results.keys() if f"channels_{channels}" in key})
        approaches = ["symmetric_per_tensor", "asymmetric_per_tensor", "estimate_symmetric"]

        plt.figure(figsize=(10, 6))

        for approach in approaches:
            errors = []
            for size in sizes:
                key = f"channels_{channels}_size_{size}x{size}"
                if key in results:
                    errors.append(np.mean(results[key][approach]))
                else:
                    errors.append(None)

            plt.plot(sizes, errors, label=approach.replace('_', ' ').title(), marker="o")

        plt.xlabel("Image Size")
        plt.ylabel("Loss")
        plt.title(f"Quantization Loss for {channels} Output Channels")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"study/approaches_channels_{channels}.png")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization Study")
    parser.add_argument("--tolerance", type=float, default=5e-4, help="Tolerable MSE threshold")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output_file", type=str, default="study/results.png", help="File to save plot")
    parser.add_argument("--results_file", type=str, default="study/results.json", help="File to save results")
    parser.add_argument("--repetitions", type=int, default=1000, help="Number of repetitions for statistical relevance")
    args = parser.parse_args()

    study = QuantizationStudy(tolerance=args.tolerance, verbose=args.verbose)

    # Test configurations
    output_channels_list = [2**i for i in range(6)]
    image_sizes = [(2**i, 2**i) for i in range(1, 10)]

    all_results = {}

    for output_channels in tqdm(output_channels_list, desc="Output Channels"):
        for image_size in tqdm(image_sizes, desc="Image Sizes", leave=False):
            key = f"channels_{output_channels}_size_{image_size[0]}x{image_size[1]}"
            all_results[key] = study.run_experiment(output_channels, image_size, repetitions=args.repetitions)

    # Save and plot results 
    study.save_results(all_results, filename=args.results_file)

    study.plot_error_trends(all_results)

    study.plot_approaches_for_channels(all_results, channels=4)
