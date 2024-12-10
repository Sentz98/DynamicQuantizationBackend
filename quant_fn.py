import torch
import math
from dataclasses import dataclass


@dataclass
class Qtensor:
    tensor: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor


def quantize_tensor(
    tensor,
    q_min=None,
    q_max=None,
    batch=True,
    symmetric=False,
    per_channel=False,
    channel_dim=1,
    dtype=torch.int8,
    cast=False
):
    """
    Quantizes a tensor using either per-tensor or per-channel quantization.
    Supports symmetric and asymmetric quantization.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        q_min (float or torch.Tensor, optional): Minimum quantization range. For per-channel, should be a tensor.
        q_max (float or torch.Tensor, optional): Maximum quantization range. For per-channel, should be a tensor.
        batch (bool, optional): Whether to handle batch dimension. Default: True.
        symmetric (bool, optional): Whether to use symmetric quantization. Default: False.
        per_channel (bool, optional): Whether to apply per-channel quantization. Default: False.
        channel_dim (int, optional): Dimension for per-channel quantization. Default: 1.
        dtype (torch.dtype, optional): Quantized data type. Default: torch.int8.
        cast (bool, optional): Whether to cast the quantized tensor to the given dtype. Default: True.

    Returns:
        Qtensor: Quantized tensor with scale and zero-point metadata.
    """
    if not torch.is_tensor(tensor):
       raise ValueError("Input must be a torch.Tensor.")

    if not dtype.is_floating_point:
        dtype_info = torch.iinfo(dtype)
    else:
        dtype_info = torch.finfo(dtype)
    dtype_min, dtype_max = dtype_info.min, dtype_info.max

    if not batch:
        tensor = tensor.unsqueeze(0)

    if per_channel:
        q_dim = channel_dim + 1
    else:
        q_dim = channel_dim
    data_dims = list(range(q_dim, tensor.ndim))

    # Compute min and max values
    if q_min is None or q_max is None:
        tensor_min = tensor.amin(dim=data_dims, keepdim=True)
        tensor_max = tensor.amax(dim=data_dims, keepdim=True)
    else:
        if not torch.is_tensor(q_min) or not torch.is_tensor(q_max):
            raise ValueError("q_min and q_max must be tensors.")
        if q_min.shape != q_max.shape:
            raise ValueError("q_min and q_max must have the same shape.")
        expected_numel = math.prod(dim for idx, dim in enumerate(tensor.shape) if idx not in data_dims)
        if q_min.numel() != expected_numel:
            raise ValueError(
                "Dimension mismatch: The total number of elements in `q_min` does not match "
                "the expected number of elements based on the tensor shape and specified dimensions. "
                f"Expected elements: {expected_numel}, but `q_min` contains {q_min.numel()} elements."
            )
        
        view_shape = list(tensor.shape)
        for dim in data_dims:
            view_shape[dim] = 1
        tensor_min = q_min.view(view_shape)
        tensor_max = q_max.view(view_shape)

    # Compute scale and zero-point
    if symmetric:
        if dtype_min != 0:
            tensor_range = 2 * torch.max(tensor_min.abs(), tensor_max.abs())
        else:
            tensor_range = torch.where(
                tensor_min.abs() > tensor_max.abs(),
                tensor_min,         #T                
                tensor_max          #F                          
            )    
        scale = tensor_range / (dtype_max - dtype_min) 
        zero_point = torch.zeros_like(tensor_min)
    else:
        scale = (tensor_max - tensor_min) / (dtype_max - dtype_min)
        zero_point = torch.round(dtype_min - tensor_min / scale)

    # Quantize the tensor
    quantized_tensor = torch.round((tensor / scale) + zero_point)

    # Cast to dtype
    if cast:
        quantized_tensor = quantized_tensor.to(dtype)
    return Qtensor(quantized_tensor, scale.squeeze(data_dims), zero_point.squeeze(data_dims))




def dequantize_tensor(qt, batch=True, per_channel=False, channel_dim=1, dtype=torch.float32, cast=True, verbose=False):
    if not isinstance(qt, Qtensor):
        raise ValueError("Input must be a Qtensor object with tensor, scale, and zero_point attributes.")

    if not batch:
        qt.tensor = qt.tensor.unsqueeze(0)
    
    sc = qt.scale[(...,) + (None,) * (qt.tensor.ndim - qt.scale.ndim)]
    zp = qt.zero_point[(...,) + (None,) * (qt.tensor.ndim - qt.zero_point.ndim)]

    # Perform dequantization
    dequantized_tensor = (qt.tensor - zp) * sc

    # Cast to the desired dtype
    if cast:
        if not dtype.is_floating_point:
            dtype_info = torch.iinfo(dtype)
        else:
            dtype_info = torch.finfo(dtype)
        dtype_min, dtype_max = dtype_info.min, dtype_info.max
        dequantized_tensor = dequantized_tensor.clamp(dtype_min, dtype_max).to(dtype)

    return dequantized_tensor


if __name__ == "__main__":
    input_tensor = torch.randn(10, 3, 5, 5)
    quant_st_pt = quantize_tensor(input_tensor, per_channel=False, symmetric=True, cast=False, dtype=torch.int8)
    deq_st_pt = dequantize_tensor(quant_st_pt)
    loss = torch.nn.MSELoss()
    ll = loss(input_tensor, deq_st_pt)
    print(ll)



