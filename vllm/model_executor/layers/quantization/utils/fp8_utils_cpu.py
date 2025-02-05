# Adapted from https://github.com/sgl-project/sglang/pull/2575
import functools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch


from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def apply_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])

    dequant_weight = dequant_block_fp8_weight_naive(weight, weight_scale, block_size, input_2d.dtype)

    output = torch.nn.functional.linear(input_2d, dequant_weight, bias=None)
    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype)



def dequant_block_fp8_weight_naive(weight, weight_scale, block_size, dtype):

    assert len(block_size) == 2
    
    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale = weight_scale.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[0], dim=1)
        dequant_weight = weight.to(dtype) * weight_scale[:weight.shape[0], :weight.shape[1]].to(dtype)
    elif weight_shape_len == 3:
        weight_scale = weight_scale.repeat_interleave(block_size[0], dim=1).repeat_interleave(block_size[0], dim=2)
        dequant_weight = weight.to(dtype) * weight_scale[:, :weight.shape[1], :weight.shape[2]].to(dtype)
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    return dequant_weight


def input_to_float8(
        x: torch.Tensor,
        dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values "
    "with tensor-wise quantization."""
    if dtype is None:
        dtype = (torch.float8_e4m3fnuz
                 if current_platform.is_rocm() else torch.float8_e4m3fn)
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


def block_quant_to_tensor_quant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function converts block-wise quantization to tensor-wise
    quantization. The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are tensor-wise quantization tensor and tensor-wise
    quantization scale. Note only float8 is supported for now.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    x_dq_block_tiles = [[
        x_dq_block[
            j * block_n:min((j + 1) * block_n, n),
            i * block_k:min((i + 1) * block_k, k),
        ] for i in range(k_tiles)
    ] for j in range(n_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block_tiles[j][i][:, :] = x_dq_block_tiles[j][i] * x_s[j][i]

    x_q_tensor, scale = input_to_float8(x_dq_block, dtype=x_q_block.dtype)
    return x_q_tensor, scale



@functools.lru_cache
def get_w8a8_block_fp8_configs(N: int, K: int, block_n: int,
                               block_k: int) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.
    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n}, {block_k}].json"  # noqa: E501

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                "Using configuration from %s for W8A8 Block FP8 kernel.",
                config_file_path,
            )
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        "Using default W8A8 Block FP8 kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        config_file_path,
    )
    return None