

import torch
import torch.nn.functional as F

dtype = torch.bfloat16

m = 26
n = 2048
k = 7168
e = 256
topk = 8

a = torch.randn((m, k), dtype=dtype) / 10
w1 = torch.randn((e, 2 * n, k),  dtype=dtype) / 10
w2 = torch.randn((e, k, n), dtype=dtype) / 10

score = torch.randn((m, e), dtype=dtype)

def SiluAndMul(x):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

def torch_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


def fused_experts_cpu(hidden_states,
                        w1,
                        w2,
                        topk_weights,
                        topk_ids,
                        inplace=False,
                        use_fp8_w8a8=False,
                        use_int8_w8a16=False,
                        use_int4_w4a16=False,
                        w1_scale=None,
                        w2_scale=None,
                        w1_zp=None,
                        w2_zp=None,
                        a1_scale=None,
                        a2_scale=None,
                        block_shape=None):
    
    input_2d = hidden_states.view(-1, input.shape[-1])
    original_M = original_M.data
    original_N = original_N.data
    output_shape = [*input.shape[:-1], original_M]
    dequant_weight = dequant_block_fp8_weight_naive(weight, weight_scale, block_size, input_2d.dtype, original_M, original_N)
    
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


topk_weight, topk_ids = torch.topk(score, topk)
inplace = True
use_fp8_w8a8 = True
use_int8_w8a16 = False
use_int4_w4a16 = False

w1_scale =  torch.rand(e, 2 * n // 128, k//128)
w2_scale = torch.rand(e, k//128, n // 128)
block_shape = [128,128]

w1_zp = None
w2_zp = None
a1_scale = None
a2_scale = None

fused_experts_cpu(a, w1, w2, topk_weight, topk_ids,
                        inplace=inplace,
                        use_fp8_w8a8=use_fp8_w8a8,
                        use_int8_w8a16=use_int8_w8a16,
                        use_int4_w4a16=use_int4_w4a16,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        w1_zp=w1_zp,
                        w2_zp=w2_zp,
                        a1_scale=a1_scale,
                        a2_scale=a2_scale,
                        block_shape=block_shape)