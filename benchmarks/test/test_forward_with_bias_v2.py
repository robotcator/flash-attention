import torch
import torch.nn as nn

from functools import partial
import math
from typing import Optional, Callable, List, Tuple, Sequence
import numpy as np
import deepspeed

from flash_attn.flash_attn_interface import flash_attn_unpadded_func


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype
    # if(d is torch.bfloat16 and not deepspeed.utils.is_initialized()):
    #     with torch.cuda.amp.autocast(enabled=False):
    #         s = torch.nn.functional.softmax(t, dim=dim)
    # else:
    #     s = torch.nn.functional.softmax(t, dim=dim)
    s = torch.nn.functional.softmax(t, dim=dim)
    return s


def _attention(query, key, value, mask=None, biases=None, upcast=False) -> torch.Tensor:
    # upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
    # output back to fp16/bf16.
    dtype_og = query.dtype
    if upcast:
        query = query.float()
        key = key.float()
        value = value.float()
        if mask is not None:
            mask = mask.float()
        if bias is not None:
            biases = biases.float()

    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)
    # print ("q * k: ", a)
    # import pdb; pdb.set_trace()

    if biases is not None:
        print ("attn_shape = {}, bias_shape = {}".format(a.shape, biases.shape))
        a += biases
    # print ("after bias:", a)
    
    if mask is not None:
        # a += mask
        # import pdb; pdb.set_trace()
        # please do not use add now
        a.masked_fill_(mask < 0, float('-inf'))
        
    # print ("after mask:", a)

    a = softmax_no_cast(a, -1)
    # print ("softmax :", a)

    # [*, H, Q, C_hidden]
    b = torch.matmul(a, value)
    # print ("p * v: ", a)
    return b.to(dtype_og), a.to(dtype_og)


def _flash_attn(q, k, v, attn_mask=None, attn_bias=None):
    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype

    # [*, B, N, H, C]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # [B_flat, N, H, C]
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]
    
    # [B_flat * N, H, C]
    q = q.reshape(-1, *q.shape[-2:])
    k = k.reshape(-1, *k.shape[-2:])
    v = v.reshape(-1, *v.shape[-2:])
    
    q_max_s = n
    q_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device
    )

    k_max_s = n
    k_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=k.device
    )

    if attn_mask is not None:
        # import pdb; pdb.set_trace()
        attn_mask = attn_mask.reshape([batch_size , no_heads, n, n]).contiguous()

    if attn_bias is not None:
        # import pdb; pdb.set_trace()
        if attn_bias.is_contiguous:
            print ("attn_bias it not contiguous, stride is", attn_bias.stride())
        attn_bias = attn_bias.reshape([batch_size , no_heads, n, n]).contiguous()
        # attn_bias = attn_bias.reshape([batch_size , no_heads, n, n])
        print ("attn_bias stride is", attn_bias.stride())

    print ("check shapes q_shape = {} k_shape = {} v_shape = {}".format(q.shape, k.shape, v.shape))
    print ("check shapes q_cu_shape = {} k_cu_shape = {}".format(q_cu_seqlens.shape, k_cu_seqlens.shape))
    if attn_bias is not None:
        print ("attn_bias shape = {}".format(attn_bias.shape))

    out = flash_attn_unpadded_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        q_max_s,
        k_max_s,
        attn_mask=None,
        attn_bias=attn_bias,
        dropout_p = 0.,
        softmax_scale = 1., # q has been scaled already
    )

    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c) 
    return out


def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


torch.manual_seed(0)
# v2
# bs = 1
# seq = 128
# head = 1
# c_dim = 16

# mini
bs = 1
seq = 128
head = 1
c_dim = 16

seq_q = seq_k = seq_v = 128

print (10 * "*" + "prepare data" + 10 * "*" )
# dtype = torch.bfloat16
dtype = torch.half
device = "cuda"

# orig_tensor = torch.stack(
#     [ (i+1) * 0.1 * torch.randn((bs, seq, head, c_dim))  for i in range(seq) ]
#     ,dim = 1
# ).to(device).to(dtype)

orig_tensor = torch.empty((bs, seq, head, seq_q, c_dim), dtype=dtype, device=device).normal_(mean=0, std=.5)
orig_tensor.requires_grad = True
# print ("tensor: ", orig_tensor)
print ("origin shape: ", orig_tensor.shape)
# [bs, seq, seq, head, c_dim]

bias = torch.ones(
    1, 1, head, seq_q, seq_k, dtype=dtype, device=device
) * 1

print ("bias shape: ", bias.shape)
bias_broadcast = bias.expand([bs, seq, head, seq_q, seq_k])
print ("bias_broadcast shape: ", bias_broadcast.shape)

# print ("bias_broadcast: ", bias_broadcast)

print (10 * "*" + "normal attn fp32" + 10 * "*" )
normal_attn_v1 = orig_tensor.clone()
output_ref, softmax_output_ref = _attention(normal_attn_v1, normal_attn_v1, normal_attn_v1, biases=bias_broadcast, upcast=True)
# be careful here
output_ref = output_ref.transpose(-2, -3)
print ("attention output shape: ", output_ref.shape)
print (10 * "*" + "normal attn fp32" + 10 * "*" )
print ()


print (10 * "*" + "normal attn fp16" + 10 * "*" )
normal_attn_v2 = orig_tensor.clone()
output_pt, softmax_output_pt = _attention(normal_attn_v2, normal_attn_v2, normal_attn_v2, biases=bias_broadcast)
# be careful here
output_pt = output_pt.transpose(-2, -3)
print ("attention output shape: ", output_pt.shape)
print (10 * "*" + "normal attn fp32" + 10 * "*" )
print ()


print (10 * "*" + "flash attn" + 10 * "*" )
normal_attn_flash = orig_tensor.clone()
output3 = _flash_attn(normal_attn_flash, normal_attn_flash, normal_attn_flash, attn_bias=bias_broadcast)
# import pdb; pdb.set_trace()
print ("flash attn output shape: ", output3.shape)
print (10 * "*" + "flash attn" + 10 * "*" )
print ()

# print ("max abs error: ", (output3 - output_ref).abs().max())
# print ("all close at pre@.2: ", torch.allclose(output3, output_ref, atol=1e-2))

print (10 * "*" + "comparing forward" + 10 * "*" )
print("Output max diff: {0}".format((output3 - output_ref).abs().max().item()))
print("Output mean diff: {0}".format((output3 - output_ref).abs().mean().item()))

# print("Output max diff: {0}".format((output3[:,0,:,:,:] - output_ref[:,0,:,:,:]).abs().max().item()))
# print("Output max diff: {0}".format((output3[:,3,:,:,:] - output_ref[:,3,:,:,:]).abs().max().item()))

print("Pytorch max diff: {0}".format((output_pt - output_ref).abs().max().item()))
print("Pytorch mean diff: {0}".format((output_pt - output_ref).abs().mean().item()))

print("Output max diff with Pytorch: {0}".format((output3 - output_pt).abs().max().item()))
print("Output mean diff with Pytorch: {0}".format((output3 - output_pt).abs().mean().item()))

print ("less than twice error: ", (output3 - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item())
print (10 * "*" + "comparing forward" + 10 * "*" )
print ()

# max_diff = (output3 - output_ref).abs().max().item()
# relative_diff = (output_pt - output_ref).abs().max().item()

# for i in range(bs):
#     for j in range(seq_q):
#         for k in range(seq_k):
#             if (output3[i, j, k, :, :] - output_ref[i, j, k, :, :]).abs().max().item() >= 2 * (relative_diff):
#                 print ("i={}, j={}, k={} output3={}".format(i, j, k, output3[i, j, k, :, :].data))
#                 print ("i={}, j={}, k={} output_pt={}".format(i, j, k, output_ref[i, j, k, :, :].data))

# test backward

g = torch.randn_like(output3)
dq_ref, dk_ref, dv_ref  = torch.autograd.grad(output_ref, (normal_attn_v1, normal_attn_v1, normal_attn_v1), g)
dq_pt, dk_pt, dv_pt = torch.autograd.grad(output_pt, (normal_attn_v2, normal_attn_v2, normal_attn_v2), g)
dq, dk, dv, = torch.autograd.grad(output3, (normal_attn_flash, normal_attn_flash, normal_attn_flash), g)

print("Output dQ max diff: {0}".format( (dq - dq_ref).abs().max().item() ))
print("Output dK max diff: {0}".format( (dk - dk_ref).abs().max().item() ))
print("Output dV max diff: {0}".format( (dv - dv_ref).abs().max().item() ))

print("Pytorch dQ max diff: {0}".format( (dq_pt - dq_ref).abs().max().item() ))
print("Pytorch dK max diff: {0}".format( (dk_pt - dk_ref).abs().max().item() ))
print("Pytorch dV max diff: {0}".format( (dv_pt - dv_ref).abs().max().item() ))

print("Output dQ max diff with Pytorch: {0}".format( (dq - dq_pt).abs().max().item() ))
print("Output dK max diff with Pytorch: {0}".format( (dk - dk_pt).abs().max().item() ))
print("Output dV max diff with Pytorch: {0}".format( (dv - dv_pt).abs().max().item() ))

print ("less than twice error: ", ((dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()) )