import torch
import torch.nn as nn

from functools import partial
import math
from typing import Optional, Callable, List, Tuple, Sequence
import numpy as np
import deepspeed

from time import perf_counter_ns

from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func, flash_attn_unpadded_func

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
    if(d is torch.bfloat16 and not deepspeed.utils.is_initialized()):
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s

def _attention(query, key, value, mask=None, biases=None) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # import pdb; pdb.set_trace()
    # [*, H, Q, K]
    a = torch.matmul(query, key)

    print ("q * k: ", a)

    if biases is None:
        biases = []
    for b in biases:
        a += b

    print ("after bias:", a)

    if mask is not None:
        a += mask

    print ("after mask:", a)

    a = softmax_no_cast(a, -1)
    print ("softmax :", a)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)
    print ("p * v: ", a)

    return a


torch.manual_seed(0)
# v2
bs = 1
seq = 2
head = 1
c_dim = 16

# import pdb; pdb.set_trace()

print (10 * "*" + "prepare data" + 10 * "*" )
# dtype = torch.bfloat16
dtype = torch.half
device = "cuda"

orig_tensor = torch.stack(
    [ (i+1) * 0.1 * torch.ones((bs, seq, head, c_dim))  for i in range(seq) ]
    ,dim = 1
).cuda().to(dtype)

print ("tensor: ", orig_tensor)
print ("origin shape: ", orig_tensor.shape)
# [bs, seq, seq, head, c_dim]

batch_size = bs * seq
seqlen = seq
max_s = seqlen
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                        device=orig_tensor.device)

print ("cu_seqlens: ", cu_seqlens)

# [bs, seq, seq, head, c_dim]
orig_tensor = orig_tensor.permute([0, 1, 3, 2, 4])
# [bs, seq, head, seq, c_dim]
print ("after permute: ", orig_tensor.shape)

print (10 * "*" + "end prepare data" + 10 * "*" )

print (10 * "*" + "normal attn" + 10 * "*" )
print ("normal attn: ", _attention(orig_tensor, orig_tensor, orig_tensor))
print (10 * "*" + "end normal attn" + 10 * "*" )

tensor_2d_pad = orig_tensor.reshape(-1, head, c_dim) 

print (10 * "*" + "flash attn without mask" + 10 * "*" )
output3 = flash_attn_unpadded_func(
    tensor_2d_pad,
    tensor_2d_pad,
    tensor_2d_pad,
    cu_seqlens,
    cu_seqlens,
    max_s,
    max_s,
    dropout_p = 0.,
    softmax_scale = 1., # q has been scaled already
)

print ("output3 shape: ", output3.shape)
output3 = output3.reshape((bs, seq, seq, head, c_dim))
print ("output3: ", output3.shape)
print (10 * "*" + "end flash attn without mask" + 10 * "*" )

def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask

# mask = gen_attn_mask(
#         (
#             # [bs, seq, h, seq, seq_k]
#             # [bs, seq, 1, 1, seq_k]
#             torch.ones(
#                 bs,
#                 seq,
#                 1, 
#                 1,
#                 seq,
#                 dtype=dtype,
#                 device="cuda",
#             )
#             > 0.2
#         ).type(dtype),
#         -1e5,
#     )
# unicore mask
#  torch.rand(
#     n_batch,
#     n_groups,
#     1,
#     1,
#     last_dim,
#     dtype=dtype,
#     device=test_device,
# )

print (10 * "*" + "flash attn with mask" + 10 * "*" )
mask = torch.randn(
                bs,
                seq,
                1, 
                1,
                seq,
                dtype=dtype,
                device="cuda",
            )

# [bs, group, 1, 1, seq_k]
seq_q = seq
seq_k = seq
print ("mask: ", mask.shape)
mask_exp = mask.expand([bs, seq_q, head, seq_q, seq_k])
print ("mask_exp: ", mask_exp.shape)
mask_batch = mask_exp.reshape((bs*seq_q, head, seq_q, seq_k))
print ("mask_exp: ", mask_batch.shape)

print ("mask: ", mask_batch)
print ("tensor: ", tensor_2d_pad)
print ("mask maximum number :", mask_batch.abs().max())

# bs * seq
# batch_size, num_heads, max_seqlen_q, max_seqlen_k
output4 = flash_attn_unpadded_func(tensor_2d_pad, 
                        tensor_2d_pad, 
                        tensor_2d_pad, 
                        cu_seqlens, 
                        cu_seqlens, 
                        max_s, 
                        max_s, 
                        # None,
                        attn_mask=mask_batch,
                        attn_bias=mask_batch,
                        dropout_p=0.0, 
                        softmax_scale=1.0)

output4 = output4.reshape((bs, seq, seq, head, c_dim))

print ("output4: ", output4.shape)

print (10 * "*" + "end flash attn with mask" + 10 * "*" )

print (10 * "*" + "normal attn with mask" + 10 * "*" )
print ("normal attn: ", _attention(orig_tensor, orig_tensor, orig_tensor, mask=mask))
print (10 * "*" + "end normal attn with mask" + 10 * "*" )

print ("all close on output3 and output4 max error", (output3 - output4).abs().max())
print ("all close on output3 and output4 min error", (output3 - output4).abs().min())
print ("all close on output3 and output4 num less min error", torch.sum( (output3 - output4).abs() <=(output3 - output4).abs().min()  ))
