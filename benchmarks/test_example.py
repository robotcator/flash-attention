import torch
import torch.nn as nn

from functools import partial
import math
from typing import Optional, Callable, List, Tuple, Sequence
import numpy as np

from time import perf_counter_ns

from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func, flash_attn_unpadded_func

# bs = 1
# seq = 20
# head = 32
# c_dim = 16

torch.manual_seed(0)
# v2
bs = 1
seq = 128
head = 16
c_dim = 32

orig_tensor = torch.stack(
    [ (i+1) * 0.1 * torch.ones((bs, seq, head, c_dim))  for i in range(seq) ]
    ,dim = 1
).cuda().to(torch.bfloat16)

print ("origin shape: ", orig_tensor.shape)

batch_size = bs * seq
seqlen = seq
max_s = seqlen
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                        device=orig_tensor.device)

print (cu_seqlens)
tensor_2d_pad = orig_tensor.reshape(-1, head, c_dim) 

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

print (output3.shape, output3.shape)
output3 = output3.reshape((bs, seq, seq, head, c_dim))