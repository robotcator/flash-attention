import torch
from typing import Optional, Callable, List, Tuple, Sequence

from unicore.modules import softmax_dropout


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def _attention(query, key, value, mask=None, bias=None, upcast=False) -> torch.Tensor:
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
            bias = bias.float()

    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    # if biases is not None:
    #     a += biases
    
    # if mask is not None:
    #     a.masked_fill_(mask < 0, float('-inf'))

    # a = softmax_no_cast(a, -1)
    a = softmax_dropout(a, dropout_prob=0, is_training=True, mask=mask, bias=bias)

    # [*, H, Q, C_hidden]
    b = torch.matmul(a, value)

    return b.to(dtype_og)
