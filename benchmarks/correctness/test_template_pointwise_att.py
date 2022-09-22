import torch

# from attention import _attention
from torch_attention import _torch_attention as _attention
from flash_attention import _flash_attn

import numpy as np
import pytest


from flash_attn.flash_attn_interface import flash_attn_unpadded_func


def is_same_matrix(pred, gt, abs_eps=0.01, relative_rps=0.03, verbose=False):
    diff = np.abs(pred - gt)

    cnt = 0
    for index, x in np.ndenumerate(diff):
        if x > abs_eps:
            relative_diff = np.abs(x / gt[index])
            if relative_diff > relative_rps:
                cnt += 1
                if verbose:
                    print (index, x, gt[index], relative_diff)

    if cnt > 0:
        print ("not so match")
        return False
    else:
        return True


def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


def test_attn():
    dtype = torch.half
    device = "cuda"

    q = torch.randn(1, 256, 256, 4, 1, 16, dtype=dtype, device=device)
    k = torch.randn(1, 256, 256, 4, 4, 16, dtype=dtype, device=device)
    v = torch.randn(1, 256, 256, 4, 4, 16, dtype=dtype, device=device)

    # print ("q shape = {}, k shape = {}, v shape = {}".format(q.shape, k.shape, v.shape))

    o = _attention(q, k, v, mask=None, bias=None)
    o = o.transpose(-2, -3).contiguous()

    output_flash = _flash_attn(q, k, v, mask=None, bias=None)

    print("Output max diff: {0}".format((output_flash - o).abs().max().item()))
    print (is_same_matrix(o.detach().cpu().numpy(), output_flash.detach().cpu().numpy()))

test_attn()