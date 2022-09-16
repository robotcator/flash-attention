import torch
import torch.utils.benchmark as benchmark

from flash_attention import _flash_attn
from attention import _attention
from torch_attention import _torch_attention

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--has_mask_bias", required=False, help="add bias in attention", type=bool, default=False)
parser.add_argument("--eval", required=False, help="test whether has backward", type=bool, default=False)

args = parser.parse_args()
print(args)


def benchmark_memory(fn, inputs, mask=None, bias=None, grad=None, eval=True, desc='', verbose=False, **kwinputs):
    def fwd(grad, inputs, mask=mask, bias=bias, **kwinputs):
        with torch.no_grad():
            y = fn(inputs, inputs, inputs, mask=mask, bias=bias, **kwinputs)

    
    def fwd_bwd(grad, inputs, mask=mask, bias=bias, **kwinputs):
        y = fn(inputs, inputs, inputs, mask=mask, bias=bias, **kwinputs)
        if type(y) is tuple:
            y = y[0]
        if grad is None:
            grad = torch.randn_like(y)
        else:
            if grad.shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')
        y.backward(grad, retain_graph=False)

    if eval:
        f = fwd
        if verbose:
            print ("using fwd func...")
    else:
        f = fwd_bwd
        if verbose:
            print ("using fwd and bwd func...")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    f(None, inputs, mask, bias)

    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2 ** 20) * 1000)
    if verbose:
        print(f"{desc} max memory: ", mem)
    torch.cuda.empty_cache()
    return mem


def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


def fun(seqlen=128, verbose=False, has_bias=True, has_mask=True, eval=True):
    bs = 1
    head = 4
    c_dim = 32
    seq_q = seq_k = seq_v = seqlen
    dtype = torch.bfloat16
    device = "cuda"

    inputs = torch.empty((bs, seq_q, head, seq_q, c_dim), dtype=dtype, device=device).normal_(mean=0, std=.5)
    inputs.requires_grad = True
    if verbose:
        print ("inputs shape: ", inputs.shape)
    # [bs, seq, seq, head, c_dim]

    if has_bias:
        bias = torch.randn(
            1, 1, head, seq_q, seq_k, dtype=dtype, device=device
        )
        bias.requires_grad = True
        if verbose:
            print ("bias shape: ", bias.shape)
        # [1, 1, seq, head, seq_k]
    else:
        bias = None

    if has_mask:
        mask = gen_attn_mask(
            (
                torch.rand(bs, seq_q, 1, 1, seq_k, dtype=dtype, device=device,) > 0.2
            ).type(dtype),
            -3e4,
        )
        if verbose:
            print ("mask shape: ", mask.shape)
    else:
        mask = None

    print ("processing seq length: {} in eval model {} ......".format(seqlen, eval))

    try:
        m1 = benchmark_memory(_attention, inputs, mask=mask, bias=bias, eval=eval, desc='Normal Attention forward')
        print (m1)
    except:
        print ("Normal Attention OOM")

    try:
        m2 = benchmark_memory(_flash_attn, inputs, mask=mask, bias=bias, eval=eval, desc='Flash Attention forward')
        print (m2)
    except:
        print ("Flash Attention OOM")


for seqlen in [2**8, 2**9, 600, 700, 800, 2**10, 1200, 1400, 2**11, 2500, 3000, 3500, 2**12]:
    if args.has_mask_bias:
        if not args.eval:
            fun(seqlen=seqlen, eval=False)
        else:
            fun(seqlen=seqlen, eval=True)
    else:
        if not args.eval:
            fun(seqlen=seqlen, has_bias=None, has_mask=None, eval=False)
        else:
            fun(seqlen=seqlen, has_bias=None, has_mask=None, eval=True)
    
