import torch
import torch.utils.benchmark as benchmark

from flash_attention import _flash_attn
from attention import _attention
from torch_attention import _torch_attention

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--has_mask_bias", required=False, help="add bias in attention", type=bool, default=False)

args = parser.parse_args()
print(args)


def benchmark_combined(fn, inputs, mask=None, bias=None, grad=None, repeats=10, desc='', verbose=False, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Forward + Backward pass')

    def f(grad, inputs, mask=mask, bias=bias, **kwinputs):
        y = fn(inputs, inputs, inputs, mask=mask, bias=bias, **kwinputs)
        if type(y) is tuple:
            y = y[0]
        if grad is None:
            grad = torch.randn_like(y)
        else:
            if grad.shape != y.shape:
                raise RuntimeError('Grad shape does not match output shape')
        y.backward(grad, retain_graph=True)
    
    t = benchmark.Timer(
            stmt='f(grad, inputs, mask=mask, bias=bias, **kwinputs)',
            globals={'f': f, 'fn': fn, 'inputs': inputs, 'mask': mask, 'bias': bias, 'grad': grad, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_forward(fn, inputs, mask=None, bias=None, grad=None, repeats=10, desc='', verbose=False, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Forward pass with no grad')

    def f(grad, inputs, mask=mask, bias=bias, **kwinputs):
        with torch.no_grad():
            y = fn(inputs, inputs, inputs, mask=mask, bias=bias, **kwinputs)
    
    t = benchmark.Timer(
            stmt='f(grad, inputs, mask=mask, bias=bias, **kwinputs)',
            globals={'f': f, 'fn': fn, 'inputs': inputs, 'mask': mask, 'bias': bias, 'grad': grad, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


def fun(seqlen=128, verbose=False, has_bias=True, has_mask=True):
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

    print ("processing seq length: {} ......".format(seqlen))
    try:
        t1, m1 = benchmark_combined(_attention, inputs, mask=mask, bias=bias, repeats=100, desc='Normal Attention forward')
        # import pdb; pdb.set_trace()
        # print (m1)
        # raw_times / number_per_run * 1000 ms
        print (m1.raw_times[0])
    except:
        print ("normal attention OOM")

    try:
        t2, m2 = benchmark_combined(_flash_attn, inputs, mask=mask, bias=bias, repeats=100, desc='Flash Attention forward')
        # print (m2)
        print (m2.raw_times[0])
    except:
        print ("flash attention OOM")


for seqlen in [2**8, 2**9, 600, 700, 800, 2**10, 1200, 1400, 2**11, 2500, 3000, 3500, 2**12]:
    if has_mask_bias:
        fun(seqlen=seqlen)
    else:
        fun(seqlen=seqlen, has_bias=None, has_mask=None)

