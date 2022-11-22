import torch

# from attention import _attention
from torch_attention import _torch_attention as _attention
from flash_attention import _flash_attn

import numpy as np
import pytest


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

# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# # @pytest.mark.parametrize('c_dim', [64, 32, 16])
# @pytest.mark.parametrize('c_dim', [16])
# @pytest.mark.parametrize('seqlen', [64, 128, 256, 512, 1024, 1536, 2048])
def test_flash_attn_unpadded_shape1(seqlen, c_dim, dtype, device = "cuda"):
    # mini
    # bs = 2
    # head = 8
    # c_dim = 16
    # seq_q = seq_k = seq_v = 128
    # dtype = torch.half
    # device = "cuda"

    bs = 1
    head = 1
    c_dim = c_dim
    bs_seq = 1
    seq_q = seq_k = seq_v = seqlen
    dtype = dtype
    device = device

    inputs = torch.empty((bs, bs_seq, head, seq_q, c_dim), dtype=dtype, device=device).normal_(mean=0, std=.1)
    # debug data
    # inputs = torch.zeros((bs, bs_seq, seq_q, head, c_dim), dtype=dtype, device=device)
    # cnt = 0
    # for i in range(bs):
    #     for j in range(bs_seq):
    #         for k in range(seq_q):
    #             for l in range(head):
    #                 for m in range(c_dim):
    #                     # inputs[i][j][k][l][m] = (cnt % 1000) * 0.1
    #                     # cnt += 1
    #                     inputs[i][j][k][l][m] = 0.1
    # inputs = inputs.permute(0, 1, 3, 2, 4)
    inputs.requires_grad = True

    print ("inputs shape: ", inputs.shape)
    # [bs, seq, seq, head, c_dim]

    bias = torch.randn(
        bs, bs_seq, head, seq_q, seq_k, dtype=dtype, device=device
    )
    # debug data
    # cnt = 0
    # for i in range(1):
    #     for j in range(1):
    #         for k in range(head):
    #             for l in range(seq_q):
    #                 for m in range(seq_k):
    #                     # if m % 2 == 0:
    #                         # bias[i][j][k][l][m] = (cnt % 100)
    #                         # cnt += 1
    #                     bias[i][j][k][l][m] = 0.01
    bias.requires_grad = True
    print ("bias shape: ", bias.shape)
    # [1, 1, seq, head, seq_k]

    mask = gen_attn_mask(
        (
            torch.randn((bs, bs_seq, 1, 1, seq_k), dtype=dtype, device=device,) > 0.2
        ).type(dtype),
        -3e4,
    )
    # [bs, bs_seq, head, 1, seq_k]

    # debug data
    # cnt = 0
    # mask = torch.ones(bs, bs_seq, head, seq_q, seq_k, dtype=dtype, device=device,) * -1
    # for i in range(bs):
    #     for j in range(bs_seq):
    #         for k in range(head):
    #             for l in range(seq_q):
    #                 for m in range(seq_k):
    #                     # if m % 2 == 0:
    #                     #     mask[i][j][k][l][m] = cnt % 100
    #                     #     cnt += 1
    #                     mask[i][j][k][l][m] = 0.001

    # mask = mask.expand(bs, bs_seq, head, seq_q, seq_k)
    print ("mask shape: ", mask.shape)

    # bias = None
    # mask = None
    # [bs, seq_q, 1, 1, seq_k]

    # dump test data
    # import numpy as np
    # np.savetxt("benchmark_input.txt", X=inputs.detach().cpu().numpy().flatten())
    # if mask is not None:
    #     np.savetxt("benchmark_mask.txt", X=mask.detach().cpu().numpy().flatten())
    # if bias is not None:
    #     np.savetxt("benchmark_bias.txt", X=bias.detach().cpu().numpy().flatten())

    normal_attn_v1 = inputs.clone()
    if bias is not None:
        bias_v1 = bias.clone()
    else:
        bias_v1 = None
    output_ref = _attention(normal_attn_v1, normal_attn_v1, normal_attn_v1, bias=bias_v1, mask=mask, upcast=True)
    output_ref = output_ref.transpose(-2, -3)
    print ("attention ref output shape: ", output_ref.shape)

    normal_attn_v2 = inputs.clone()
    if bias is not None:
        bias_v2 = bias.clone()
    else:
        bias_v2 = None
    output_pt = _attention(normal_attn_v2, normal_attn_v2, normal_attn_v2, bias=bias_v2, mask=mask)
    # be careful here
    output_pt = output_pt.transpose(-2, -3)
    print ("attention output shape: ", output_pt.shape)

    normal_attn_flash = inputs.clone()
    if bias is not None:
        bias_v3 = bias.clone()
    else:
        bias_v3 = None
    output_flash = _flash_attn(normal_attn_flash, normal_attn_flash, normal_attn_flash, bias=bias_v3, mask=mask)
    print ("flash attn output shape: ", output_flash.shape)

    print (10 * "*" + "comparing forward" + 10 * "*" )
    # fp32 result
    print("Output max diff: {0}".format((output_flash - output_ref).abs().max().item()))
    print("Output mean diff: {0}".format((output_flash - output_ref).abs().mean().item()))

    print("Pytorch max diff: {0}".format((output_pt - output_ref).abs().max().item()))
    print("Pytorch mean diff: {0}".format((output_pt - output_ref).abs().mean().item()))

    print("Output max diff with Pytorch: {0}".format((output_flash - output_pt).abs().max().item()))
    print("Output mean diff with Pytorch: {0}".format((output_flash - output_pt).abs().mean().item()))

    # Check that FlashAttention's numerical error is at most twice the numerical error of a Pytorch implementation.
    print ("less than twice error: ", (output_flash - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item())
    print ()
    assert ((output_flash - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item())

    # g = torch.randn_like(output_flash)
    # cnt = 0
    g = torch.ones([bs, bs_seq, seq_q, head, c_dim], dtype=dtype, device=device)
    # for i in range(bs):
    #     for j in range(bs_seq):
    #         for k in range(seq_q):
    #             for l in range(head):
    #                 for m in range(c_dim):
    #                     g[i][j][k][l][m] = (cnt % 100) * 0.1
    #                     cnt += 1
    g.requires_grad = True

    if bias is None:
        dq_ref, dk_ref, dv_ref,  = torch.autograd.grad(output_ref, (normal_attn_v1, normal_attn_v1, normal_attn_v1, ), g)
        dq_pt, dk_pt, dv_pt,  = torch.autograd.grad(output_pt, (normal_attn_v2, normal_attn_v2, normal_attn_v2, ), g)
        dq, dk, dv,  = torch.autograd.grad(output_flash, (normal_attn_flash, normal_attn_flash, normal_attn_flash, ), g)
    else:
        dq_ref, dk_ref, dv_ref, dbias_ref = torch.autograd.grad(output_ref, (normal_attn_v1, normal_attn_v1, normal_attn_v1, bias_v1), g)
        dq_pt, dk_pt, dv_pt, dbias_pt = torch.autograd.grad(output_pt, (normal_attn_v2, normal_attn_v2, normal_attn_v2, bias_v2), g)
        dq, dk, dv, dbias = torch.autograd.grad(output_flash, (normal_attn_flash, normal_attn_flash, normal_attn_flash, bias_v3), g)
    
    print("Output dQ max diff: {0}".format( (dq - dq_ref).abs().max().item() ))
    print("Output dK max diff: {0}".format( (dk - dk_ref).abs().max().item() ))
    print("Output dV max diff: {0}".format( (dv - dv_ref).abs().max().item() ))

    print("Pytorch dQ max diff: {0}".format( (dq_pt - dq_ref).abs().max().item() ))
    print("Pytorch dK max diff: {0}".format( (dk_pt - dk_ref).abs().max().item() ))
    print("Pytorch dV max diff: {0}".format( (dv_pt - dv_ref).abs().max().item() ))

    print("Output dQ max diff with Pytorch: {0}".format( (dq - dq_pt).abs().max().item() ))
    print("Output dK max diff with Pytorch: {0}".format( (dk - dk_pt).abs().max().item() ))
    print("Output dV max diff with Pytorch: {0}".format( (dv - dv_pt).abs().max().item() ))

    print ("dq less than twice error: ", ((dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()) )
    print ("dk less than twice error: ", ((dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item()) )
    print ("dv less than twice error: ", ((dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item()) )

    if bias is not None:
        print ("dbias less than twice error: ", ((dbias - dbias_ref).abs().max().item() <= 2 * (dbias_pt - dbias_ref).abs().max().item()) )

    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item(), "dq larger than twice error"
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item(), "dq larger than twice error"
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item(), "dq larger than twice error"

    if bias is not None:
        print("Output dbias max diff: {0}".format( (dbias - dbias_ref).abs().max().item() ))
        print("Pytorch dbias max diff: {0}".format( (dbias - dbias_pt).abs().max().item() ))
        
        def is_same_matrix(pred, gt, abs_eps=0.01, relative_rps=0.03, verbose=False):
            diff = np.abs(pred - gt)

            cnt = 0
            for index, x in np.ndenumerate(diff):
                if x > abs_eps:
                    relative_diff = np.abs(x / gt[index])
                    if relative_diff > relative_rps:
                        cnt += 1
                        if verbose:
                            print (index, x, gt[index], pred[index], relative_diff)

            if cnt > 0:
                print ("not so match")
                return False
            else:
                return True

        # print ("is same matrix: ", is_same_matrix(dbias.detach().cpu().numpy(), 
        #                                         dbias_pt.detach().cpu().numpy(),
        #                                         verbose=True))
        assert (dbias - dbias_ref).abs().max().item() <= 2 * (dbias_pt - dbias_ref).abs().max().item(), "dbias larger than twice error"



# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# # @pytest.mark.parametrize('c_dim', [64, 32, 16])
# @pytest.mark.parametrize('c_dim', [16])
# @pytest.mark.parametrize('seqlen', [64, 128, 256, 512, 1024, 1536, 2048])
def test_flash_attn_unpadded_shape2(seqlen, c_dim, dtype, device = "cuda"):
    # mini
    # bs = 2
    # head = 8
    # c_dim = 16
    # seq_q = seq_k = seq_v = 128
    # dtype = torch.half
    # device = "cuda"

    bs = 1
    head = 1
    c_dim = c_dim
    bs_seq = 1
    seq_q = seq_k = seq_v = seqlen
    dtype = dtype
    device = device

    inputs = torch.empty((bs, bs_seq, head, seq_q, c_dim), dtype=dtype, device=device).normal_(mean=0, std=.5)
    # debug data
    # inputs = torch.zeros((bs, bs_seq, seq_q, head, c_dim), dtype=dtype, device=device)
    # cnt = 0
    # for i in range(bs):
    #     for j in range(bs_seq):
    #         for k in range(seq_q):
    #             for l in range(head):
    #                 for m in range(c_dim):
    #                     inputs[i][j][k][l][m] = (cnt % 10000) * 0.001
    #                     cnt += 1

    # inputs = inputs.permute(0, 1, 3, 2, 4)
    inputs.requires_grad = True

    print ("inputs shape: ", inputs.shape)
    # [bs, seq, seq, head, c_dim]

    bias = torch.randn(
        1, bs_seq, head, seq_q, seq_k, dtype=dtype, device=device
    )
    bias.requires_grad = True

    print ("bias shape: ", bias.shape)
    # [1, 1, seq, head, seq_k]

    mask = gen_attn_mask(
        (
            torch.randn((bs, bs_seq, head, 1, seq_k), dtype=dtype, device=device,) > 0.2
        ).type(dtype),
        -3e4,
    )
    # [bs, bs_seq, head, 1, seq_k]

    # debug data
    # mask = torch.ones(bs, bs_seq, head, 1, seq_k, dtype=dtype, device=device,) * -1
    # for i in range(bs):
    #     for j in range(bs_seq):
    #         for k in range(head):
    #             for l in range(1):
    #                 for m in range(seq_q):
    #                     if m % 2 == 0:
    #                         mask[i][j][k][l][m] = 0

    # mask = mask.expand(bs, bs_seq, head, seq_q, seq_k)
    print ("mask shape: ", mask.shape)

    # bias = None
    # mask = None
    # [bs, seq_q, 1, 1, seq_k]

    # np.savetxt("inputs_flash_seq{0}.data".format(seqlen), inputs.detach().cpu().numpy().flatten(), delimiter=" ")
    # if mask is not None:
    #     np.savetxt("attn_mask_flash_seq{0}.data".format(seqlen), mask.detach().cpu().numpy().flatten(), delimiter=" ")

    normal_attn_v1 = inputs.clone()
    output_ref = _attention(normal_attn_v1, normal_attn_v1, normal_attn_v1, bias=bias, mask=mask, upcast=True)
    output_ref = output_ref.transpose(-2, -3)
    print ("attention ref output shape: ", output_ref.shape)

    normal_attn_v2 = inputs.clone()
    output_pt = _attention(normal_attn_v2, normal_attn_v2, normal_attn_v2, bias=bias, mask=mask)
    # be careful here
    output_pt = output_pt.transpose(-2, -3)
    print ("attention output shape: ", output_pt.shape)

    normal_attn_flash = inputs.clone()
    output_flash = _flash_attn(normal_attn_flash, normal_attn_flash, normal_attn_flash, bias=bias, mask=mask)
    print ("flash attn output shape: ", output_flash.shape)
    # [bs, bs_seq, head, seq_k c_dim]
    
    # np.savetxt("output_torch_seq{0}.data".format(seqlen), output_pt.detach().cpu().numpy().flatten(), delimiter=" ")
    # np.savetxt("output_flash_seq{0}.data".format(seqlen), output_flash.detach().cpu().numpy().flatten(), delimiter=" ")
 
    print (10 * "*" + "comparing forward" + 10 * "*" )
    # fp32 result
    print("Output max diff: {0}".format((output_flash - output_ref).abs().max().item()))
    print("Output mean diff: {0}".format((output_flash - output_ref).abs().mean().item()))

    print("Pytorch max diff: {0}".format((output_pt - output_ref).abs().max().item()))
    print("Pytorch mean diff: {0}".format((output_pt - output_ref).abs().mean().item()))

    print("Output max diff with Pytorch: {0}".format((output_flash - output_pt).abs().max().item()))
    print("Output mean diff with Pytorch: {0}".format((output_flash - output_pt).abs().mean().item()))

    # Check that FlashAttention's numerical error is at most twice the numerical error of a Pytorch implementation.
    print ("less than twice error: ", (output_flash - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item())
    print ()
    assert (output_flash - output_ref).abs().max().item() <= 2 * (output_pt - output_ref).abs().max().item()

    g = torch.randn_like(output_flash)
    # dq_ref, dk_ref, dv_ref,  = torch.autograd.grad(output_ref, (normal_attn_v1, normal_attn_v1, normal_attn_v1, ), g)
    # dq_pt, dk_pt, dv_pt,  = torch.autograd.grad(output_pt, (normal_attn_v2, normal_attn_v2, normal_attn_v2, ), g)
    # dq, dk, dv,  = torch.autograd.grad(output_flash, (normal_attn_flash, normal_attn_flash, normal_attn_flash, ), g)

    dq_ref, dk_ref, dv_ref, dbias_ref = torch.autograd.grad(output_ref, (normal_attn_v1, normal_attn_v1, normal_attn_v1, bias), g)
    dq_pt, dk_pt, dv_pt, dbias_pt = torch.autograd.grad(output_pt, (normal_attn_v2, normal_attn_v2, normal_attn_v2, bias), g)
    dq, dk, dv, dbias = torch.autograd.grad(output_flash, (normal_attn_flash, normal_attn_flash, normal_attn_flash, bias), g)

    print("Output dQ max diff: {0}".format( (dq - dq_ref).abs().max().item() ))
    print("Output dK max diff: {0}".format( (dk - dk_ref).abs().max().item() ))
    print("Output dV max diff: {0}".format( (dv - dv_ref).abs().max().item() ))

    print("Pytorch dQ max diff: {0}".format( (dq_pt - dq_ref).abs().max().item() ))
    print("Pytorch dK max diff: {0}".format( (dk_pt - dk_ref).abs().max().item() ))
    print("Pytorch dV max diff: {0}".format( (dv_pt - dv_ref).abs().max().item() ))

    print("Output dQ max diff with Pytorch: {0}".format( (dq - dq_pt).abs().max().item() ))
    print("Output dK max diff with Pytorch: {0}".format( (dk - dk_pt).abs().max().item() ))
    print("Output dV max diff with Pytorch: {0}".format( (dv - dv_pt).abs().max().item() ))

    print ("dq less than twice error: ", ((dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item()) )
    print ("dk less than twice error: ", ((dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item()) )
    print ("dv less than twice error: ", ((dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item()) )

    assert (dq - dq_ref).abs().max().item() <= 2 * (dq_pt - dq_ref).abs().max().item(), "dq larger than twice error"
    assert (dk - dk_ref).abs().max().item() <= 2 * (dk_pt - dk_ref).abs().max().item(), "dq larger than twice error"
    assert (dv - dv_ref).abs().max().item() <= 2 * (dv_pt - dv_ref).abs().max().item(), "dq larger than twice error"

    if bias is not None:
        print("Output dbias max diff: {0}".format( (dbias - dbias_ref).abs().max().item() ))
        print("Pytorch dbias max diff: {0}".format( (dbias - dbias_pt).abs().max().item() ))
        assert (dbias - dbias_ref).abs().max().item() <= 2 * (dbias_pt - dbias_ref).abs().max().item(), "dbias larger than twice error"


# for dtype in [torch.float16]:
for dtype in [torch.float16, torch.bfloat16]:
    #for c_dim in [64]:
    for c_dim in [32]:
    # why 32 failed
    #for c_dim in [16]:
        # for seqlen in [64, 128, 256, 512]:
        # for seqlen in [64, 65, 127, 128, 255, 256, 257, 512]:
        for seqlen in [128, 255, 257]:
            print ("dtype={}, c_dim={}, seqlen={}".format(dtype, c_dim, seqlen))
            test_flash_attn_unpadded_shape1(seqlen, c_dim, dtype)


# for dtype in [torch.float16]:
# for dtype in [torch.float16, torch.bfloat16]:
#     for c_dim in [16, 32, 64]:
#         for seqlen in [64, 128, 256, 512, 1024, 2048]:
#             print ("dtype={}, c_dim={}, seqlen={}".format(dtype, c_dim, seqlen))
#             test_flash_attn_unpadded_shape2(seqlen, c_dim, dtype)
