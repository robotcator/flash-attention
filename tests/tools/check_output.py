from audioop import bias
from operator import truediv
from socket import NI_NAMEREQD
import numpy as np
import torch

batch_size = 1
nheads = 1
headdim = 16
seq = 8
max_seqlen_q_ = seq 
max_seqlen_k_ = seq

dtypes = np.float16


q_cpu = np.zeros((batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim), dtype=dtypes)
k_cpu = np.zeros((batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim), dtype=dtypes)
v_cpu = np.zeros((batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim), dtype=dtypes)
  
cnt = 0
for i in range(batch_size * max_seqlen_k_ * max_seqlen_k_):
    for j in range(nheads):
        for k in range(headdim):
            q_cpu[i][j][k] = cnt * 0.001
            k_cpu[i][j][k] = cnt * 0.001
            v_cpu[i][j][k] = cnt * 0.001
            cnt += 1

bias_ref = np.zeros([batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_], dtype=dtypes)
cnt = 0
for i in range(batch_size * max_seqlen_k_):
    for j in range(nheads):
        for k in range(max_seqlen_q_):
            for l in range(max_seqlen_k_):
                bias_ref[i][j][k][l] = cnt * 0.1
                cnt += 1


# dout = np.random.rand(batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim).astype(dtype=dtypes)
cnt = 0
dout = np.ones([batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim], dtype=dtypes)
for i in range(batch_size * max_seqlen_k_ * max_seqlen_k_):
    for j in range(nheads):
        for k in range(headdim):
            dout[i][j][k] = cnt * 0.001
            cnt += 1

def softmax(logit):
    max_value_over_last_dim = np.max(logit, axis=-1, keepdims=True)
    logit_sub_max_value = logit - max_value_over_last_dim

    exp_x = np.exp(logit_sub_max_value)
    
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return probs


def fwd(q, k, v, max_seqlen_q, bias=None):
    
    batch_size = int(q.shape[0] / max_seqlen_q)
    head_num = q.shape[1]
    head_dim = q.shape[2]

    q = q.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    k = k.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    v = v.reshape(batch_size, max_seqlen_q, head_num, head_dim)

    q = q.transpose(0,2,1,3)
    k = k.transpose(0,2,1,3)
    v = v.transpose(0,2,1,3)

    # print ("data q block 0 = {}".format(q[0, 0, :, :]))

    s = np.matmul(q, k.transpose(0, 1, 3, 2))
    
    if bias is not None:
        s = s + bias

    p = softmax(s)

    o = np.matmul(p, v)
    
    # o = o.transpose(0,2,1,3).reshape(batch_size * max_seqlen_q, head_num, head_dim)
    return s, p, o, q, k, v


def bwd(dout, q, k, v, max_seqlen_q, bias=None):
    s, p, o, _, _, _ = fwd(q, k, v, max_seqlen_q=max_seqlen_q, bias=bias)

    batch_size = int(q.shape[0] / max_seqlen_q)
    head_num = q.shape[1]
    head_dim = q.shape[2]

    dout = dout.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    dout = dout.transpose(0, 2, 1, 3)
    # import pdb; pdb.set_trace()

    q = q.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    k = k.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    v = v.reshape(batch_size, max_seqlen_q, head_num, head_dim)

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # get dv
    dv = np.matmul(p.transpose(0, 1, 3, 2), dout)

    # get dp
    dp = np.matmul(dout, v.transpose(0, 1, 3, 2))

    # ds_{i:} = P_{i:} \dot dP_{i:} - D_{i}P_{i:}

    ds = np.zeros([batch_size, head_num, max_seqlen_q, max_seqlen_q])
    for b in range(batch_size):
        for h in range(head_num):
            for i in range(max_seqlen_q):
                # please refer equation 4
                Di = 0.0
                for l in range(max_seqlen_q):
                    Di += p[b][h][i][l] * dp[b][h][i][l]

                for j in range(max_seqlen_q):
                    ds[b][h][i][j] = p[b][h][i][j] * (dp[b][h][i][j] - Di)

    # get dq
    dq = np.matmul(ds, k)
    # dq = dq.transpose(0, 2, 1, 3)

    # get dk
    dk = np.matmul(ds.transpose(0, 1, 3, 2), q)
    # dk = dk.transpose(0, 2, 1, 3)

    if bias is None:
        dbias = None
    else:
        dbias = ds.reshape(-1, *bias.shape).sum(axis=0)

    return dq, dk, dv, ds, dp, dbias


def fwd_pt(q_pt, k_pt, v_pt, bias=None):
    s = torch.matmul(q_pt, k_pt.transpose(-1, -2))

    if bias is not None:
        s = s + bias
    
    p = torch.nn.functional.softmax(s, dim=-1)

    o = torch.matmul(p, v_pt)
    return s, p, o


def bwd_pt(dout, q, k, v, max_seqlen_q, bias=None):
    # q is [batch * seq * seq, head, head_dim]
    q_pt, k_pt, v_pt, dout_pt, bias_pt = prepare_pt_data(dout, q, k, v, max_seqlen_q, bias=bias)

    s, p, o = fwd_pt(q_pt, k_pt, v_pt, bias=bias_pt)

    if bias is None:
        dq, dk, dv = torch.autograd.grad(o, (q_pt, k_pt, v_pt), dout_pt)
        return dq, dk, dv, None
    else:
        dq, dk, dv, dbias = torch.autograd.grad(o, (q_pt, k_pt, v_pt, bias_pt), dout_pt)
        return dq, dk, dv, dbias


def compute_lse(s):
    # import pdb; pdb.set_trace()
    # og_dtype = s.dtype
    # s = s.astype(np.float32)
    
    max_value_over_last_dim = np.max(s, axis=-1, keepdims=True)
    logit_sub_max_value = s - max_value_over_last_dim

    exp_x = np.exp(logit_sub_max_value)

    softmax_lse = np.max(s, axis=-1, keepdims=True) + np.log(np.sum(exp_x, axis=-1, keepdims=True))
    
    # softmax_lse = softmax_lse.astype(og_dtype)
    return softmax_lse


def check_fwd_kernel(has_bias=False):
    print ("==== check fwd kernel with np ====")
    if has_bias:
        s, p, o, _, _, _ = fwd(q_cpu, k_cpu, v_cpu, max_seqlen_k_, bias=bias_ref)
    else:
        s, p, o, _, _, _ = fwd(q_cpu, k_cpu, v_cpu, max_seqlen_k_)
    # print ("q * k = p'shape = {} p = {}".format(p.shape, p))

    # attn_output = np.loadtxt("attn_output.data", delimiter=" ")
    if has_bias:
        prefix = "has_bias"
        print ("has bias on, prefix is ", prefix)
    else:
        prefix = ""
    
    attn_output = np.genfromtxt("{}_attn_output.data".format(prefix), delimiter=" ", dtype=np.float32)
    attn_output = attn_output.reshape(batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim)
    attn_output = attn_output.reshape(batch_size * max_seqlen_k_, max_seqlen_k_, nheads, headdim)
    attn_output = attn_output.transpose(0, 2, 1, 3)
    print ("output max error: ", np.abs(o - attn_output).max())

    attn_lse = np.genfromtxt("{}_attn_lse.data".format(prefix), delimiter=" ", dtype=np.float32)
    max_seqlen_q_pad = ((max_seqlen_q_ + 16 - 1) // 16) * 16
    attn_lse = attn_lse.reshape(batch_size * max_seqlen_k_ , nheads, max_seqlen_q_pad)
    # print ("attn lse: ", attn_lse)
    attn_lse = attn_lse[:,:,:max_seqlen_q_]
    
    lse_ref = compute_lse(s)
    lse_ref = lse_ref.reshape(batch_size * max_seqlen_k_ , nheads, max_seqlen_q_)
    # print ("ref lse: ", lse_ref)

    import pdb; pdb.set_trace()

    print ("lse_ref shape = {}, attn_lse shape = {}".format(lse_ref.shape, attn_lse.shape))
    print ("lse max error: ", np.abs(lse_ref - attn_lse).max())


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


def check_bwd_kernel(has_bias=False):
    print ("==== check bwd kernel with np ====")
    if has_bias:
        dq, dk, dv, ds, dp, dbias = bwd(dout, q_cpu, k_cpu, v_cpu, max_seqlen_k_, bias=bias_ref)
    else:
        dq, dk, dv, ds, dp, dbias = bwd(dout, q_cpu, k_cpu, v_cpu, max_seqlen_k_)

    if has_bias:
        prefix = "has_bias"
        print ("has bias on, prefix is ", prefix)
    else:
        prefix = ""

    attn_dq = np.genfromtxt("{}_attn_dq.data".format(prefix), delimiter=" ", dtype=np.float32)
    attn_dk = np.genfromtxt("{}_attn_dk.data".format(prefix), delimiter=" ", dtype=np.float32)
    attn_dv = np.genfromtxt("{}_attn_dv.data".format(prefix), delimiter=" ", dtype=np.float32)

    attn_dq = attn_dq.reshape(batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim)
    attn_dk = attn_dk.reshape(batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim)
    attn_dv = attn_dv.reshape(batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim)

    attn_dq = attn_dq.reshape(batch_size * max_seqlen_k_, max_seqlen_k_, nheads, headdim)
    attn_dk = attn_dk.reshape(batch_size * max_seqlen_k_, max_seqlen_k_, nheads, headdim)
    attn_dv = attn_dv.reshape(batch_size * max_seqlen_k_, max_seqlen_k_, nheads, headdim)
    
    attn_dq = attn_dq.transpose(0, 2, 1, 3)
    attn_dk = attn_dk.transpose(0, 2, 1, 3)
    attn_dv = attn_dv.transpose(0, 2, 1, 3)

    assert (dq.shape == attn_dq.shape), "oh dq shape didn't match"
    assert (dk.shape == attn_dk.shape), "oh dk shape didn't match"
    assert (dv.shape == attn_dv.shape), "oh dv shape didn't match"

    print ("max error in dq: ", np.abs(attn_dq - dq).max(), )
    print ("max error in dk: ", np.abs(attn_dk - dk).max(), )
    print ("max error in dv: ", np.abs(attn_dv - dv).max(), )

    print ("same matrix check q: ", is_same_matrix(attn_dq, dq))
    print ("same matrix check k: ", is_same_matrix(attn_dk, dk))
    print ("same matrix check v: ", is_same_matrix(attn_dv, dv))


def check_bwd_np(has_bias=False):
    print ("==== check bwd np ====")
    if has_bias:
        dq, dk, dv, ds, dp, dbias = bwd(dout, q_cpu, k_cpu, v_cpu, max_seqlen_q=max_seqlen_q_, bias=bias_ref)
        dq_pt, dk_pt, dv_pt, dbias_pt = bwd_pt(dout, q_cpu, k_cpu, v_cpu, max_seqlen_q=max_seqlen_q_, bias=bias_ref)
    else:
        dq, dk, dv, ds, dp, dbias = bwd(dout, q_cpu, k_cpu, v_cpu, max_seqlen_q=max_seqlen_q_, bias=None)
        dq_pt, dk_pt, dv_pt, dbias_pt = bwd_pt(dout, q_cpu, k_cpu, v_cpu, max_seqlen_q=max_seqlen_q_, bias=None)

    assert (dq.shape == dq_pt.detach().cpu().numpy().shape), "oh dq shape didn't match"
    assert (dk.shape == dk_pt.detach().cpu().numpy().shape), "oh dk shape didn't match"
    assert (dv.shape == dv_pt.detach().cpu().numpy().shape), "oh dv shape didn't match"
    if has_bias:
        assert (dbias.shape == dbias_pt.detach().cpu().numpy().shape), "oh dbias shape didn't match"

    print ("max error in dq: ", np.abs( dq - dq_pt.detach().cpu().numpy() ).max())
    print ("max error in dk: ", np.abs( dk - dk_pt.detach().cpu().numpy() ).max())
    print ("max error in dv: ", np.abs( dv - dv_pt.detach().cpu().numpy() ).max())
    if has_bias:
        print ("max error in dbias: ", np.abs( dbias - dbias_pt.detach().cpu().numpy() ).max())    

    return 


def prepare_pt_data(dout, q, k, v, max_seqlen_q, bias=None):
    q_pt = torch.from_numpy(q.copy())
    k_pt = torch.from_numpy(k.copy())
    v_pt = torch.from_numpy(v.copy())

    batch_size = int(q.shape[0] / max_seqlen_q)
    head_num = q.shape[1]
    head_dim = q.shape[2]

    dout_pt = torch.from_numpy(dout.copy())
    dout_pt = dout_pt.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    dout_pt = dout_pt.permute(0, 2, 1, 3).cuda()

    q_pt = q_pt.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    k_pt = k_pt.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    v_pt = v_pt.reshape(batch_size, max_seqlen_q, head_num, head_dim)
    
    q_pt = q_pt.permute(0, 2, 1, 3).cuda()
    k_pt = k_pt.permute(0, 2, 1, 3).cuda()
    v_pt = v_pt.permute(0, 2, 1, 3).cuda()

    if bias is not None:
        bias_pt = torch.from_numpy(bias.copy()).cuda()
        bias_pt.requires_grad = True
    else:
        bias_pt = None

    q_pt.requires_grad = True
    k_pt.requires_grad = True
    v_pt.requires_grad = True

    return q_pt, k_pt, v_pt, dout_pt, bias_pt


def check_fwd_np(has_bias=False):
    print ("==== check fwd np ====")
    if has_bias:
        s, p, o, q, k, v = fwd(q_cpu, k_cpu, v_cpu, max_seqlen_q=max_seqlen_q_, bias=bias_ref)

        q_pt, k_pt, v_pt, dout_pt, bias_pt = prepare_pt_data(dout, q_cpu, k_cpu, v_cpu, max_seqlen_q=max_seqlen_q_, bias=bias_ref)
        s_pt, p_pt, o_pt = fwd_pt(q_pt, k_pt, v_pt, bias_pt)
    else:
        s, p, o, q, k, v = fwd(q_cpu, k_cpu, v_cpu, max_seqlen_q=max_seqlen_q_)

        q_pt, k_pt, v_pt, dout_pt, _ = prepare_pt_data(dout, q_cpu, k_cpu, v_cpu, max_seqlen_q=max_seqlen_q_)
        s_pt, p_pt, o_pt = fwd_pt(q_pt, k_pt, v_pt)

    def check_input(a, b):
        print ("max error in input: ", np.abs(a - b).max())
    
    check_input(q, q_pt.detach().cpu().numpy())
    check_input(k, q_pt.detach().cpu().numpy())
    check_input(v, q_pt.detach().cpu().numpy())

    assert (s.shape == s_pt.detach().cpu().numpy().shape), "oh s shape didn't match"
    assert (p.shape == p_pt.detach().cpu().numpy().shape), "oh p shape didn't match"
    assert (o.shape == o_pt.detach().cpu().numpy().shape), "oh o shape didn't match"

    print ("max error in s: ", np.abs( s - s_pt.detach().cpu().numpy() ).max())
    print ("max error in p: ", np.abs( p - p_pt.detach().cpu().numpy() ).max())
    print ("max error in o: ", np.abs( o - o_pt.detach().cpu().numpy() ).max())

    return 


def parse_softmax_load(filename):
    from parse import parse
    format_string = 'bwd softmax: threadIdx={}, l={}, mi={}, ki={}, ii={}, jj={}, elt={}'
    softmax_p = np.zeros([batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_], dtype=np.float16)

    with open(filename, "r") as f:
        for line in f.readlines():
            # print (line)
            if line.startswith("bwd softmax: "):
                # print (line.strip())
                result = parse(format_string, line.strip())
                # print (result)

                tidx_ = int(result[0])
                mi = int(result[2])
                ni = int(result[3])
                ii = int(result[4])
                jj = int(result[5])
                value = float(result[6])

                warp = tidx_ // 32
                lane = tidx_ % 32

                warp_n = (warp // 1)
                warp_m = (warp % 1)

                quad = lane // 4
                tid = (lane % 4) * 2

                row = warp_m * 16 + quad
                col = warp_n * 16 + tid

                current_row = mi * 16 + ii * 8 + row
                # current_col = ni * 64 + jj * 8 + col
                # current_col = ni * 64 + (jj & 2) * 4 + (jj & 1) + col
                current_col = ni * 64 + (jj & 2) * 8 + (jj & 1) + col
                # print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_row={}, value={}".format(
                #             warp, lane, quad, tid, current_row, current_col, value))
                if (current_row < 8 and current_col < 8):
                    print (line.strip())
                    print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_row={}, value={}".format(
                            warp, lane, quad, tid, current_row, current_col, value))
                    softmax_p[0, 0, current_row, current_col] = value

    return softmax_p


def check_softmax_p(softmax_data, has_bias=False):
    if has_bias:
        s, p, o, _, _, _ = fwd(q_cpu, k_cpu, v_cpu, max_seqlen_k_, bias=bias_ref)
    else:
        s, p, o, _, _, _ = fwd(q_cpu, k_cpu, v_cpu, max_seqlen_k_)
    # print ("q * k = p'shape = {} p = {}".format(p.shape, p))
    import pdb; pdb.set_trace()
    print ("max error in p: ", np.abs(p[0, 0, :, :] - softmax_data[0, 0, :, :]).max(), )
    print ("same matrix check p: ", is_same_matrix(p[0, 0, :, :], softmax_data[0, 0, :, :]))
    return


def parse_dsoftmax_load(filename):
    from parse import parse
    format_string = 'bwd dsoftmax: threadIdx={}, l={}, mi={}, ki={}, ii={}, jj={}, elt={}'
    dsoftmax = np.zeros([batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_], dtype=np.float16)

    with open(filename, "r") as f:
        for line in f.readlines():
            # print (line)
            if line.startswith("bwd dsoftmax: "):
                # print (line.strip())
                result = parse(format_string, line.strip())
                # print (result)

                tidx_ = int(result[0])
                mi = int(result[2])
                ni = int(result[3])
                ii = int(result[4])
                jj = int(result[5])
                value = float(result[6])

                warp = tidx_ // 32
                lane = tidx_ % 32

                warp_n = (warp // 1)
                warp_m = (warp % 1)

                quad = lane // 4
                tid = (lane % 4) * 2

                row = warp_m * 16 + quad
                col = warp_n * 16 + tid

                current_row = mi * 16 + ii * 8 + row
                # current_col = ni * 64 + jj * 8 + col
                # current_col = ni * 64 + (jj & 2) * 4 + (jj & 1) + col
                current_col = ni * 64 + (jj & 2) * 8 + (jj & 1) + col
                # print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_row={}, value={}".format(
                #             warp, lane, quad, tid, current_row, current_col, value))
                if (current_row < 8 and current_col < 8):
                    print (line.strip())
                    print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_row={}, value={}".format(
                            warp, lane, quad, tid, current_row, current_col, value))
                    dsoftmax[0, 0, current_row, current_col] = value

    return dsoftmax


def check_dsoftmax_p(softmax_data, has_bias=False):
    if has_bias:
        dq, dk, dv, ds, dp, dbias = bwd(dout, q_cpu, k_cpu, v_cpu, max_seqlen_k_, bias=bias_ref)
    else:
        dq, dk, dv, ds, dp, dbias = bwd(dout, q_cpu, k_cpu, v_cpu, max_seqlen_k_)

    # print ("q * k = p'shape = {} p = {}".format(p.shape, p))
    import pdb; pdb.set_trace()
    print ("max error in p: ", np.abs(ds[0, 0, :, :] - softmax_data[0, 0, :, :]).max(), )
    print ("same matrix check p: ", is_same_matrix(ds[0, 0, :, :], softmax_data[0, 0, :, :]))
    return


if __name__ == '__main__':
    # print ("====test without bias====")
    # has_bias = False
    # check_fwd_np(has_bias=has_bias)
    # check_bwd_np(has_bias=has_bias)
    # print ("====test without bias====")

    # print ("====test with bias====")
    # has_bias = True
    # check_fwd_np(has_bias=has_bias)
    # check_bwd_np(has_bias=has_bias)
    # print ("====test with bias====")

    # print ("====test kernel without bias====")
    # has_bias = False
    # check_fwd_kernel(has_bias=has_bias)
    # check_bwd_kernel(has_bias=has_bias)

    # print ("====test kernel with bias====")
    # has_bias = True
    # check_fwd_kernel(has_bias=has_bias)
    # check_bwd_kernel(has_bias=has_bias)

    # print ("====test bwd kernel softmax without bias====")
    # has_bias = False
    # softmax_data = parse_softmax_load("output.log")
    # check_softmax_p(softmax_data=softmax_data, has_bias=has_bias)

    # print ("====test bwd kernel softmax with bias====")
    # has_bias = True
    # softmax_data = parse_softmax_load("output.log")
    # check_softmax_p(softmax_data=softmax_data, has_bias=has_bias)

    # print ("====test bwd kernel softmax without bias====")
    # has_bias = False
    # dsoftmax_data = parse_dsoftmax_load("output.log")
    # check_dsoftmax_p(softmax_data=dsoftmax_data, has_bias=has_bias)

    print ("====test bwd kernel softmax with bias====")
    has_bias = True
    dsoftmax_data = parse_dsoftmax_load("output.log")
    check_dsoftmax_p(softmax_data=dsoftmax_data, has_bias=has_bias)
