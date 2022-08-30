import numpy as np

batch_size = 1
nheads = 1
headdim = 16
seq = 8
max_seqlen_q_ = seq 
max_seqlen_k_ = seq


q_cpu = np.zeros((batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim), dtype=np.float16)
k_cpu = np.zeros((batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim), dtype=np.float16)
v_cpu = np.zeros((batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim), dtype=np.float16)
  
cnt = 0
for i in range(batch_size * max_seqlen_k_ * max_seqlen_k_):
    for j in range(nheads):
        for k in range(headdim):
            q_cpu[i][j][k] = cnt * 0.001
            k_cpu[i][j][k] = cnt * 0.001
            v_cpu[i][j][k] = cnt * 0.001
            cnt += 1

bias_ref = np.zeros([batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_], dtype=np.float32)
cnt = 0

for i in range(batch_size * max_seqlen_k_):
    for j in range(nheads):
        for k in range(max_seqlen_q_):
            for l in range(max_seqlen_k_):
                bias_ref[i][j][k][l] = cnt * 0.001
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

    print ("data q block 0 = {}".format(q[0, 0, :, :]))

    s = np.matmul(q, k.transpose(0,1,3,2))
    
    if bias is not None:
        s = s + bias

    p = softmax(s)

    o = np.matmul(p, v)
    
    o = o.transpose(0,2,1,3).reshape(batch_size * max_seqlen_q, head_num, head_dim)

    return s, p, o


def compute_lse(s):
    max_value_over_last_dim = np.max(s, axis=-1, keepdims=True)
    logit_sub_max_value = s - max_value_over_last_dim

    exp_x = np.exp(logit_sub_max_value)

    softmax_lse = np.max(s, axis=-1, keepdims=True) + np.log(np.sum(exp_x, axis=-1, keepdims=True))
    return softmax_lse



if __name__ == '__main__':
    
    has_bias = True

    if has_bias:
        s, p, o = fwd(q_cpu, k_cpu, v_cpu, max_seqlen_k_, bias=bias_ref)
    else:
        s, p, o = fwd(q_cpu, k_cpu, v_cpu, max_seqlen_k_)
    # print ("q * k = p'shape = {} p = {}".format(p.shape, p))

    # attn_output = np.loadtxt("attn_output.data", delimiter=" ")
    attn_output = np.genfromtxt("attn_output.data", delimiter=" ", dtype=np.float16)
    attn_output = attn_output.reshape(batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim)
    # attn_output = attn_output.reshape(batch_size * max_seqlen_k_, max_seqlen_k_, nheads, headdim)
    print ("output max error: ", np.abs(o - attn_output).max())

    attn_lse = np.genfromtxt("attn_lse.data", delimiter=" ", dtype=np.float32)
    max_seqlen_q_pad = ((max_seqlen_q_ + 16 - 1) // 16) * 16
    attn_lse = attn_lse.reshape(batch_size * max_seqlen_k_ , nheads, max_seqlen_q_pad)
    print ("attn lse: ", attn_lse)
    attn_lse = attn_lse[:,:,:max_seqlen_q_]
    
    lse_ref = compute_lse(s)
    lse_ref = lse_ref.reshape(batch_size * max_seqlen_k_ , nheads, max_seqlen_q_)
    print ("ref lse: ", lse_ref)

    print ("lse_ref shape = {}, attn_lse shape = {}".format(lse_ref.shape, attn_lse.shape))
    print ("lse max error: ", np.abs(lse_ref - attn_lse).max())


