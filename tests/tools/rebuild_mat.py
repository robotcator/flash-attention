from parse import parse
import sys
import numpy as np

filename = "./output.log"
if len(sys.argv) > 1:
    filename = sys.argv[1]

# AttnBias: threadIdx.x = 0, threadIdx.y = 0, mi = 0, ni = 0, ii = 0, jj = 0, value = 0.000000
format_string = 'AttnBias: threadIdx.x = {}, threadIdx.y = {}, mi = {}, ni = {}, ii = {}, jj = {}, value = {}, ldx = {}, blockIdx.x = {}'
batch_size = 1
nheads = 1
headdim = 16
seq = 8
seq_q = 8
max_seqlen_q_ = seq_q 
max_seqlen_k_ = seq_q


mask_ref = np.zeros([batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_], dtype=np.float16)
cnt = 0

for i in range(batch_size * max_seqlen_k_):
    for j in range(nheads):
        for k in range(max_seqlen_q_):
            for l in range(max_seqlen_k_):
                mask_ref[i][j][k][l] = cnt * 0.001
                cnt += 1

# mask = np.zeros([1, 1, max_seqlen_q_, max_seqlen_k_], dtype=np.float32)
# batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_
mask = np.zeros([batch_size * max_seqlen_k_, nheads, 16, 128], dtype=np.float16)


def parse_bias_load(filename):
    with open(filename, "r") as f:
        for line in f.readlines():
            # print (line)
            if line.startswith("AttnBias:"):
                # print (line.strip())
                
                result = parse(format_string, line.strip())
                print (result)
                # import pdb; pdb.set_trace()
                # if result[0] == 0:
                #     print (result[0], result[1], result[2], result[3], result[4], result[5], result[6])
                tidx_ = int(result[0])
                mi = int(result[2])
                ni = int(result[3])
                ii = int(result[4])
                jj = int(result[5])
                value = float(result[6])
                block_idx = int(result[8])

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
                current_col = ni * 64 + (jj & 2) * 4 + (jj & 1) + col
                
                # if (current_row < 8 and current_col < 8):
                #     print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_row={}, value={}".format(
                #             warp, lane, quad, tid, current_row, current_col, value))
                #     mask[0, 0, current_row, current_col] = value
                print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_col={}, value={}".format(
                            warp, lane, quad, tid, current_row, current_col, value))
                mask[block_idx, 0, current_row, current_col] = value


def check(mask, mask_ref, block_idx=0):
    flag = True
    bs, nheads, max_seqlen_q_, max_seqlen_k_ = mask_ref.shape
    for i in range(max_seqlen_q_):
        for j in range(max_seqlen_k_):
            if (abs(mask[0, 0, i, j] - mask_ref[block_idx, 0, i, j]) > 1e-3):
                print ("False in block_idx = {}, i = {}, j = {}, mask = {}, mask_ref = {}".format(block_idx, 
                    i, j, mask[0, 0, i, j] - mask_ref[block_idx, 0, i, j]))
                flag = False
    return flag

parse_bias_load(filename)

# block_idx = 1
# print (check(mask, mask_ref, block_idx))