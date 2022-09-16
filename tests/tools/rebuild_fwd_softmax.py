from parse import parse
import sys
import numpy as np

def is_same_matrix(pred, gt, abs_eps=0.01, relative_rps=0.03, verbose=False):
    diff = np.abs(pred - gt)

    cnt = 0
    for index, x in np.ndenumerate(diff):
        if x > abs_eps:
            if abs(gt[index]) < 1e-9:
                relative_diff = 100
            else:
                relative_diff = np.abs(x / gt[index])
            if relative_diff > relative_rps:
                cnt += 1
                if verbose:
                    print ("index={0}, diff={1}, pred={2}, true={3}, relative_diff={4}".format(
                        index, x, pred[index], gt[index], relative_diff))

    if cnt > 0:
        print ("not so match")
        return False
    else:
        return True

filename = "./output512.log"
if len(sys.argv) > 1:
    filename = sys.argv[1]

# Attnmask: threadIdx.x = 98, threadIdx.y = 0, mi = 0, ni = 0, ii = 0, jj = 2, value = 0.000000, softmax = 0.608030, l = 0, loop_step_idx=1, blockIdx.x = 0
format_string = 'Attnmask: threadIdx.x = {0}, threadIdx.y = {1}, mi = {2}, ni = {3}, ii = {4}, jj = {5}, value = {6}, softmax = {7}, l = {8}, loop_step_idx={9}, blockIdx.x = {10}'
batch_size = 1
nheads = 1
headdim = 16
bs_seq = 1
seq_q = 512
max_seqlen_q_ = seq_q 
max_seqlen_k_ = seq_q

Cta_tile_p_N = 256
Cta_tile_p_M = 16


def parse_fwd_softmax_load(filename):
    print ("processing... reconstruct from ", filename)
    softmax_data = np.zeros([batch_size * bs_seq, nheads, max_seqlen_q_, max_seqlen_k_], dtype=np.float16)
    with open(filename, "r") as f:
        for line in f.readlines():
            # print (line)
            if line.startswith("Attnmask: "):
                # print (line.strip())
                result = parse(format_string, line.strip())
                # print (result)

                tidx_ = int(result[0])
                mi = int(result[2])
                ni = int(result[3])
                ii = int(result[4])
                jj = int(result[5])
                value = float(result[6])
                softmax_elt = float(result[7])
                q_loop = int(result[8])
                k_loop = int(result[9])

                warp = tidx_ // 32
                lane = tidx_ % 32
                # thread per warp = 32
            
                warp_n = (warp // 1)
                warp_m = (warp % 1)
                # WARPS_M = 1

                quad = lane // 4
                tid = (lane % 4) * 2

                row = warp_m * 16 + quad
                col = warp_n * 16 + tid

                current_row = Cta_tile_p_M * q_loop + mi * 16 + ii * 8 + row

                current_col = k_loop * Cta_tile_p_N + ni * 64 + (jj & 2) * 4 + (jj & 1) + col
                # print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_row={}, value={}".format(
                #             warp, lane, quad, tid, current_row, current_col, value))
                if current_col > 510:
                    print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_row={}, value={}".format(
                            warp, lane, quad, tid, current_row, current_col, value))
                    print (line.strip())
                if (current_row < 16 and current_col < 512):
                    # print ("warp={}, lane={}, quad={}, tid={}, current_row={}, current_row={}, value={}".format(
                    #         warp, lane, quad, tid, current_row, current_col, value))
                    # print ("")
                    softmax_data[0, 0, current_row, current_col] = value

    return softmax_data


softmax_cpp = parse_fwd_softmax_load(filename)
softmax_python = parse_fwd_softmax_load("../" + filename)


print (is_same_matrix(softmax_cpp, softmax_python, verbose=True))

for i in range(16):
    print ("first part idx = {} softmax cpp = {}: ".format(i, softmax_cpp[0, 0, i, :256]))
    print ("first part idx = {} softmax python = {}: ".format(i, softmax_python[0, 0, i, :256]))

    print (np.allclose(softmax_cpp[0, 0, i, :256],softmax_python[0, 0, i, :256]))

    print ("second part idx = {} softmax cpp = {}: ".format(i, softmax_cpp[0, 0, i, 256:]))
    print ("second part idx = {} softmax python = {}: ".format(i, softmax_python[0, 0, i, 256:]))

    print (np.allclose(softmax_cpp[0, 0, i, 256:],softmax_python[0, 0, i, 256:]))
