from parse import parse
import sys
import numpy as np

filename = "./output.log"
if len(sys.argv) > 1:
    filename = sys.argv[1]

# bwd softmax: threadIdx=195, l=0, mi=0, ki=1, ii=3, jj=0, elt=0.000000
format_string = 'bwd softmax: threadIdx={}, l={}, mi={}, ki={}, ii={}, jj={}, elt={}'
batch_size = 1
nheads = 1
headdim = 16
seq = 8
seq_q = 8
max_seqlen_q_ = seq_q 
max_seqlen_k_ = seq_q


d_softmax = np.zeros([batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_], dtype=np.float16)

def parse_dsoftmax_load(filename):
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
                    d_softmax[0, 0, current_row, current_col] = value


parse_dsoftmax_load(filename)
print ("output block 0 d_softmax: ", d_softmax[0, 0, :, :])
