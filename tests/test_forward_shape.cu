#include <cmath>
#include <fmha_api.h>
//#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdlib>


void dump_tensor(const std::string &tensor_name, at::Tensor &tensor, const std::string &label) {
    std::string file_name = label + "_" + tensor_name + ".data";
    std::ofstream file(file_name.c_str());
    // file << tensor_name << std::endl;
    // file << tensor << std::endl;
    std::cout << "tensor_name size: " << tensor_name << " " <<  tensor.sizes() << std::endl;
    auto flatten_tensor = tensor.flatten();
    auto size = flatten_tensor.numel();

    for (int i = 0; i < size; i ++) {
        file << flatten_tensor[i].item() << " ";
        // file << flatten_tensor[i] << " ";
    }
    file << std::endl;
}

void test_fwd_with_mask(int seq, int has_mask=1) {
    int batch_size = 1;
    int nheads = 1;
    int headdim = 16;
    // int seq = 400;

    int bs_seq = 1;
    int max_seqlen_q_ = seq; 
    int max_seqlen_k_ = seq;
    
    float softmax_scale = 1;
    
    bool zero_tensors = false;
    bool is_causal = false;
    bool return_softmax = false;

    // q -> [bs * seq, head, head_dim]
    // q -> [1 * 128, 1, 16]
    // block q -> [128, 16]

    // k -> [bs * seq, head, head_dim]
    // k -> [1 * 128, 1, 16]
    // block k -> [128, 16]

    // v -> [bs * seq, head, head_dim]
    // v -> [1 * 128, 1, 16]
    // block k -> [128, 16]
    
    at::Tensor q_cpu = at::zeros({batch_size * bs_seq * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor k_cpu = at::zeros({batch_size * bs_seq * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor v_cpu = at::zeros({batch_size * bs_seq * max_seqlen_k_, nheads, headdim}, at::kHalf);
  
    int cnt = 0;
    for (int i = 0; i < batch_size * bs_seq * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < headdim; k ++) {
                q_cpu[i][j][k] = (cnt % 10000) * 0.001;
                k_cpu[i][j][k] = (cnt % 10000) * 0.001;
                v_cpu[i][j][k] = (cnt % 10000) * 0.001;
                cnt ++;
            }
	    }
    }

    auto q = q_cpu.cuda();
    auto k = k_cpu.cuda();
    auto v = v_cpu.cuda();

    at::Tensor cu_seqlens_q_cpu = at::zeros({batch_size * bs_seq + 1}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({batch_size * bs_seq + 1}, at::kInt);
    
    for (int i = 0; i < batch_size * bs_seq + 1; ++i) {
        cu_seqlens_q_cpu[i] = i * max_seqlen_q_;
        cu_seqlens_k_cpu[i] = i * max_seqlen_k_;
    }
    
    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();
    
    at::Tensor attn_mask = at::ones({batch_size * bs_seq, nheads, max_seqlen_q_, max_seqlen_k_}, at::kHalf) * -1;

    // cnt = 0;
    // for (int i = 0; i < batch_size * max_seqlen_k_; i ++) {
    // 	for (int j = 0; j < nheads; j ++) {
    //         for (int k = 0; k < max_seqlen_q_; k ++) {
    //             for (int l = 0; l < max_seqlen_k_; l ++) {
    //                 // attn_mask[i][j][k][l] = cnt * 0.001;
    //                 // cnt ++;
    //                 if (l % 2 == 0) {
    //                     attn_mask[i][j][k][l] = 0;
    //                 }
    //                 cnt ++;
    //             }
    //         }
	//     }
    // }

    for (int i = 0; i < batch_size * bs_seq; i ++) {
    	for (int j = 0; j < 1; j ++) {
            for (int k = 0; k < 1; k ++) {
                for (int l = 0; l < max_seqlen_k_; l ++) {
                    if (l % 2 == 0) {
                        attn_mask[i][0][0][l] = 0;
                    }
                }
            }
	    }
    }

    for (int i = 0; i < batch_size * bs_seq; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < max_seqlen_q_; k ++) {
                attn_mask[i][j][k] = attn_mask[i][0][0]; 
            }
	    }
    }

    attn_mask = attn_mask.cuda();

    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_bias;

    std::vector<at::Tensor> ret;

    ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            cu_seqlens_q,  // b + 1
            cu_seqlens_k,  // b + 1
            max_seqlen_q_,
            max_seqlen_k_,
            0.0,
            softmax_scale,
            zero_tensors, // False
            is_causal, // False
            return_softmax, // False
            gen_, 
            attn_mask,
            attn_bias
        );
        dump_tensor("attn_output", ret[0], "has_mask");
        dump_tensor("attn_lse", ret[1], "has_mask");
    

    return ;
    // std::cout << "Ret vec size is " << ret.size();
    // for (int i = 0; i < ret.size(); i ++) {
    //     ret[i].cpu();
    //     std::cout << ret[i] << std::endl;
    // }

    at::Tensor dout_cpu = at::ones({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);

    cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_ * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < headdim; k ++) {
                dout_cpu[i][j][k] = cnt * 0.001;
                cnt ++;
            }
	    }
    }
    
    at::Tensor dq_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor dk_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor dv_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);

    auto dout = dout_cpu.cuda();
    auto dq = dq_cpu.cuda();
    auto dk = dk_cpu.cuda();
    auto dv = dv_cpu.cuda();
    std::vector<at::Tensor> bwd_ret;

    if (has_mask) {
        bwd_ret = mha_bwd(
            dout,
            q,
            k,
            v,
            ret[0],
            ret[1],
            dq,
            dk,
            dv,
            cu_seqlens_q,  // b + 1
            cu_seqlens_k,  // b + 1
            max_seqlen_q_,
            max_seqlen_k_,
            0.0,
            softmax_scale,
            zero_tensors,
            is_causal,
            gen_,
            attn_mask,
            attn_bias
        );
        dump_tensor("attn_dq", dq, "has_mask");
        dump_tensor("attn_dk", dk, "has_mask");
        dump_tensor("attn_dv", dv, "has_mask");
        // dump_tensor("attn_ds", bwd_ret[5], "has_mask");
    }else{
        bwd_ret = mha_bwd(
            dout,
            q,
            k, 
            v, 
            ret[0],
            ret[1],
            dq,
            dk,
            dv,
            cu_seqlens_q,  // b + 1
            cu_seqlens_k,  // b + 1
            max_seqlen_q_,
            max_seqlen_k_,
            0.0,
            softmax_scale,
            zero_tensors,
            is_causal,
            gen_,
            attn_bias,
            attn_bias
            // placeholder
        );
        dump_tensor("attn_dq", dq, "");
        dump_tensor("attn_dk", dk, "");
        dump_tensor("attn_dv", dv, "");
    }
}

int main(int argc, char** argv){
    
    if ( argc >= 2 ) {
        std::cout << "argv: " << argv[1] << std::endl;
        int seq = atoi(argv[1]);

        test_fwd_with_mask(seq);

    }
    return 0;
}
