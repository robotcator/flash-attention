#include <cmath>
#include <fmha_api.h>
//#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void test_fwd_with_mask() {
    int batch_size = 1;
    int nheads = 1;
    int headdim = 16;
    int max_seqlen_q_ = 128; 
    int max_seqlen_k_ = 128;
    
    float softmax_scale = 0.1;
    
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
    
    at::Tensor q_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor k_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor v_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
  
    int cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_ * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < headdim; k ++) {
                q_cpu[i][j][k] = cnt * 0.001;
                k_cpu[i][j][k] = cnt * 0.001;
                v_cpu[i][j][k] = cnt * 0.001;
                cnt ++;
            }
	}
    }

    auto q = q_cpu.cuda();
    auto k = k_cpu.cuda();
    auto v = v_cpu.cuda();

    at::Tensor cu_seqlens_q_cpu = at::zeros({batch_size * max_seqlen_k_ + 1}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({batch_size * max_seqlen_k_ + 1}, at::kInt);
    
    for (int i = 0; i < batch_size * max_seqlen_k_ + 1; ++i) {
        cu_seqlens_q_cpu[i] = i * max_seqlen_q_;
        cu_seqlens_k_cpu[i] = i * max_seqlen_k_;
    }
    
    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();
    
    at::Tensor attn_mask = at::ones({batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_}, at::kHalf).cuda();

    cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < max_seqlen_q_; k ++) {
                for (int l = 0; l < max_seqlen_k_; l ++) {
                    attn_mask[i][j][k][l] = cnt * 0.001;
                    cnt ++;
                }
            }
	    }
    }
    
    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_bias;

    // std::cout << "attn bias" << attn_bias << std::endl;

    std::vector<at::Tensor> ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            cu_seqlens_q,  // b + 1
            cu_seqlens_k,  // b + 1
            max_seqlen_q_,
            max_seqlen_k_,
            0.0,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            gen_,
            attn_mask,
            attn_bias
	    );

    std::cout << "Ret vec size is " << ret.size();
    for (int i = 0; i < ret.size(); i ++) {
        ret[i].cpu();
        std::cout << ret[i] << std::endl;
    }
}


void test_fwd_with_mask_mini() {
    int batch_size = 1;
    int nheads = 1;
    int headdim = 16;
    int max_seqlen_q_ = 2; 
    int max_seqlen_k_ = 2;
    
    float softmax_scale = 0.1;
    
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
    
    at::Tensor q_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor k_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor v_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
  
    int cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_ * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < headdim; k ++) {
                q_cpu[i][j][k] = cnt * 0.001;
                k_cpu[i][j][k] = cnt * 0.001;
                v_cpu[i][j][k] = cnt * 0.001;
                cnt ++;
            }
	    }
    }

    auto q = q_cpu.cuda();
    auto k = k_cpu.cuda();
    auto v = v_cpu.cuda();

    at::Tensor cu_seqlens_q_cpu = at::zeros({batch_size * max_seqlen_k_ + 1}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({batch_size * max_seqlen_k_ + 1}, at::kInt);
    
    for (int i = 0; i < batch_size * max_seqlen_k_ + 1; ++i) {
        cu_seqlens_q_cpu[i] = i * max_seqlen_q_;
        cu_seqlens_k_cpu[i] = i * max_seqlen_k_;
    }
    
    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();
    
    at::Tensor attn_mask_cpu = at::zeros({batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_}, at::kHalf);

    cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < max_seqlen_q_; k ++) {
                for (int l = 0; l < max_seqlen_k_; l ++) {  
                    // if (l == 0) attn_mask[i][j][k][l] = -INFINITY;
                    if (l == 0) attn_mask_cpu[i][j][k][l] = -3e4;
                    else attn_mask_cpu[i][j][k][l] = 0;

                    attn_mask_cpu[i][j][k][l] = -3e4;
                    printf("i=%d, j=%d, k=%d, l=%d attn_mask=%f\n", i, j, k, l, attn_mask_cpu[i][j][k][l]);
                }
            }
	    }
    }

    auto attn_mask = attn_mask_cpu.cuda();

    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_bias;

    // std::cout << "attn bias: " << attn_bias << std::endl; 

    std::vector<at::Tensor> ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            cu_seqlens_q,  // b + 1
            cu_seqlens_k,  // b + 1
            max_seqlen_q_,
            max_seqlen_k_,
            0.0,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            gen_,
            attn_mask,
            attn_bias
	    );

    // ret: std::vector<at::Tensor> result = {o, softmax_lse};
    // [bs * seq * seq, head, head_dim]
    // [1 * 2 * 2, 1, 16]
    std::cout << "Ret vec size is " << ret.size();
    for (int i = 0; i < ret.size(); i ++) {
        ret[i].cpu();
        std::cout << ret[i] << std::endl;
    }
}


void test_fwd_with_bias_mini() {
    int batch_size = 1;
    int nheads = 1;
    int headdim = 16;
    int max_seqlen_q_ = 2; 
    int max_seqlen_k_ = 2;
    
    float softmax_scale = 0.1;
    
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
    
    at::Tensor q_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor k_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor v_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
  
    int cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_ * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < headdim; k ++) {
                q_cpu[i][j][k] = cnt * 0.001;
                k_cpu[i][j][k] = cnt * 0.001;
                v_cpu[i][j][k] = cnt * 0.001;
                cnt ++;
            }
	    }
    }

    auto q = q_cpu.cuda();
    auto k = k_cpu.cuda();
    auto v = v_cpu.cuda();

    at::Tensor cu_seqlens_q_cpu = at::zeros({batch_size * max_seqlen_k_ + 1}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({batch_size * max_seqlen_k_ + 1}, at::kInt);
    
    for (int i = 0; i < batch_size * max_seqlen_k_ + 1; ++i) {
        cu_seqlens_q_cpu[i] = i * max_seqlen_q_;
        cu_seqlens_k_cpu[i] = i * max_seqlen_k_;
    }
    
    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();
    
    at::Tensor attn_bias_cpu = at::zeros({batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_}, at::kHalf);

    cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < max_seqlen_q_; k ++) {
                for (int l = 0; l < max_seqlen_k_; l ++) {  
                    // if (l == 0) attn_mask[i][j][k][l] = -INFINITY;
                    if (l == 0) attn_bias_cpu[i][j][k][l] = -3e4;
                    else attn_bias_cpu[i][j][k][l] = 0;

                    attn_bias_cpu[i][j][k][l] = 100;
                    printf("i=%d, j=%d, k=%d, l=%d attn_bias=%f\n", i, j, k, l, attn_bias_cpu[i][j][k][l]);
                    // std::cout << "i=" << i << ", j=" << j << ", k=" << k << ", l" 
                    //     << l << << ", attn_bias=" << attn_bias_cpu[i][j][k][l] << std::endl;
                }
            }
	    }
    }

    auto attn_bias = attn_bias_cpu.cuda();

    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_mask;

    // std::cout << attn_mask << std::endl;

    std::vector<at::Tensor> ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            cu_seqlens_q,  // b + 1
            cu_seqlens_k,  // b + 1
            max_seqlen_q_,
            max_seqlen_k_,
            0.0,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            gen_,
            attn_mask,
            attn_bias
	    );

    // ret: std::vector<at::Tensor> result = {o, softmax_lse};
    // [bs * seq * seq, head, head_dim]
    // [1 * 2 * 2, 1, 16]
    std::cout << "Ret vec size is " << ret.size();
    for (int i = 0; i < ret.size(); i ++) {
        ret[i].cpu();
        std::cout << ret[i] << std::endl;
    }
}


void test_fwd_with_bias() {
    int batch_size = 1;
    int nheads = 1;
    int headdim = 16;
    int max_seqlen_q_ = 128; 
    int max_seqlen_k_ = 128;
    
    float softmax_scale = 0.1;
    
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
    
    at::Tensor q_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor k_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
    at::Tensor v_cpu = at::zeros({batch_size * max_seqlen_k_ * max_seqlen_k_, nheads, headdim}, at::kHalf);
  
    int cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_ * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < headdim; k ++) {
                q_cpu[i][j][k] = cnt * 0.001;
                k_cpu[i][j][k] = cnt * 0.001;
                v_cpu[i][j][k] = cnt * 0.001;
                cnt ++;
            }
	    }
    }

    auto q = q_cpu.cuda();
    auto k = k_cpu.cuda();
    auto v = v_cpu.cuda();

    at::Tensor cu_seqlens_q_cpu = at::zeros({batch_size * max_seqlen_k_ + 1}, at::kInt);
    at::Tensor cu_seqlens_k_cpu = at::zeros({batch_size * max_seqlen_k_ + 1}, at::kInt);
    
    for (int i = 0; i < batch_size * max_seqlen_k_ + 1; ++i) {
        cu_seqlens_q_cpu[i] = i * max_seqlen_q_;
        cu_seqlens_k_cpu[i] = i * max_seqlen_k_;
    }
    
    auto cu_seqlens_q = cu_seqlens_q_cpu.cuda();
    auto cu_seqlens_k = cu_seqlens_k_cpu.cuda();
    
    at::Tensor attn_bias_cpu = at::zeros({batch_size * max_seqlen_k_, nheads, max_seqlen_q_, max_seqlen_k_}, at::kHalf);

    cnt = 0;
    for (int i = 0; i < batch_size * max_seqlen_k_; i ++) {
    	for (int j = 0; j < nheads; j ++) {
            for (int k = 0; k < max_seqlen_q_; k ++) {
                for (int l = 0; l < max_seqlen_k_; l ++) {  
                    // if (l == 0) attn_mask[i][j][k][l] = -INFINITY;
                    // if (l == 0) attn_bias_cpu[i][j][k][l] = -3e4;
                    // else attn_bias_cpu[i][j][k][l] = 0;
                    
                    attn_bias_cpu[i][j][k][l] = 0;
                    // attn_bias_cpu[i][j][k][l] = cnt * 0.001;
                    cnt ++;
                    // printf("i=%d, j=%d, k=%d, l=%d attn_bias=%f\n", i, j, k, l, attn_bias_cpu[i][j][k][l]);
                    // std::cout << "i=" << i << ", j=" << j << ", k=" << k << ", l" 
                    //     << l << << ", attn_bias=" << attn_bias_cpu[i][j][k][l] << std::endl;
                }
            }
	    }
    }

    auto attn_bias = attn_bias_cpu.cuda();

    c10::optional<at::Generator> gen_;
    c10::optional<at::Tensor> attn_mask;

    // std::cout << attn_mask << std::endl;

    std::vector<at::Tensor> ret = mha_fwd(
            q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            cu_seqlens_q,  // b + 1
            cu_seqlens_k,  // b + 1
            max_seqlen_q_,
            max_seqlen_k_,
            0.0,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            gen_,
            attn_mask,
            attn_bias
	    );

    // ret: std::vector<at::Tensor> result = {o, softmax_lse};
    // [bs * seq * seq, head, head_dim]
    // [1 * 2 * 2, 1, 16]
    std::cout << "Ret vec size is " << ret.size();
    for (int i = 0; i < ret.size(); i ++) {
        ret[i].cpu();
        std::cout << ret[i] << std::endl;
    }
}

int main(){
    // test_fwd();
    // test_fwd_with_bias_mini();
    test_fwd_with_bias();
    return 0;
}
