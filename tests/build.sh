#!/bin/bash
# csrc_path=../csrc/flash_attn
# csrc_path=/workspace/openfold/single_test/flash_attn/flash-attention_v2/csrc/flash_attn
csrc_path=../csrc/flash_attn
src_file=
src_file+=test_forward.cu
src_file+=" ${csrc_path}/fmha_api.cpp"
src_file+=" ${csrc_path}/src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu"
src_file+=" ${csrc_path}/src/fmha_block_fprop_fp16_kernel.sm80.cu"
src_file+=" ${csrc_path}/src/fmha_dgrad_fp16_kernel_loop.sm80.cu"
src_file+=" ${csrc_path}/src/fmha_fprop_fp16_kernel.sm80.cu"

echo ${src_file}

echo ${csrc_path}/
echo ${csrc_path}/src
echo ${csrc_path}/cutlass/include

# nvcc -o test ${src_file} \
/usr/local/cuda-11.3/bin/nvcc -v -o test ${src_file} \
    --compiler-options='-Wl\,--no-as-needed' \
    -lc10 -ltorch -ltorch_cpu -lcudart -lc10_cuda -ltorch_cuda -ltorch_cuda_cu -ltorch_cuda_cpp \
    -I ./ \
    -I ${csrc_path} \
    -I ${csrc_path}/src \
    -I ${csrc_path}/cutlass/include \
    -I /opt/conda/lib/python3.7/site-packages/torch/include \
    -I /opt/conda/lib/python3.7/site-packages/torch/include/torch/csrc/api/include \
    -I /opt/conda/lib/python3.7/site-packages/torch/include/TH \
    -I /opt/conda/lib/python3.7/site-packages/torch/include/THC \
    -I /opt/conda/include \
    -I /opt/conda/include/python3.7m \
    -L /opt/conda/lib/python3.7/site-packages/torch/lib/ \
    -L /usr/local/cuda-11.3/lib64/ \
    -L /opt/conda/lib64/ \
    -L /opt/conda/lib/ \
    -g -G \
    -t 4 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DDEBUG_PRINT \
    -DDEBUG_USING_NVCC \
    -gencode arch=compute_80,code=sm_80 \
    -U__CUDA_NO_HALF_OPERATORS__ \
    -U__CUDA_NO_HALF_CONVERSIONS__ \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    --use_fast_math 


