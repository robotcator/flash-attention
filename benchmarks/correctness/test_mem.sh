python benchmarks/correctness/benchmark_memory.py --has_mask_bias=true --eval=false 2>&1 |tee has_mask_bias_train.txt
python benchmarks/correctness/benchmark_memory.py --has_mask_bias=false --eval=false 2>&1 |tee no_mask_bias_train.txt

python benchmarks/correctness/benchmark_memory.py --has_mask_bias=true --eval=true 2>&1 |tee has_mask_bias_test.txt
python benchmarks/correctness/benchmark_memory.py --has_mask_bias=false --eval=true 2>&1 |tee no_mask_bias_test.txt