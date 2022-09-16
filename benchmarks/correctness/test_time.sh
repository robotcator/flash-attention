python benchmarks/correctness/check_speed_forward.py --has_mask_bias=false 2>&1 |tee no_mask_bias_test.txt
python benchmarks/correctness/check_speed_forward.py --has_mask_bias=true 2>&1 |tee has_mask_bias_test.txt

python benchmarks/correctness/check_speed_backward.py --has_mask_bias=false 2>&1 |tee no_mask_bias_train.txt
python benchmarks/correctness/check_speed_backward.py --has_mask_bias=true 2>&1 |tee has_mask_bias_train.txt