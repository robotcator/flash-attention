#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <string>
#include <fstream>


torch::Tensor load_tensor(std::string filename) {
	std::cout << filename << std::endl;
    std::ifstream sfile(filename.c_str());

	torch::Tensor tensor2;
  	torch::load(tensor2, sfile);

	// std::cout << tensor2 << std::endl;
	return tensor2;
}

int main(){

	std::string label = "";
	std::string tensor_name = "input_mask";
	std::string sfile_name = label + "_" + tensor_name + ".pt";

    // std::ifstream sfile(sfile_name.c_str());
	// torch::Tensor tensor2;
  	// torch::load(tensor2, sfile);

	torch::Tensor tensor_c = load_tensor(sfile_name);
	std::cout << tensor_c << std::endl;


	std::string python_file_name = "../" + label + "_" + tensor_name + ".pt";
	torch::Tensor tensor_python = load_tensor(python_file_name);
	std::cout << tensor_python << std::endl;


	// int batch_size = 2;
	// int num_heads = 4;
	// int max_seqlen_q = 8;
	// int max_seqlen_k = 8;

	// auto bias = torch::ones({1, num_heads, max_seqlen_q, max_seqlen_k});
	// auto ds = torch::ones({batch_size, num_heads, max_seqlen_q, max_seqlen_k});
	// // batch_size, 1, num_heads, max_seqlen_q, max_seqlen_k
	
	
	// auto shape = bias.sizes();
	// // auto newshape = std::vector<int64_t>(shape);
	// // newshape.insert(newshape.begin(), -1);
	// // std::cout << newshape << std::endl;

	// auto dbias = ds.reshape({-1, shape[0], shape[1], shape[2], shape[3] }).sum({0});

	// std::cout << dbias.sizes() << std::endl;
	return 0;
}


