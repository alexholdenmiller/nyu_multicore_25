#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>

std::vector<at::Tensor> base_forward(
        int64_t input_size, int64_t n_hidden, int64_t output_size, int64_t model_layers) {
    torch::nn::ReLU relu;
    torch::nn::Linear linear1, linear2, linear3;
    int64_t layers;
    register_module("relu", relu);
    register_module("linear1", linear1);
    register_module("linear2", linear2);
    register_module("linear3", linear3);

    x = relu(linear1(x));
    for (int i = 0; i < layers - 1; ++i) {
        x = relu(linear2(x));
    }
    return linear3(x);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &base_forward, "Base forward");
}