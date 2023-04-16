#include <torch/extension.h>

class Base : public torch::nn::Module {
public:
    Base(int64_t input_size, int64_t n_hidden, int64_t output_size, int64_t model_layers);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Flatten flatten;
    torch::nn::ReLU relu;
    torch::nn::Linear linear1, linear2, linear3;
    int64_t layers;
};

Base::Base(int64_t input_size, int64_t n_hidden, int64_t output_size, int64_t model_layers)
        : flatten(torch::nn::Flatten()), relu(torch::nn::ReLU()),
          linear1(torch::nn::Linear(input_size, n_hidden)),
          linear2(torch::nn::Linear(n_hidden, n_hidden)),
          linear3(torch::nn::Linear(n_hidden, output_size)),
          layers(model_layers) {
    register_module("flatten", flatten);
    register_module("relu", relu);
    register_module("linear1", linear1);
    register_module("linear2", linear2);
    register_module("linear3", linear3);
}

torch::Tensor Base::forward(torch::Tensor x) {
    x = flatten(x);
    x = relu(linear1(x));
    for (int i = 0; i < layers - 1; i++) {
        x = relu(linear2(x));
    }
    return linear3(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("forward", &forward, "Base forward");
}