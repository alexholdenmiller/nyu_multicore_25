#include <torch/extension.h>
#include <iostream>
#include <vector>

// matrix multiply with a transpose
at::Tensor mm_t(
    torch::Tensor input,
    torch::Tensor weights
) {
    return input.mm(weights.t());
}

// matrix multiply with transpose and relu
at::Tensor mm_t_relu(
    torch::Tensor input,
    torch::Tensor weights
) {
    return torch::relu(input.mm(weights.t()));
}

// full mlp forward given input weights, output weights, and 3D matrix of hidden layer weights
at::Tensor mlp_forward(
    torch::Tensor input,
    torch::Tensor in_weights,
    torch::Tensor hidden_weights,
    torch::Tensor output_weights,
    int64_t num_layers
) {
    torch::Tensor state = torch::relu(input.mm(in_weights));
    for(int i = 0; i < num_layers; i++) {
        state = torch::relu(state.mm(hidden_weights[i]));
    }
    return state.mm(output_weights);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm_t", &mm_t, "matrix multiply with transpose");
  m.def("mm_t_relu", &mm_t_relu, "matrix multiply with transpose and relu");
  m.def("mlp_forward", &mlp_forward, "mlp forward");
}