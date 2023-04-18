#include <torch/extension.h>
#include <iostream>
#include <vector>

// matrix multiply with a transpose
at::Tensor mv(
    torch::Tensor input,
    torch::Tensor weights
) {
    return weights.mv(input);
}

// matrix_vector multiply with transpose and relu
at::Tensor mv_relu(
    torch::Tensor input,
    torch::Tensor weights
) {
    return torch::relu(weights.mv(input));
}

// full mlp forward given input weights, output weights, and 3D matrix of hidden layer weights
at::Tensor mlp_forward(
    torch::Tensor input,
    torch::Tensor in_weights,
    torch::Tensor hidden_weights,
    torch::Tensor output_weights,
    int64_t num_layers
) {
    torch::Tensor state = torch::relu(in_weights.mv(input));
    for(int i = 0; i < num_layers; i++) {
        state = torch::relu(hidden_weights[i].mv(state));
    }
    return output_weights.mv(state);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm_t", &mv, "matrix-vector multiply with transpose");
  m.def("mm_t_relu", &mv_relu, "matrix-vector multiply with transpose and relu");
  m.def("mlp_forward", &mlp_forward, "mlp forward");
}