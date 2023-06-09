#include <torch/extension.h>

#include <iostream>
#include <vector>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

std::vector<at::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  at::Tensor input_gate;
  at::Tensor output_gate;
  at::Tensor candidate_cell;

  # pragma omp parallel sections num_threads(3)
  {
    # pragma omp section
    input_gate = torch::sigmoid(gates[0]);

    # pragma omp section
    output_gate = torch::sigmoid(gates[1]);

    # pragma omp section
    candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);
  }

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  const auto state_size = grad_h.size(1);
  
  at::Tensor d_weights;
  at::Tensor d_bias;
  at::Tensor d_X;
  at::Tensor d_old_h;
  at::Tensor d_input;
  // this isn't good: num_threads(1) slows it down a TON, num_threads(>1) crashes
  // maybe can't use pragmas in sections with matrix multiplies? may clash with pytorch multithreading
  // # pragma omp parallel sections
  {
    // # pragma omp section
    d_weights = d_gates.t().mm(X);

    // # pragma omp section
    d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

    // # pragma omp section
    d_X = d_gates.mm(weights);
    
    // # pragma omp section
    d_old_h = d_X.slice(/*dim=*/1, 0, state_size);

    // # pragma omp section
    d_input = d_X.slice(/*dim=*/1, state_size);
  }

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}