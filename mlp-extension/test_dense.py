import math
import time
import torch

from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from baseline_model import MLP as MLPpy

# Our module!
import mlp_cpp_lib


class MLPcpp_primitives(MLPpy):
    """
    copies pure-python init, uses own matrix multiply / relu calls
    runs at basically the same speed as python version
    """

    def forward(self, x):
        x = mlp_cpp_lib.mm_t_relu(x, self.lin_in.weight)
        for layer in self.layers:
            x = mlp_cpp_lib.mm_t_relu(x, layer.weight)

        return mlp_cpp_lib.mm_t(x, self.lin_out.weight)


class MLPcpp_forward(nn.Module):
    """
    this version places the entire forward call into cpp
    expected this to be clearly faster, but was about the same speed

    improvements:
    - calling into cpp once vs once per Linear layer
    - no transposes
    - no nn.Parameter overhead

    maybe would work better for different matrix sizes / shapes
    """

    def __init__(self, input_size, hidden_dim, output_size, n_hidden):
        super().__init__()

        if n_hidden < 1:
            raise RuntimeError("n_hidden must be at least one")

        self.hidden_dim = hidden_dim
        self.lin_in = torch.Tensor(input_size, hidden_dim)
        self.lin_out = torch.Tensor(hidden_dim, output_size)
        self.layers = torch.Tensor(n_hidden - 1, hidden_dim, hidden_dim)
        self.num_hidden_layers = n_hidden - 1

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in (self.lin_in, self.lin_out, self.layers):
            weight.uniform_(-stdv, +stdv)

    def forward(self, x):
        return mlp_cpp_lib.mlp_forward(x, self.lin_in, self.layers, self.lin_out, self.num_hidden_layers)


if __name__ == "__main__":
    input_size = 65536
    model_layers = 32
    hidden_layer_features = 1024
    output_size = 32

    X = torch.randn(1, input_size)

    mlp_py = MLPpy(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_p = MLPcpp_primitives(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_f = MLPcpp_forward(input_size, hidden_layer_features, output_size, model_layers)

    forward_py = 0
    forward_cpp_p = 0
    forward_cpp_f = 0


    def cpp_p_compute():
        global forward_cpp_p
        start = time.time()
        _output = mlp_cpp_p(X)
        forward_cpp_p += time.time() - start
        return _output


    def cpp_f_compute():
        global forward_cpp_f
        start = time.time()
        # I am wondering if this line should be mlp_cpp_f(x) instead of _output = mlp_cpp_p(X)
        _output = mlp_cpp_f(X)
        forward_cpp_f += time.time() - start
        return _output


    def py_compute():
        global forward_py
        start = time.time()
        _output = mlp_py(X)
        forward_py += time.time() - start
        return _output


    N = 100
    with torch.no_grad():
        for _ in range(N):
            o1 = py_compute()
            o2 = cpp_p_compute()
            o3 = cpp_f_compute()

    mlp_py.prune()
    # check if the final parameters are same
    print("Are parameter values of model1 and model2 the same?",
          torch.equal(o1, o2))
    print("Are parameter values of model1 and model3 the same?",
          torch.equal(o1, o3))
    print("Are parameter values of model2 and model3 the same?",
          torch.equal(o2, o3))

    print(f'Python   == Forward: {forward_py:.3f} s')
    print(f'C++ Prim == Forward: {forward_cpp_p:.3f} s')
    print(f'C++ Forw == Forward: {forward_cpp_f:.3f} s')
    print(f'C++ primitives version ran at {forward_py / forward_cpp_p:.3f}x speed vs python')
    print(f'C++ full forward version ran at {forward_py / forward_cpp_f:.3f}x speed vs python')
    print(f'C++ full forward version ran at {forward_cpp_p / forward_cpp_f:.3f}x speed vs cpp primitives')
