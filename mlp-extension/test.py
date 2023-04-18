import math
import time
import torch

from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from baseline_model import MLP as MLPpy, prune

# Our module!
import mlp_cpp_lib


class MLPcpp_primitives(MLPpy):
    """
    copies pure-python init, uses own matrix multiply / relu calls
    runs at basically the same speed as python version
    """

    def forward(self, x):
        x = x.squeeze()
        x = mlp_cpp_lib.mm_t_relu(x, self.lin_in.weight)
        for layer in self.layers:
            x = mlp_cpp_lib.mm_t_relu(x, layer.weight)

        return mlp_cpp_lib.mm_t(x, self.lin_out.weight)


class MLPcpp_forward(nn.Module):
    """
    this version places the entire forward call into cpp
    expected this to be clearly faster, but was about the same

    since this implements the whole function in cpp it should be a bit faster
    maybe would work better for different matrix sizes / shapes?
    """

    def __init__(self, input_size, hidden_dim, output_size, n_hidden):
        super().__init__()

        if n_hidden < 1:
            raise RuntimeError("n_hidden must be at least one")

        self.hidden_dim = hidden_dim
        self.lin_in = nn.Parameter(torch.Tensor(hidden_dim, input_size))
        self.lin_out = nn.Parameter(torch.Tensor(output_size, hidden_dim))
        self.layers = nn.Parameter(torch.Tensor(n_hidden - 1, hidden_dim, hidden_dim))
        self.num_hidden_layers = n_hidden - 1

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in (self.lin_in, self.lin_out, self.layers):
            weight.data.uniform_(-stdv, +stdv)

    def load_parameters(self, state_dict):
        self.lin_in.data = state_dict['lin_in.weight']
        self.lin_out.data = state_dict['lin_out.weight']
        self.layers.data = torch.stack([state_dict[layer_name] for layer_name in filter(lambda l: not (l == 'lin_in.weight' or l == 'lin_out.weight'), state_dict.keys())])

    def forward(self, x):
        return mlp_cpp_lib.mlp_forward(x.squeeze(), self.lin_in, self.layers, self.lin_out, self.num_hidden_layers)

    def prune(self, amt=0.9):
        prune(self, amt)


class MLPcpp_sparse(MLPcpp_forward):
    """
    copies pure-python init, uses own matrix multiply / relu calls
    runs at basically the same speed as python version
    """

    def forward_next(self, x):
        return mlp_cpp_lib.mlp_sparse_forward(x.squeeze(), self.lin_in, self.layers, self.lin_out,
                                              self.num_hidden_layers)


if __name__ == "__main__":
    input_size = 65536
    model_layers = 32
    hidden_layer_features = 1024
    output_size = 32
    PRUNE = True

    X = torch.randn(1, input_size)  # fix batch size to one

    mlp_py = MLPpy(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_p = MLPcpp_primitives(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_f = MLPcpp_forward(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_s = MLPcpp_sparse(input_size, hidden_layer_features, output_size, model_layers)

    #Copy weights from mlp_py to mlp_cpp_p

    if PRUNE:
        mlp_py.prune()

    # set models to same underlying weights
    mlp_cpp_p.load_state_dict(mlp_py.state_dict())
    mlp_cpp_f.load_parameters(mlp_py.state_dict())

    # confirm the model parameters and computation are the same
    o1 = mlp_py(X)
    o2 = mlp_cpp_p(X)
    o3 = mlp_cpp_f(X)
    print("Are parameter values of model1 and model2 the same?",
        not False in torch.eq(o1, o2))
    print("Are parameter values of model1 and model3 the same?",
        not False in torch.eq(o1, o3))
    print("Are parameter values of model2 and model3 the same?",
        not False in torch.eq(o2, o3))
    

    forward_py = 0
    forward_cpp_p = 0
    forward_cpp_f = 0
    forward_cpp_s = 0


    def cpp_p_compute():
        global forward_cpp_p
        start = time.time()
        _output = mlp_cpp_p(X)
        forward_cpp_p += time.time() - start


    def cpp_f_compute():
        global forward_cpp_f
        start = time.time()
        _output = mlp_cpp_f(X)
        forward_cpp_f += time.time() - start


    def py_compute():
        global forward_py
        start = time.time()
        _output = mlp_py(X)
        forward_py += time.time() - start


    def cpp_s_compute():
        global forward_cpp_s
        start = time.time()
        _output = mlp_cpp_s(X)
        forward_cpp_s += time.time() - start


    N = 100
    with torch.no_grad():
        for _ in range(N):
            py_compute()
            cpp_p_compute()
            cpp_f_compute()
            cpp_s_compute()

    print(f'Python   == Forward: {forward_py:.3f} s')
    print(f'C++ Prim == Forward: {forward_cpp_p:.3f} s')
    print(f'C++ Forw == Forward: {forward_cpp_f:.3f} s')
    print(f'C++ CSR  == Forward: {forward_cpp_s:.3f} s')
    print(f'C++ primitives version ran at {forward_py / forward_cpp_p:.3f}x speed vs python')
    print(f'C++ full forward version ran at {forward_py / forward_cpp_f:.3f}x speed vs python')
    print(f'C++ full forward version ran at {forward_cpp_p / forward_cpp_f:.3f}x speed vs cpp primitives')
