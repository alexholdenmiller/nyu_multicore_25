import math
import time
import torch

from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from baseline_model import MLP as MLPpy, prune

# Our module!
import mlp_cpp_lib

device = 'cpu'


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
        self.lin_in = torch.Tensor(hidden_dim, input_size)
        self.lin_out = torch.Tensor(output_size, hidden_dim)
        self.layers = torch.Tensor(n_hidden - 1, hidden_dim, hidden_dim)
        self.num_hidden_layers = n_hidden - 1

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in (self.lin_in, self.lin_out, self.layers):
            weight.uniform_(-stdv, +stdv)

    def load_parameters(self, state_dict):
        self.lin_in = state_dict['lin_in.weight']
        self.lin_out = state_dict['lin_out.weight']
        self.layers = torch.stack([state_dict[layer_name] for layer_name in filter(lambda l: not (l == 'lin_in.weight' or l == 'lin_out.weight'), state_dict.keys())])

    def forward(self, x):
        return mlp_cpp_lib.mlp_forward(x.squeeze(), self.lin_in, self.layers, self.lin_out, self.num_hidden_layers)

    def prune(self, amt=0.9):
        prune(self, amt)


class MLPcpp_sparse(MLPpy):
    """
    uses sparse CSR format instead of dense computations
    """
    def __init__(self, input_size, hidden_dim, output_size, n_hidden, sparse_alg = 'csr'):
        super().__init__(input_size, hidden_dim, output_size, n_hidden)
        self.sparse_alg = sparse_alg
        

    def sparsify(self):
        if self.sparse_alg == 'csr':
            self.sparse_lin_in = mlp_cpp_lib.to_csr(self.lin_in.weight)
            self.sparse_lin_out = mlp_cpp_lib.to_csr(self.lin_out.weight)
            self.sparse_layers = [mlp_cpp_lib.to_csr(self.layers[i].weight) for i in range(len(self.layers))]
        elif self.sparse_alg == 'coo':
            self.sparse_lin_in = mlp_cpp_lib.to_coo(self.lin_in.weight)
            self.sparse_lin_out = mlp_cpp_lib.to_coo(self.lin_out.weight)
            self.sparse_layers = [mlp_cpp_lib.to_coo(self.layers[i].weight) for i in range(len(self.layers))]

    def forward(self, x, num_threads=1):
        if self.sparse_alg == 'csr':
            if num_threads == 1:
                return mlp_cpp_lib.mlp_sparse_forward_csr(
                    x.squeeze(),
                    self.sparse_lin_in,
                    self.sparse_layers,
                    self.sparse_lin_out,
                    self.num_hidden_layers
                )
            else:
                return mlp_cpp_lib.mlp_sparse_forward_mt_csr(
                    x.squeeze(),
                    self.sparse_lin_in,
                    self.sparse_layers,
                    self.sparse_lin_out,
                    self.num_hidden_layers,
                    num_threads,
                )
        elif self.sparse_alg == 'coo':
            if num_threads == 1:
                return mlp_cpp_lib.mlp_sparse_forward_coo(
                    x.squeeze(),
                    self.sparse_lin_in,
                    self.sparse_layers,
                    self.sparse_lin_out,
                    self.num_hidden_layers
                )
            else:
                return mlp_cpp_lib.mlp_sparse_forward_mt_coo(
                    x.squeeze(),
                    self.sparse_lin_in,
                    self.sparse_layers,
                    self.sparse_lin_out,
                    self.num_hidden_layers,
                    num_threads,
                )


if __name__ == "__main__":
    input_size = 2048
    model_layers = 5
    hidden_layer_features = 512
    output_size = 8
    NUM_THREADS = 16
    PRUNE = True

    X = torch.randn(1, input_size, device=device, requires_grad=False)  # fix batch size to one

    print("initializing models...")
    mlp_py = MLPpy(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_p = MLPcpp_primitives(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_f = MLPcpp_forward(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_s_csr = MLPcpp_sparse(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_s_coo = MLPcpp_sparse(input_size, hidden_layer_features, output_size, model_layers, 'coo')

    print("pruning model...")
    if PRUNE:
        # https://arxiv.org/abs/1803.03635
        # some models can maintain over 90% accuracy with 99.5% pruning
        # the bigger the model, mostly likely the more true this is
        PAPER_MAX_PRUNE_RATE = 0.995
        mlp_py.prune(PAPER_MAX_PRUNE_RATE)

    print("copying model weights and creating csr weights...")
    # set models to same underlying weights
    mlp_cpp_p.load_state_dict(mlp_py.state_dict())
    mlp_cpp_f.load_parameters(mlp_py.state_dict())
    mlp_cpp_s_csr.load_state_dict(mlp_py.state_dict())
    mlp_cpp_s_csr.sparsify()  # need to sparsify with new weights
    mlp_cpp_s_coo.load_state_dict(mlp_py.state_dict())
    mlp_cpp_s_coo.sparsify()

    # confirm the model parameters and computation are the same
    print()
    print("Confirming all models output the same values...")
    o1 = mlp_py(X)
    o2 = mlp_cpp_p(X)
    o3 = mlp_cpp_f(X)
    o4 = mlp_cpp_s_csr(X)
    o5 = mlp_cpp_s_csr(X, NUM_THREADS)
    o6 = mlp_cpp_s_coo(X)
    o7 = mlp_cpp_s_coo(X, NUM_THREADS)
    print("  ...is cpp primitives the same?", torch.allclose(o1, o2))
    print("  ...is cpp full forward the same?", torch.allclose(o1, o3))
    print("  ...is cpp sparse csr the same?", torch.allclose(o1, o4))
    print("  ...is cpp multithreaded csr the same?", torch.allclose(o1, o5))
    print("  ...is cpp sparse coo the same?", torch.allclose(o1, o6))
    print("  ...is cpp multithreaded coo the same?", torch.allclose(o1, o7))
    

    forward_py = 0
    forward_cpp_p = 0
    forward_cpp_f = 0
    forward_cpp_s = 0
    forward_cpp_mt = 0
    forward_cpp_s_coo = 0
    forward_cpp_mt_coo = 0


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
        _output = mlp_cpp_s_csr(X)
        forward_cpp_s += time.time() - start
    
    def cpp_mt_compute():
        global forward_cpp_mt
        start = time.time()
        _output = mlp_cpp_s_csr(X, NUM_THREADS)
        forward_cpp_mt += time.time() - start

    def cpp_s_coo_compute():
        global forward_cpp_s_coo
        start = time.time()
        _output = mlp_cpp_s_coo(X)
        forward_cpp_s_coo += time.time() - start
    
    def cpp_mt_coo_compute():
        global forward_cpp_mt_coo
        start = time.time()
        _output = mlp_cpp_s_coo(X, NUM_THREADS)
        forward_cpp_mt_coo += time.time() - start


    print()
    print("Running model simulations...")
    N = 50
    with torch.no_grad():
        for _ in range(N):
            py_compute()
            cpp_p_compute()
            cpp_f_compute()
            cpp_s_compute()
            cpp_mt_compute()
            cpp_s_coo_compute()
            cpp_mt_coo_compute()


    print(f'  ... Python   == Forward: {forward_py:.3f} s')
    print(f'  ... C++ Prim == Forward: {forward_cpp_p:.3f} s')
    print(f'  ... C++ Forw == Forward: {forward_cpp_f:.3f} s')
    print(f'  ... C++ CSR  == Forward: {forward_cpp_s:.3f} s')
    print(f'  ... C++ CSR mult  == Forward: {forward_cpp_mt:.3f} s')
    print(f'  ... C++ COO  == Forward: {forward_cpp_s_coo:.3f} s')
    print(f'  ... C++ COO mult  == Forward: {forward_cpp_mt_coo:.3f} s')

    print()
    print('Speedup comparisons for quicker reading...')
    print(f'  ... C++ primitives version ran at {forward_py / forward_cpp_p:.3f}x speed vs python')
    print(f'  ... C++ full forward version ran at {forward_py / forward_cpp_f:.3f}x speed vs python')
    print(f'  ... C++ sparsified version ran at {forward_py / forward_cpp_s:.3f}x speed vs python')
    print(f'  ... C++ full forward version ran at {forward_cpp_p / forward_cpp_f:.3f}x speed vs cpp primitives')
    print(f'  ... C++ sparsified csr version ran at {forward_cpp_f / forward_cpp_s:.3f}x speed vs cpp full forward')
    print(f'  ... C++ multithreaded csr version ran at {forward_cpp_s / forward_cpp_mt:.3f}x speed vs cpp sparsified')
    print(f'  ... C++ sparsified coo version ran at {forward_cpp_f / forward_cpp_s_coo:.3f}x speed vs cpp full forward')
    print(f'  ... C++ multithreaded coo version ran at {forward_cpp_s / forward_cpp_mt_coo:.3f}x speed vs cpp sparsified')
