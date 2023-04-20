import math
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune_lib

# Our module!
import mlp_cpp_lib


def prune(parameters, amt):
    prune_lib.global_unstructured(
        parameters,
        pruning_method=prune_lib.L1Unstructured,
        amount=amt,
    )
    for module, name in parameters:
        prune_lib.remove(module, name)


class MLPpy(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, n_hidden):
        super().__init__()

        if n_hidden < 1:
            raise RuntimeError("n_hidden must be at least one")

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.lin_in = nn.Linear(input_size, hidden_dim, False)
        self.lin_out = nn.Linear(hidden_dim, output_size, False)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, False) for _ in range(n_hidden - 1)])
        self.num_hidden_layers = n_hidden - 1

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    
    def __len__(self):
        total = 0
        for layer in [self.lin_in, self.lin_out] + [layer for layer in self.layers]:
            total += layer.weight.nelement()
        return total

    def forward(self, x):
        x = self.relu(self.lin_in(x))
        for layer in self.layers:
            x = self.relu(layer(x))

        return self.flatten(self.lin_out(x))
    
    def prune(self, amt=0.9):
        param_list = [(self.lin_in, 'weight'), (self.lin_out, 'weight')] + [(layer, 'weight') for layer in self.layers]
        prune(param_list, amt)
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
    
    def __len__(self):
        total = 0
        for tensor in [self.lin_in, self.lin_out, self.layers]:
            total += tensor.nelement()
        return total

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
    
    def __len__(self):
        total = 0
        for layer in [self.sparse_lin_in, self.sparse_lin_out] + self.sparse_layers:
            for tensor in layer[:3]:
                total += tensor.nelement() if self.sparse_alg == 'csr' else len(tensor)

        return total

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