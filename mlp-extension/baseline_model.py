import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


# function to count number of parameters
def get_n_params(model):
    count = 0
    for param in list(model.parameters()):
        n = 1
        for s in list(param.size()):
            n = n * s
        count += n
    return count


def prune(parameters, amt):
    prune.global_unstructured(
        parameters,
        pruning_method=prune.L1Unstructured,
        amount=amt,
    )


class MLP(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, n_hidden):
        super().__init__()

        if n_hidden < 1:
            raise RuntimeError("n_hidden must be at least one")

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

    def forward(self, x):
        x = self.relu(self.lin_in(x))
        for layer in self.layers:
            x = self.relu(layer(x))

        return self.flatten(self.lin_out(x))
    
    def prune(self, amt=0.9):
        return prune(self.parameters(), amt)


if __name__ == "__main__":
    input_size = 65536
    model_layers = 10
    hidden_layer_features = 2048
    output_size = 8

    X = torch.randn(input_size)

    model = MLP(input_size, hidden_layer_features, output_size, model_layers)

    total_time = 0
    N = 1000000
    for _ in range(N):
        start = time.time()
        total_time += time.time() - start

    print('Total Model Params: {}'.format(get_n_params(model)))
    print('Python Time == Forward: {:.3f} s'.format(total_time))
