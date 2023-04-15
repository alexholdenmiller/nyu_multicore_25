import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# function to count number of parameters
def get_n_params(model):
    count=0
    for param in list(model.parameters()):
        n=1
        for s in list(param.size()):
            n = n*s
        count += n
    return count

class Base(nn.Module):
    def __init__(self, input_size, n_hidden, output_size, model_layers):
        super(Base, self).__init__()
        
        #define our functions
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, output_size)
        self.layers = model_layers

    def forward(self, x):
        #create the architechture for the model
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        for _ in range(self.layers - 1):
            x = self.relu(self.linear2(x))
        
        return self.linear3(x)

if __name__ == "__main__":
    batch_size = 16
    input_size = 65536
    model_layers = 10
    hidden_layer_features = 2048
    output_size= 8

    X = torch.randn(batch_size, input_size)

    model = Base(input_size, hidden_layer_features, output_size, model_layers)

    total_time = 0
    N = 1000000
    for _ in range(N):
        start = time.time()
        total_time += time.time() - start

    print('Total Model Params: {}'.format(get_n_params(model)))
    print('Python Time == Forward: {:.3f} s'.format(total_time))