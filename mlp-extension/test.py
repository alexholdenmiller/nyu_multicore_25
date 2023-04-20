
import time
import torch
from models import *


if __name__ == "__main__":
    input_size = 128
    model_layers = 7
    hidden_layer_features = 256
    output_size = 1
    NUM_THREADS = 32
    PRUNE = True

    X = torch.randn(1, input_size, device='cpu', requires_grad=False)  # fix batch size to one

    print("initializing models...")
    mlp_py = MLPpy(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_p = MLPcpp_primitives(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_f = MLPcpp_forward(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_s_csr = MLPcpp_sparse(input_size, hidden_layer_features, output_size, model_layers)
    mlp_cpp_s_coo = MLPcpp_sparse(input_size, hidden_layer_features, output_size, model_layers, 'coo')

    if PRUNE:
        # https://arxiv.org/abs/1803.03635
        # some models can maintain over 90% accuracy with 99.5% pruning
        # the bigger the model, mostly likely the more true this is
        print("pruning model...")
        PAPER_MAX_PRUNE_RATE = 0.995
        mlp_py.prune(PAPER_MAX_PRUNE_RATE)

    print("copying model weights and creating csr weights...")
    # set models to same underlying weights
    mlp_cpp_p.load_state_dict(mlp_py.state_dict())
    mlp_cpp_f.load_parameters(mlp_py.state_dict())
    mlp_cpp_s_csr.load_state_dict(mlp_py.state_dict())
    mlp_cpp_s_csr.sparsify()  # need to sparsify with new weights
    mlp_cpp_s_coo.load_state_dict(mlp_py.state_dict())
    mlp_cpp_s_coo.sparsify()  # need to sparsify with new weights

    print()
    print("model sizes (number of 32-bit parameters)...")
    print("  ...size of base python model:", len(mlp_py))
    print("  ...size of base cpp model:", len(mlp_cpp_p))
    print("  ...size of base cpp model:", len(mlp_cpp_f))
    print("  ...size of sparse csr model:", len(mlp_cpp_s_csr))
    print("  ...size of sparse coo model:", len(mlp_cpp_s_coo))

    # confirm the model parameters and computation are the same
    print()
    print("Confirming all models output the same values...")
    o1 = mlp_py(X)
    o2 = mlp_cpp_p(X)
    print("  ...is cpp primitives the same?", torch.allclose(o1, o2))
    o3 = mlp_cpp_f(X)
    print("  ...is cpp full forward the same?", torch.allclose(o1, o3))
    o4 = mlp_cpp_s_csr(X)
    print("  ...is cpp sparse csr the same?", torch.allclose(o1, o4))
    o5 = mlp_cpp_s_csr(X, NUM_THREADS)
    print("  ...is cpp multithreaded csr the same?", torch.allclose(o1, o5))
    o6 = mlp_cpp_s_coo(X)
    print("  ...is cpp sparse coo the same?", torch.allclose(o1, o6))
    o7 = mlp_cpp_s_coo(X, NUM_THREADS)
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
    N = 100
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
