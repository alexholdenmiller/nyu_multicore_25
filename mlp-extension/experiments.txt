input_size = 128
model_layers = 3
hidden_layer_features = 2048
output_size = 1
NUM_THREADS = 32
PRUNE = True

(venv) (base) [ahm9968@crunchy5 mlp-extension]$ python test.py
initializing models...
pruning model...
copying model weights and creating csr weights...

model sizes (number of 32-bit parameters)...
  ...size of base python model: 8652800
  ...size of base cpp model: 8652800
  ...size of base cpp model: 8652800
  ...size of sparse csr model: 92677
  ...size of sparse coo model: 129792

Confirming all models output the same values...
  ...is cpp primitives the same? True
  ...is cpp full forward the same? True
  ...is cpp sparse csr the same? True
  ...is cpp multithreaded csr the same? True
  ...is cpp sparse coo the same? True
  ...is cpp multithreaded coo the same? True

Running model simulations...
  ... Python   == Forward: 0.877 s
  ... C++ Prim == Forward: 0.757 s
  ... C++ Forw == Forward: 0.698 s
  ... C++ CSR  == Forward: 153.317 s
  ... C++ CSR mult  == Forward: 20.867 s
  ... C++ COO  == Forward: 133.312 s
  ... C++ COO mult  == Forward: 15.123 s

Speedup comparisons for quicker reading...
  ... C++ primitives version ran at 1.158x speed vs python
  ... C++ full forward version ran at 1.256x speed vs python
  ... C++ sparsified version ran at 0.006x speed vs python
  ... C++ full forward version ran at 1.085x speed vs cpp primitives
  ... C++ sparsified csr version ran at 0.005x speed vs cpp full forward
  ... C++ multithreaded csr version ran at 7.347x speed vs cpp sparsified
  ... C++ sparsified coo version ran at 0.005x speed vs cpp full forward
  ... C++ multithreaded coo version ran at 10.138x speed vs cpp sparsified

==========================================

input_size = 128
model_layers = 7
hidden_layer_features = 256
output_size = 1
NUM_THREADS = 32
PRUNE = True

(venv) (base) [ahm9968@crunchy5 mlp-extension]$ python test.py
initializing models...
pruning model...
copying model weights and creating csr weights...

model sizes (number of 32-bit parameters)...
  ...size of base python model: 426240
  ...size of base cpp model: 426240
  ...size of base cpp model: 426240
  ...size of sparse csr model: 6063
  ...size of sparse coo model: 6393

Confirming all models output the same values...
  ...is cpp primitives the same? True
  ...is cpp full forward the same? True
  ...is cpp sparse csr the same? True
  ...is cpp multithreaded csr the same? True
  ...is cpp sparse coo the same? True
  ...is cpp multithreaded coo the same? True

Running model simulations...
  ... Python   == Forward: 0.126 s
  ... C++ Prim == Forward: 0.085 s
  ... C++ Forw == Forward: 0.070 s
  ... C++ CSR  == Forward: 9.363 s
  ... C++ CSR mult  == Forward: 1.890 s
  ... C++ COO  == Forward: 6.817 s
  ... C++ COO mult  == Forward: 1.303 s

Speedup comparisons for quicker reading...
  ... C++ primitives version ran at 1.475x speed vs python
  ... C++ full forward version ran at 1.795x speed vs python
  ... C++ sparsified version ran at 0.013x speed vs python
  ... C++ full forward version ran at 1.217x speed vs cpp primitives
  ... C++ sparsified csr version ran at 0.007x speed vs cpp full forward
  ... C++ multithreaded csr version ran at 4.955x speed vs cpp sparsified
  ... C++ sparsified coo version ran at 0.010x speed vs cpp full forward
  ... C++ multithreaded coo version ran at 7.189x speed vs cpp sparsified

==========================================

input_size = 128
model_layers = 24
hidden_layer_features = 128
output_size = 1
NUM_THREADS = 32
PRUNE = True

(venv) (base) [ahm9968@crunchy5 mlp-extension]$ python test.py
initializing models...
pruning model...
copying model weights and creating csr weights...

model sizes (number of 32-bit parameters)...
  ...size of base python model: 393344
  ...size of base cpp model: 393344
  ...size of base cpp model: 393344
  ...size of sparse csr model: 7032
  ...size of sparse coo model: 5901

Confirming all models output the same values...
  ...is cpp primitives the same? True
  ...is cpp full forward the same? True
  ...is cpp sparse csr the same? True
  ...is cpp multithreaded csr the same? True
  ...is cpp sparse coo the same? True
  ...is cpp multithreaded coo the same? True

Running model simulations...
  ... Python   == Forward: 0.272 s
  ... C++ Prim == Forward: 0.165 s
  ... C++ Forw == Forward: 0.142 s
  ... C++ CSR  == Forward: 10.013 s
  ... C++ CSR mult  == Forward: 3.117 s
  ... C++ COO  == Forward: 6.184 s
  ... C++ COO mult  == Forward: 2.051 s

Speedup comparisons for quicker reading...
  ... C++ primitives version ran at 1.655x speed vs python
  ... C++ full forward version ran at 1.923x speed vs python
  ... C++ sparsified version ran at 0.027x speed vs python
  ... C++ full forward version ran at 1.162x speed vs cpp primitives
  ... C++ sparsified csr version ran at 0.014x speed vs cpp full forward
  ... C++ multithreaded csr version ran at 3.212x speed vs cpp sparsified
  ... C++ sparsified coo version ran at 0.023x speed vs cpp full forward
  ... C++ multithreaded coo version ran at 4.883x speed vs cpp sparsified