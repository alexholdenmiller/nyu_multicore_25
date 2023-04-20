# Multicore Computing: Parallel PyTorch on Multicore
## Authors: Alexander Miller, Jiajing Chen, Ari Khaytser, Jinhao Pang

We target a reduction in memory usage while maintaining reasonable speed for linear layers in a neural netword by pruning a matrix down to a sparce version that maintains nearly exact results, and applying an optimized sparce-matrix-multiplication algorithm on the sparce matrix. 

This algorithm uses C++ extensions for pytorch and represent a sparse matrix using either COO and CSR format, and uses OpenMP to run a multi-threaded matrix multiplication algorithm on these formats. 

## Setup
Clone this repo
Make sure you have pytorch installed on your pytorch enviroment, and OpenMP for C++.

### IF running on CIMS Servers
Snappy1 has Intel CPU if we try to use Intel-extension-for-PyTorch.
Run the following commands

```shell
//ssh to CIMS accounts
//ssh xxx@snappy1.cims.nyu.edu
//clone the project to the CIMS
cd nyu_multicore_25
module load python-3.9
virtualenv venv
source venv/bin/activate
pip install torch ninja --index-url https://download.pytorch.org/whl/cpu
module load gcc-9.2 
cd lltm-extension/ && python setup.py install
python test.py
```

### Running the MLP evaluation
Create the mlp-extension, and then run the python tests
```
cd mlp-extension
python setup.py install
python test.py
```

You can customize the following parameters within the test.py file main function:
```
input_size: dimention of the input to the networks
model_layers: number of layers (input layer + hidden layers + output layer) in the models
hidden_layer_features: dimension of each of the hidden layers
output_size: dimension of the model output
NUM_THREADS: number of threads to use for multithreaded models. 1 means single threaded/sequential
PRUNE: If True, matrixes will be pruned before being represented in sparse format. 
```
