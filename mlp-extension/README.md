# Multicore Computing: Parallel PyTorch on Multicore
## Authors: Alexander Miller, Jiajing Chen, Ari Khaytser, Jinhao Pang
We target a reduction in memory usage while maintaining reasonable speed for linear layers in a neural netword by pruning a matrix down to a sparce version that maintains nearly exact results, and applying an optimized sparce-matrix-multiplication algorithm on the sparce matrix. 
This algorithm uses C++ extensions for pytorch and represent a sparse matrix using either COO and CSR format, and uses OpenMP to run a multi-threaded matrix multiplication algorithm on these formats. 

# setup

```shell
scp nyu_multicore_25.zip xxx@access.cims.nyu.edu:~/
ssh xxx@access.cims.nyu.edu
ssh crunchy5.cims.nyu.edu
mkdir tmp
mv nyu_multicore_25.zip tmp
cd tmp
unzip nyu_multicore_25
module load python-3.9
virtualenv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ninja
module load gcc-9.2 
python setup.py install
python test.py
```

# customizing the run
You can customize the following parameters within the test.py file main function:
```
input_size: dimention of the input to the networks
model_layers: number of layers (input layer + hidden layers + output layer) in the models
hidden_layer_features: dimension of each of the hidden layers
output_size: dimension of the model output
NUM_THREADS: number of threads to use for multithreaded models. 1 means single threaded/sequential
PRUNE: If True, matrixes will be pruned before being represented in sparse format. 
```