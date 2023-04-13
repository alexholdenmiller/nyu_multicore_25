# nyu_multicore_25
final project for group 25 of NYU's spring 2023 multicore class


# setup
clone pytorch from https://github.com/pytorch/pytorch
clone this repo, it doesn't have to be within the pytorch repo

if you're using visual studio code, modify your C/C++ Configurations to add the path to the pytorch directory, e.g.
C:/Users/tuzon/Documents/GitHub/pytorch/**
I still got a warning:
`cannot open source file "c10/macros/cmake_macros.h"`
but couldn't figure out why this was the case (and it didn't prevent the code from working)



snappy1

```shell
//ssh to cims accounts
//ssh xxx@snappy1.cims.nyu.edu
//upload the project to the cims
cd nyu_multicore_25
module load python-3.9
virtualenv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
module load gcc-9.2
cd lltm-extension/ && python setup.py install
python test.py
```


# testing the lltm tutorial
https://pytorch.org/tutorials/advanced/cpp_extension.html

in the lltm-extension directory, run `python setup.py install` in order to compile the c++ extension

then run `python test.py` to evaluate the difference between implementing the model in c++ vs python