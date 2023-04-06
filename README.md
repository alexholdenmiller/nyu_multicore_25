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

# testing the lltm example
in the lltm-extension directory, run `python setup.py install` in order to compile the c++ extension
then run `python test.py` to evaluate the difference between implementing the model in c++ vs python