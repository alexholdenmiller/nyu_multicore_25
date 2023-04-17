from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'], extra_compile_args=['-fopenmp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
