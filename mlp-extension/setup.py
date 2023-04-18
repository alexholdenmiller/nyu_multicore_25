from setuptools import setup, Extension
from torch.utils import cpp_extension


# original codes
setup(name='cpp_extension',
      ext_modules=[
            cpp_extension.CppExtension(
                'mlp_cpp_lib',
                ['mlp_lib.cpp'],
                extra_compile_args=['-fopenmp']
            )
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
