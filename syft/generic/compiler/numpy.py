import os

from syft.generic import tensor
from syft.generic.compiler.generic import compile_module

# generic tensor python files to compile into numpy tensors
compile_modules = list(set(dir(tensor)) - tensor.do_not_compile_modules)

def compile_numpy(modules=None):

    if(modules is None):
        modules = compile_modules

    for module_name in compile_modules:
        compile_module(module_name, framework="Numpy")
