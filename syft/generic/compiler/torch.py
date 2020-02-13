import os

from syft.generic import tensor
from syft.generic.compiler.generic import compile_module

# generic tensor python files to compile into torch tensors
compile_modules = list(set(dir(tensor)) - tensor.do_not_compile_modules)

def compile_torch(modules=None):

    if(modules is None):
        modules = compile_modules

    for module_name in compile_modules:
        compile_module(module_name, framework="Torch")
