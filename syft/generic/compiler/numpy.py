import os

from syft.generic import tensor

from syft.generic.compiler.util import get_complier_resources
from syft.generic.compiler.util import write_ast_to_file
from syft.generic.compiler.transformers.naming import SyftToFrameworkNameTransformer

# generic tensor python files to compile into torch tensors
compile_modules = list(set(dir(tensor)) - tensor.do_not_compile_modules)

def compile_module_to_numpy(module_name):

    # Get Tensor AST and target folder
    tree, target_folder = get_complier_resources(tensor, module_name, "Numpy")

    # Replace occurrences of "Syft"with "Torch"
    tree = SyftToFrameworkNameTransformer("Numpy").visit(tree)

    write_ast_to_file(tree, target_folder, module_name)


def compile_numpy(modules=None):

    if(modules is None):
        modules = compile_modules

    # for module_name in compile_modules[0:1]:
    for module_name in compile_modules:
        compile_module_to_numpy(module_name)
