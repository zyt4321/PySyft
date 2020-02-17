from syft.generic import tensor


from syft.generic.compiler.util import write_output_to_file
from syft.generic.compiler.util import mod2ast
from syft.generic.compiler.util import ast2src
from syft.generic.compiler.util import ROOT_DIR
from syft.generic.compiler.transformers.naming import SyftToFrameworkNameTransformer
from syft.generic.compiler.transformers.framework_decorator import (
    FrameworkSpecificMethodDecoratorFilter,
)
from syft.generic.compiler.transformers.handle_transformer import (
    DecoratorAwareFrameworkHandleTransformer,
)
from syft.generic.compiler.transformers.handle_transformer import DecoratorAwareFrameworkFunctionTransfomer
from syft.generic.compiler.overrides import cleanup
from syft.generic.compiler.overrides  import remove_unused_imports

def get_complier_resources(base_module, module_name, framework):
    # folder to deposit each Torch tensor
    target_folder = ROOT_DIR + f"syft/_{framework.lower()}/tensor/"

    print(
        f"Generic ({module_name}.py) -> {framework.capitalize()} ({target_folder.split('PySyft')[1][1:]}{module_name}.py)"
    )

    module = getattr(base_module, module_name)

    tree = mod2ast(module)

    return tree, target_folder


def compile_module(module_name, framework="Numpy"):

    # Get Tensor AST and target folder
    tree, target_folder = get_complier_resources(tensor, module_name, framework)

    # Replace occurrences of "Syft" with "{Framework}"
    tree = SyftToFrameworkNameTransformer(framework).visit(tree)

    # Ignore methods with decorators indicating that {framework} shouldn't use them.
    tree = FrameworkSpecificMethodDecoratorFilter(framework).visit(tree)

    # Convert "sy." to "<framework shorthand>." in method calls
    tree = DecoratorAwareFrameworkHandleTransformer(framework).visit(tree)

    # Convert syft methods to framework methods
    tree = DecoratorAwareFrameworkFunctionTransfomer(framework).visit(tree)

    output = ast2src(tree)

    output = cleanup(output)

    output = remove_unused_imports(output)

    write_output_to_file(output, target_folder, module_name)