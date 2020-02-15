import ast

from syft.generic.compiler.functions import syft2framework

class DecoratorAwareFrameworkHandleTransformer(ast.NodeTransformer):
    """This decorator searches through functions and methods looking
    for occurrences of 'sy.' or 'syft.' and replaces them with the
    handle for the appropriate framework.

    Note: This decorator should only be run AFTER
    FrameworkSpecificMethodDecoratorFilter has been run.

    """

    def __init__(self, framework):
        self.framework = framework
        self.lookup = syft2framework[self.framework]

    def visit_Name(self, node: ast.Name):

        # Convert parent class to Torch Tensor
        if node.id == "sy":
            if self.framework == "Torch":
                node.id = "th"
            elif self.framework == "Numpy":
                node.id = "np"
            elif self.framework == "Tensorflow":
                node.id = "tf"

        return node

class DecoratorAwareFrameworkFunctionTransfomer(ast.NodeTransformer):
    """This decorator looks for attributese with names in the syft2framework
    dictionary and uses it to replace syft method names with framework specific
    method names if necessary"""


    def __init__(self, framework):
        self.framework = framework
        self.lookup = syft2framework[self.framework]

    def visit_Attribute(self, node: ast.Attribute):

        if(node.attr in self.lookup):
            node.attr = self.lookup[node.attr]

        return node