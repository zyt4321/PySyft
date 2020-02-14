import ast


class DecoratorAwareFrameworkHandleTransformer(ast.NodeTransformer):
    """This decorator searches through functions and methods looking
    for occurrences of 'sy.' or 'syft.' and replaces them with the
    handle for the appropriate framework.

    Note: This decorator should only be run AFTER
    FrameworkSpecificMethodDecoratorFilter has been run.

    """

    def __init__(self, framework):
        self.framework = framework

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
