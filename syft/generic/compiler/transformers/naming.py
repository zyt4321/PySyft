import ast

from syft.generic.compiler.util import syft2framework_string

class SyftToFrameworkNameTransformer(ast.NodeTransformer):

    def __init__(self, framework):
        self.framework = framework

    # change imports from Syft -> Framework
    def visit_ImportFrom(self, node: ast.ImportFrom):

        node.module = node.module.replace(".generic.", f"._{self.framework.lower()}.")

        for name in node.names:
            self.visit(name)
            if ("syft" in name.name.lower()):
                name.name = syft2framework_string(name.name, self.framework)

        return node

    # change imports from Syft -> Framework
    def visit_ClassDef(self, node: ast.ImportFrom):

        if ("Syft" in node.name):
            node.name = syft2framework_string(node.name, self.framework)

        for base in node.bases:
            if (hasattr(base, 'id')):
                if ("Syft" in base.id):
                    base.id = syft2framework_string(base.id, self.framework)

        for body_part in node.body:
            body_part = self.visit(body_part)

        return node

    def visit_Name(self, node: ast.Name):

        # Convert parent class to Torch Tensor
        if 'Syft' in node.id:
            result = ast.Name(id=syft2framework_string(node.id, self.framework))
            return ast.copy_location(result, node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):

        # change occurrences of "syft" in function names to "torch"
        if 'syft' in node.name:
            node.name = syft2framework_string(node.name, self.framework)

        # change occurances of "syft" in decorator names to "torch"
        for decorator in node.decorator_list:

            if hasattr(decorator, 'func'):
                if (hasattr(decorator.func, 'id')):
                    if 'syft' in decorator.func.id:
                        decorator.func.id = syft2framework_string(decorator.func.id, self.framework)

        for body_part in node.body:
            body_part = self.visit(body_part)

        return node