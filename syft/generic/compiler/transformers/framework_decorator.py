import ast

class FrameworkSpecificMethodDecoratorFilter(ast.NodeTransformer):

    def __init__(self, framework):
        self.framework = framework

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # change occurances of "syft" in decorator names to "torch"
        for d_i, decorator in enumerate(node.decorator_list):

            if (hasattr(decorator, 'id')):
                if (decorator.id[-5:] == "_only"):
                    if (decorator.id[:-5] != self.framework.lower()):
                        return None
                    del node.decorator_list[d_i]

        for body_part in node.body:
            body_part = self.visit(body_part)

        return node
