from syft_tensor import SyftTensor
import ast
import astor


def add_two_tensors(a, b):
    return a + b


class CodeGen(object):
    def __init__(self):
        pass

    def gen_torch(self, some_function):
        # Generate tree for some_function

        # Iterate through nodes in AST
        # Replace SyftTensor.__add__ with torch.Tensor.__add__
        pass


cg = CodeGen()

cg.gen_torch(add_two_tensors)


# Convert back to Python
