
from syft.generic.tensor.util import torch_only #CLEAN
from syft.generic.tensor.util import numpy_only #CLEAN
from syft.generic.tensor.restricted import RestrictedSyftTensor

import syft as sy
import torch as th
import numpy as np

class AbstractSyftTensor(RestrictedSyftTensor):
    """A subclass of syft.Tensor which is necessary to extend Syft's ability
    to implement custom functions into an ability to also extend custom methods.
    This tensor type execute the same functionality as a normal th.Tensor, but you
    can subclass it more conveniently. When subclassed, all default PySyft functionality
    will be included.

    >>> s = AbstractSyftTensor([[1, 1], [1, 1]])
    >>> torch.mm(s, s)
    0
    >>> t = syft.tensor([[1, 1], [1, 1]])
    >>> torch.mm(s, t)
    0
    >>> torch.mm(t, s)
    0
    >>> torch.mm(t, t)
    tensor([[2, 2],
            [2, 2]])
    This is useful for testing that the semantics for overriding syft
    functions are working correctly.
    """

    @staticmethod
    def Constructor(x):
        try:
            return AbstractSyftTensor(x)
        except TypeError as e:
            result = AbstractSyftTensor(x.data)
            result.child = x
            return result

    def post_init(self, *args, **kwargs):

        self.child = args[0]

        self.extra = "some stuff"
        self.some_stuff = "more stuff"

    @torch_only
    @property
    def grad(self):
        return type(self).Constructor(self.child.grad)

    def mm(self, other):
        return sy.mm(self, other)

    def __matmul__(self, other):
        return sy.matmul(self, other)

    def dot(self, other):
        return sy.matmul(self, other)

    def __add__(self, other):
        return sy.add(self, other)

    def __radd__(self, other):
        return sy.add(other, self)

    def __iadd__(self, other):
        return sy.add(self, other, out=(self,))

# EXAMPLE OF HOW TO DO A CUSTOM FUNCTION
# from syft.generic.tensor.util import override_syft_function
# from syft.generic.tensor.util import HANDLED_FUNCTIONS_ABSTRACT
# from syft.generic.tensor.util import method_return_post_process
# from syft.generic.tensor.util import method_argument_pre_process
# @override_syft_function(sy.add, HANDLED_FUNCTIONS_ABSTRACT)
# def abstract_add(input, other, out=None):
#
#     result = sy.add(input, other, out=out)
#
#     return result
