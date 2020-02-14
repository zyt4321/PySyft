from syft._numpy.tensor.util import override_numpy_function
from syft._numpy.tensor.util import torch_only
from syft._numpy.tensor.util import numpy_only
from syft._numpy.tensor.restricted import RestrictedNumpyTensor
import syft as sy
import torch as th
import numpy as np

HANDLED_FUNCTIONS_ABSTRACT = {}


class AbstractNumpyTensor(RestrictedNumpyTensor):
    """A subclass of numpy.Tensor which is necessary to extend Numpy's ability
    to implement custom functions into an ability to also extend custom methods.
    This tensor type execute the same functionality as a normal th.Tensor, but you
    can subclass it more conveniently. When subclassed, all default PyNumpy functionality
    will be included.

    >>> s = AbstractNumpyTensor([[1, 1], [1, 1]])
    >>> torch.mm(s, s)
    0
    >>> t = numpy.tensor([[1, 1], [1, 1]])
    >>> torch.mm(s, t)
    0
    >>> torch.mm(t, s)
    0
    >>> torch.mm(t, t)
    tensor([[2, 2],
            [2, 2]])
    This is useful for testing that the semantics for overriding numpy
    functions are working correctly.
    """

    def __new__(cls, input_array, info=None):
        print("New AbstractNumpyArray")
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        """this is just to propagate attributes - this method is called
        after a method/function is run"""
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    def __array_function__(self, func, types, args, kwargs):
        """This is basically the same thing as pytorch's __torch_function__
        but it only works for one of numpy's two types of functions. To override
        all of numpy's functions you also need to use __array_ufunc__"""
        print("array function")
        if func not in HANDLED_FUNCTIONS_ABSTRACT:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS_ABSTRACT[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Unlike pytorch which only has one function type, numpy has two,
        func and ufunc. This is basically __array_function__ for ufuncs."""
        if ufunc not in HANDLED_FUNCTIONS_ABSTRACT:
            return NotImplemented
        if method == "__call__":
            return HANDLED_FUNCTIONS_ABSTRACT[ufunc](*inputs, **kwargs)
        else:
            return NotImplemented

    def mm(self, other):
        return np.mm(self, other)

    def __matmul__(self, other):
        return np.matmul(self, other)

    def dot(self, other):
        return np.matmul(self, other)

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __iadd__(self, other):
        return np.add(self, other, out=(self,))


def method_argument_pre_process(x):
    return np.asarray(x)
