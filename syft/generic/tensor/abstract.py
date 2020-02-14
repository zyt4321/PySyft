from syft.generic.tensor.util import override_syft_function
from syft.generic.tensor.util import torch_only
from syft.generic.tensor.util import numpy_only
from syft.generic.tensor.restricted import RestrictedSyftTensor

import syft as sy
import torch as th
import numpy as np


HANDLED_FUNCTIONS_ABSTRACT = {}


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

    @torch_only
    def __init__(self, *args, **kwargs):
        print("making abstract tensor")

        self.extra = "some stuff"
        self.some_stuff = "more stuff"

    @numpy_only
    def __new__(cls, input_array, info=None):
        print("New AbstractNumpyArray")
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    @numpy_only
    def __array_finalize__(self, obj):
        """this is just to propagate attributes - this method is called
        after a method/function is run"""

        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    @torch_only
    def __syft_function__(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_ABSTRACT:
            return NotImplemented

        return HANDLED_FUNCTIONS_ABSTRACT[func](*args, **kwargs)

    @numpy_only
    def __array_function__(self, func, types, args, kwargs):
        """This is basically the same thing as pytorch's __torch_function__
        but it only works for one of numpy's two types of functions. To override
        all of numpy's functions you also need to use __array_ufunc__"""

        print("array function")
        if func not in HANDLED_FUNCTIONS_ABSTRACT:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.

        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        return HANDLED_FUNCTIONS_ABSTRACT[func](*args, **kwargs)

    @numpy_only
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Unlike pytorch which only has one function type, numpy has two,
        func and ufunc. This is basically __array_function__ for ufuncs."""

        if ufunc not in HANDLED_FUNCTIONS_ABSTRACT:
            return NotImplemented
        if method == "__call__":
            return HANDLED_FUNCTIONS_ABSTRACT[ufunc](*inputs, **kwargs)
        else:
            return NotImplemented

    @torch_only
    def set(self, **kwargs):
        for name, value in kwargs.items():
            try:
                attr = self.__getattribute__(name)
                self.__setattr__(name, value)
            except Exception as e:
                raise AttributeError(
                    f"Attribute '{name}' does not exist for tensor type {type(self)}"
                )
        return self

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

@torch_only
def method_argument_pre_process(x):
    return x.data

@torch_only
def method_return_post_process(result, out=None, obj_type=AbstractSyftTensor):
    if out is None:
        return obj_type(result)
    else:
        out.data.set_(result)

    return out

@numpy_only
def method_argument_pre_process(x):
    return np.asarray(x)

@torch_only
@override_syft_function(sy.mm, HANDLED_FUNCTIONS_ABSTRACT)
def abstract_mm(input, other, out=None):

    input_data = method_argument_pre_process(input)
    other_data = method_argument_pre_process(other)

    result = sy.mm(input_data, other_data)

    return method_return_post_process(result=result, out=out, obj_type=type(input))

@torch_only
@override_syft_function(sy.add, HANDLED_FUNCTIONS_ABSTRACT)
def abstract_add(input, other, out=None):

    result = sy.add(input.data, other.data)

    if out is None:
        result = AbstractSyftTensor(result)
    else:
        out.data.set_(result)

    return result
