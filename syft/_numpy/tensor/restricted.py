import numpy as np
from syft._numpy.tensor.util import BaseTensor
from syft._numpy.tensor.util import framework
from syft._numpy.tensor.util import execute_default_function_on_child_and_wrap


class RestrictedNumpyTensor(BaseTensor(framework)):
    """A tensor class which returns a NotImplementedError for all methods which you do not
    explicitly override."""

    def init(self, *args, **kwargs):
        """"""

    def __new__(cls, input_array, *args, **kwargs):
        print('New AbstractNumpyArray')
        obj = np.asarray(input_array).view(cls)
        obj.init(input_array, *args, **kwargs)
        return obj

    def __array_finalize__(self, obj):
        """this is just to propagate attributes - this method is called
        after a method/function is run"""
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

    def __array_function__(self, func, types, args=(), kwargs=None):
        """This is basically the same thing as pytorch's __torch_function__
        but it only works for one of numpy's two types of functions. To override
        all of numpy's functions you also need to use __array_ufunc__"""
        return execute_default_function_on_child_and_wrap(self, func, args,
            kwargs, types)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """Unlike pytorch which only has one function type, numpy has two,
        func and ufunc. This is basically __array_function__ for ufuncs."""
        if method == '__call__':
            return execute_default_function_on_child_and_wrap(self, ufunc,
                args, kwargs)
        else:
            return NotImplemented


