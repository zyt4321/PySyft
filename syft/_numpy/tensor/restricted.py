import numpy as np
from syft._numpy.tensor.util import BaseTensor
from syft._numpy.tensor.util import framework
from syft._numpy.tensor.util import execute_default_function_on_child_and_wrap


class RestrictedTensor(BaseTensor(framework)):
    """A tensor class which returns a NotImplementedError for all methods which you do not
    explicitly override."""

    @staticmethod
    def Constructor(x):
        try:
            return RestrictedTensor(x)
        except TypeError as e:
            result = RestrictedTensor(x.data)
            result.child = x
            return result

    def __new__(cls, input_array, *args, **kwargs):
        print('New AbstractNumpyArray')
        obj = np.asarray(input_array).view(cls)
        obj.post_init(input_array, *args, **kwargs)
        return obj

    def post_init(self, *args, **kwargs):
        """"""

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

    def __str__(self):
        if hasattr(self.child, 'str_recurse'):
            result = f'[{type(self).__name__} -> {self.child.str_recurse()}]'
        else:
            result = f'[{type(self).__name__} -> {self.child}]'
        if 'tensor(' in result:
            split_str = str(result).split('tensor(')
            base_len = float(len(split_str[0]) + 8)
            c = '['
            ci = -1
            while c == '[':
                base_len += 0.5
                c = split_str[1][ci]
                ci += 1
            base_len = int(base_len)
            result = result.replace('        ', ' ' * base_len)
        return result

    def str_recurse(self):
        if not self.child == BaseTensor(framework):
            return f'{type(self).__name__} -> {str(self.child)}'
        else:
            return f'{type(self).__name__} -> {type(self.child).__name__}'

    def __repr__(self):
        return str(self)

    def backward(self, *args, **kwargs):
        return self.child.backward(*args, **kwargs)

    @property
    def has_grandchild(self):
        return hasattr(self.child, 'child')


