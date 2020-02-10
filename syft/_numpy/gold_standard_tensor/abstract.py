import numpy as np

HANDLED_FUNCTIONS = {}


class AbstractNumpyArray(np.ndarray):
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Unlike pytorch which only has one function type, numpy has two,
        func and ufunc. This is basically __array_function__ for ufuncs."""

        if ufunc not in HANDLED_FUNCTIONS:
            return NotImplemented
        if method == "__call__":
            return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        """This is basically the same thing as pytorch's __torch_function__
        but it only works for one of numpy's two types of functions. To override
        all of numpy's functions you also need to use __array_ufunc__"""

        print("array function")
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.

        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    # Option 2: call into one's own __array_ufunc__
    def __matmul__(self, other):
        print("mm")
        return self.__array_ufunc__(np.matmul, "__call__", self, other)

    # Option 2: call into one's own __array_ufunc__
    def matmul(self, other):
        print("mm2")
        return self.__array_ufunc__(np.matmul, "__call__", self, other)

    # Option 2: call into one's own __array_ufunc__
    def __add__(self, other):
        print("adding")
        return self.__array_ufunc__(np.add, "__call__", self, other)

    def __radd__(self, other):
        print("r adding")
        return self.__array_ufunc__(np.add, "__call__", other, self)

    def __iadd__(self, other):
        print("i adding")
        result = self.__array_ufunc__(np.add, "__call__", self, other, out=(self,))
        if result is NotImplemented:
            raise TypeError(...)


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.matmul)
def abstract_mm(x, y):
    "Implementation of np.sum for DiagonalArray objects"
    x = np.asarray(x)
    y = np.asarray(y)
    return AbstractNumpyArray(np.matmul(x, y))


@implements(np.add)
def multiply(x, y):
    "Implementation of np.sum for DiagonalArray objects"
    print("np.add")
    x = np.asarray(x)
    y = np.asarray(y)
    return AbstractNumpyArray(x + y)
