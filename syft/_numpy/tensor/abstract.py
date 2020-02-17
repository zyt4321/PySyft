from syft._numpy.tensor.restricted import RestrictedTensor
import numpy as np


class AbstractTensor(RestrictedTensor):
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

    @staticmethod
    def Constructor(x):
        try:
            return AbstractTensor(x)
        except TypeError as e:
            result = AbstractTensor(x.data)
            result.child = x
            return result

    def post_init(self, *args, **kwargs):
        self.child = args[0]
        self.extra = 'some stuff'
        self.some_stuff = 'more stuff'

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


