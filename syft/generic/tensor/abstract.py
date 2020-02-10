from syft.generic.tensor.util import override_syft_function
from syft.generic.tensor.restricted import RestrictedSyftTensor

import torch as th


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

    def __init__(self, *args, **kwargs):
        print("making abstract tensor")

        self.extra = "some stuff"
        self.some_stuff = "more stuff"

    def __syft_function__(self, func, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_ABSTRACT:
            return NotImplemented

        return HANDLED_FUNCTIONS_ABSTRACT[func](*args, **kwargs)

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
        return th.mm(self, other)

    def __add__(self, other):
        return th.add(self, other)


@override_syft_function(th.mm, HANDLED_FUNCTIONS_ABSTRACT)
def abstract_mm(input, other, out=None):

    result = th.mm(input.data, other.data)

    if out is None:
        return AbstractSyftTensor(result)
    else:
        out.data.set_(result)

    return out


@override_syft_function(th.add, HANDLED_FUNCTIONS_ABSTRACT)
def abstract_add(input, other, out=None):

    result = th.add(input.data, other.data)

    if out is None:
        result = AbstractSyftTensor(result)
    else:
        out.data.set_(result)

    return result
