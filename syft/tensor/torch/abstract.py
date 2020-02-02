import torch as th

from syft.tensor.torch.util import override_torch_function
from syft.tensor import RestrictedTorchTensor

HANDLED_FUNCTIONS_ABSTRACT = {}


class AbstractTorchTensor(RestrictedTorchTensor):
    """A subclass of torch.Tensor which is necessary to extend Torch's ability
    to implement custom functions into an ability to also extend custom methods.
    This tensor type execute the same functionality as a normal th.Tensor, but you
    can subclass it more conveniently. When subclassed, all default PyTorch functionality
    will be included.

    >>> s = AbstractTorchTensor([[1, 1], [1, 1]])
    >>> torch.mm(s, s)
    0
    >>> t = torch.tensor([[1, 1], [1, 1]])
    >>> torch.mm(s, t)
    0
    >>> torch.mm(t, s)
    0
    >>> torch.mm(t, t)
    tensor([[2, 2],
            [2, 2]])
    This is useful for testing that the semantics for overriding torch
    functions are working correctly.
    """

    def __init__(self, *args, **kwargs):
        print("making abstract tensor")

        self.extra = "some stuff"
        self.some_stuff = "more stuff"

    def __torch_function__(self, func, args=(), kwargs=None):
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


@override_torch_function(th.mm, HANDLED_FUNCTIONS_ABSTRACT)
def abstract_mm(input, other, out=None):

    result = th.mm(input.data, other.data)

    if out is None:
        return AbstractTorchTensor(result)
    else:
        out.data.set_(result)

    return out


@override_torch_function(th.add, HANDLED_FUNCTIONS_ABSTRACT)
def abstract_add(input, other, out=None):

    result = th.add(input.data, other.data)

    if out is None:
        result = AbstractTorchTensor(result)
    else:
        out.data.set_(result)

    return result
