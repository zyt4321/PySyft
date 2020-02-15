from syft._torch.tensor.restricted import RestrictedTorchTensor
import torch as th


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

    def init(self, *args, **kwargs):
        self.child = args[0]
        self.extra = 'some stuff'
        self.some_stuff = 'more stuff'

    @property
    def grad(self):
        return type(self)(self.child.grad)

    def mm(self, other):
        return th.mm(self, other)

    def __matmul__(self, other):
        return th.matmul(self, other)

    def dot(self, other):
        return th.matmul(self, other)

    def __add__(self, other):
        return th.add(self, other)

    def __radd__(self, other):
        return th.add(other, self)

    def __iadd__(self, other):
        return th.add(self, other, out=(self,))


