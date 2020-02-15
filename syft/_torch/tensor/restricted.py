from syft._torch.tensor.util import BaseTensor
from syft._torch.tensor.util import framework
from syft._torch.tensor.util import execute_default_function_on_child_and_wrap


class RestrictedTorchTensor(BaseTensor(framework)):
    """A tensor class which returns a NotImplementedError for all methods which you do not
    explicitly override."""

    def init(self, *args, **kwargs):
        """"""

    def __init__(self, *args, **kwargs):
        self.init(*args, **kwargs)

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

    def __torch_function__(self, func, args=(), kwargs=None):
        return execute_default_function_on_child_and_wrap(self, func, args,
            kwargs)


