from syft._torch.tensor.util import BaseTensor
from syft._torch.tensor.util import framework
from syft._torch.tensor.util import execute_default_function_on_child_and_wrap


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

    def __init__(self, *args, **kwargs):
        self.post_init(*args, **kwargs)

    def post_init(self, *args, **kwargs):
        """"""

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

    def __str__(self):
        if not self.child == BaseTensor(framework):
            result = f'[{type(self).__name__} -> {self.child.str_recurse()}]'
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
        else:
            return f'[{type(self).__name__} -> {self.child}]'

    def str_recurse(self):
        if not self.child == BaseTensor(framework):
            return f'{type(self).__name__} -> {str(self.child)}'
        else:
            return f'{type(self).__name__} -> {type(self.child).__name__}'

    def __repr__(self):
        return str(self)


