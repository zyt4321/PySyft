import torch as th


class RestrictedTorchTensor(th.Tensor):
    """A tensor class which returns a NotImplementedError for all methods which you do not
    explicitly override."""


def create_not_implemented_method(method_name):
    def raise_not_implemented_exception(self, *args, **kwargs):
        msg = f"You just tried to execute {method_name} on tensor type '{(type(self))}."
        msg += " However, method does not exist within this class."

        raise NotImplemented(msg)

    return raise_not_implemented_exception


for method_name in ["__add__", "__sub__"]:  # TODO: add all relevant methods
    new_method = create_not_implemented_method(method_name)
    setattr(RestrictedTorchTensor, method_name, new_method)
