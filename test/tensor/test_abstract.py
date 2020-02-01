from syft.tensor.torch.abstract import AbstractTorchTensor
import torch as th
import pytest


@pytest.mark.parametrize("method_name", ["mm", "__add__"])
def test_subclass_method_type_and_values(method_name):

    x_ = th.tensor([[1, 2], [3, 4]])
    target = x_.__getattribute__(method_name)(x_)

    x = AbstractTorchTensor([[1, 2], [3, 4]])
    out = x.__getattribute__(method_name)(x)

    assert isinstance(target, th.Tensor)
    assert isinstance(out, AbstractTorchTensor)
    assert (out == target).all()
