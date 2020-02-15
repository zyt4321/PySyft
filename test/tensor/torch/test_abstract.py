from syft._torch.tensor.abstract import AbstractTorchTensor
import torch as th
import pytest


@pytest.mark.parametrize("method_name", ["mm", "__add__"])
def test_subclass_method_type_and_values(method_name):

    x_ = th.tensor([[1, 2], [3, 4.]])
    target = x_.__getattribute__(method_name)(x_)

    x = AbstractTorchTensor(x_)
    out = x.__getattribute__(method_name)(x)

    print(th.__version__)

    assert isinstance(target, th.Tensor)
    assert isinstance(out, AbstractTorchTensor)
    assert (out == target).all()