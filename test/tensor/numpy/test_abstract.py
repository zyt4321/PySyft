from syft._numpy.gold_standard_tensor.abstract import AbstractNumpyArray
import numpy as np
import pytest


@pytest.mark.parametrize("method_name", ["__matmul__", "__add__"])
def test_subclass_method_type_and_values(method_name):
    x_ = np.array([[1, 2], [3, 4]])
    target = x_.__getattribute__(method_name)(x_)

    x = AbstractNumpyArray([[1, 2], [3, 4]])
    out = x.__getattribute__(method_name)(x)

    assert isinstance(target, np.ndarray)
    assert isinstance(out, AbstractNumpyArray)
    assert (np.asarray(out) == target).all()
