from syft.tensor import tensorflow as tf
import numpy as np
import pytest


@pytest.mark.parametrize("method_name", ["__matmul__", "__add__"])
def test_subclass_method_type_and_values(method_name):
    x_ = np.array([[1, 2], [3, 4]])
    target = x_.__getattribute__(method_name)(x_)

    x = tf.Variable(x_).on(tf.AbstractTensor)
    out = x.__getattribute__(method_name)(x)

    assert isinstance(target, np.ndarray)
    # TODO: right now the Tensor returned doesn't have any kind of
    # correct tensor chain. Gotta fix that.
    # assert isinstance(out.child, tf.AbstractTensor)
    assert (out.numpy() == target).all()
