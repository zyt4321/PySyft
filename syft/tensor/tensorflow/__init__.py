import tensorflow as _tf

# set these here based on whatever had been overridden in AbstractTensor (abstract.py)
# but they change what happens in tf.Variable (in variable.py)
override_funcs = set()
override_funcs.add("__add__")
override_funcs.add("__sub__")
override_funcs.add("__mul__")

from syft.tensor.tensorflow.util import chain_method
from syft.tensor.tensorflow.restricted import RestrictedTensor
from syft.tensor.tensorflow.abstract import AbstractTensor

from tensorflow import Tensor
from syft.tensor.tensorflow.variable import Variable

ResourceVariable = None
w = _tf.Variable([[100.0]])
with _tf.GradientTape() as tape:
    ResourceVariable = type(w)

_tf.ResourceVariable = ResourceVariable

from syft.tensor.tensorflow.experimental import PlusIsMinusTensor
from syft.tensor.tensorflow.experimental import MinusIsMultiplyTensor

from syft.tensor.tensorflow import attributes
