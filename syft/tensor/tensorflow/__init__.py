import tensorflow as _tf

from tensorflow import Tensor
from syft.tensor.tensorflow.variable import Variable

ResourceVariable = None
w = _tf.Variable([[100.0]])
with _tf.GradientTape() as tape:
    ResourceVariable = type(w)

_tf.ResourceVariable = ResourceVariable

from syft.tensor.tensorflow.util import chain_method

from syft.tensor.tensorflow.abstract import AbstractTensor
from syft.tensor.tensorflow.experimental import PlusIsMinusTensor
from syft.tensor.tensorflow.experimental import MinusIsMultiplyTensor

from syft.tensor.tensorflow import codegen

from syft.tensor.tensorflow import attributes
