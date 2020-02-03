import tensorflow as _tf

from tensorflow import Tensor
from syft.tensor.tensorflow.variable import Variable

ResourceVariable = None
w = _tf.Variable([[100.0]])
with _tf.GradientTape() as tape:
    ResourceVariable = type(w)

