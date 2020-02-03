import tensorflow as _tf


@_tf.custom_gradient
def _var_add(self, other):
    if (hasattr(self, 'child') and self.child is not None and hasattr(other, 'child') and other.child is not None):
        result = self.child + other.child
    else:
        result = tf.add(self, other)

    def grad(dy):
        return dy, dy

    return result, grad


def var_add(self, other):
    return _var_add(self, other)


_tf.Variable.__add__ = var_add