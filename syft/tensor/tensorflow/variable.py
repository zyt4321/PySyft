import tensorflow as _tf
from syft.tensor.tensorflow.util import args2child
from syft.tensor.tensorflow import override_funcs


def Variable(*args, **kwargs):
    result = _tf.Variable(*args, **kwargs).register()
    result.set_attr("end", False)
    return result


def create_new_method(method_name):

    old_method = getattr(_tf.Variable, method_name)
    setattr(_tf.Variable, "old_" + method_name, old_method)

    @_tf.custom_gradient
    def _func(self, *args, **kwargs):

        if not self.attr("end") and self.child is not None:
            self, args, kwargs = args2child(self, *args, **kwargs)
            result = getattr(self, method_name)(*args, **kwargs)
        else:
            result = getattr(self, "old_" + method_name)(*args, **kwargs)

        def grad(dy):
            return dy, dy

        return result, grad

    def func(self, other):
        return _func(self, other)

    setattr(_tf.Variable, method_name, func)


for method_name in override_funcs:
    create_new_method(method_name)
