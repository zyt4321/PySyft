import tensorflow as _tf
from syft.tensor.tensorflow.restricted import RestrictedTensor

object_store = {}


@property
def id(self):
    if hasattr(self, "_id"):
        return self._id
    return hash(self.experimental_ref())


@property
def child(self):
    return self.attr("child")


def attr(self, attr_name):
    try:
        return object_store[self.id][attr_name]
    except Exception as e:
        return None


def set_attr(self, attr_name, value):
    object_store[self.id][attr_name] = value


def register(self):
    attrs = {}
    object_store[self.id] = attrs
    return self


def on(self, tensor_type, *args, **kwargs):
    child = tensor_type(self, *args, **kwargs)

    if self.child is not None:
        grandchild = self.child
        child.set_attr("child", grandchild)

    self.set_attr("child", child)

    return self


methods = list()
methods.append(("id", id))
methods.append(("attr", attr))
methods.append(("register", register))
methods.append(("set_attr", set_attr))
methods.append(("child", child))
methods.append(("on", on))

objects = list()
objects.append(_tf.Tensor)
objects.append(_tf.Variable)
objects.append(_tf.ResourceVariable)
objects.append(RestrictedTensor)

for obj in objects:
    for method_name, method in methods:
        setattr(obj, method_name, method)
