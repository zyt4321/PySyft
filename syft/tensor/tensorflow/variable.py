import tensorflow as _tf

def Variable(*args, **kwargs):
    result = _tf.Variable(*args, **kwargs).register()
    return result