from syft._numpy.tensor import restricted
from syft._numpy.tensor import abstract
from syft._numpy.tensor import precision


def RestrictedSyftTensor(x):
    try:
        return restricted.RestrictedSyftTensor(x)
    except TypeError as e:
        result = restricted.RestrictedSyftTensor(x.data)
        result.child = x
        return result


