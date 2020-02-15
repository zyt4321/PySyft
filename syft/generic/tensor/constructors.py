from syft.generic.tensor import restricted
from syft.generic.tensor import abstract
from syft.generic.tensor import precision

def RestrictedSyftTensor(x):
    try:
        return restricted.RestrictedSyftTensor(x)
    except TypeError as e:
        result = restricted.RestrictedSyftTensor(x.data)
        result.child = x
        return result

def AbstractSyftTensor(x):
    try:
        return abstract.AbstractSyftTensor(x)
    except TypeError as e:
        result = abstract.AbstractSyftTensor(x.data)
        result.child = x
        return result