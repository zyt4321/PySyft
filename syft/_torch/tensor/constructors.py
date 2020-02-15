from syft._torch.tensor import restricted
from syft._torch.tensor import abstract
from syft._torch.tensor import precision


def RestrictedSyftTensor(x):
    try:
        return restricted.RestrictedSyftTensor(x)
    except TypeError as e:
        result = restricted.RestrictedSyftTensor(x.data)
        result.child = x
        return result


