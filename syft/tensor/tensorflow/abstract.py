from syft.tensor.tensorflow import RestrictedTensor
from syft.tensor.tensorflow import chain_method


class AbstractTensor(RestrictedTensor):
    @chain_method
    def __add__(self, other):
        return self + other

    @chain_method
    def __sub__(self, other):
        return self - other

    @chain_method
    def __matmul__(self, other):
        return self @ other
