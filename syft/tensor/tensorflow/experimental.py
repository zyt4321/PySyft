from syft.tensor.tensorflow import RestrictedTensor
from syft.tensor.tensorflow import chain_method


class PlusIsMinusTensor(RestrictedTensor):
    @chain_method
    def __add__(self, other):
        return self - other


class MinusIsMultiplyTensor(RestrictedTensor):
    @chain_method
    def __sub__(self, other):
        return self * other
