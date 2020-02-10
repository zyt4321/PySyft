from syft._tensorflow.tensor.restricted import RestrictedTensor
from syft._tensorflow.tensor.util import chain_method


class PlusIsMinusTensor(RestrictedTensor):
    @chain_method
    def __add__(self, other):
        return self - other


class MinusIsMultiplyTensor(RestrictedTensor):
    @chain_method
    def __sub__(self, other):
        return self * other
