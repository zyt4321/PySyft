from syft.tensor.tensorflow import AbstractTensor
from syft.tensor.tensorflow import chain_method

class PlusIsMinusTensor(AbstractTensor):

    @chain_method
    def __add__(self, other):
        return self - other


class MinusIsMultiplyTensor(AbstractTensor):

    @chain_method
    def __sub__(self, other):
        return self * other