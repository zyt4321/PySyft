import torch as th
from syft.tensor import AbstractTorchTensor


class FixedPrecisionTensor(AbstractTorchTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base = 10
        self.precision_fractional = 3

        self.data = self.data * self.scaling_factor

    @property
    def scaling_factor(self):
        return self.base ** self.precision_fractional
