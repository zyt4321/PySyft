from syft_tensor import SyftTensor

a = SyftTensor()
b = SyftTensor()


def add_two_tensors(a, b):
    return a + b


def matmul(a, b):
    if a > 3:
        return a * b
    else:
        return -a * b
