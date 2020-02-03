import random


class AbstractTensor():

    def __init__(self, data):
        self._id = random.randint(0, 10e30)
        self.register()
        self.data = data