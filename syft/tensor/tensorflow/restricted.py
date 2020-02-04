import random
import weakref

class RestrictedTensor(object):
    def __init__(self, data):
        self._id = random.randint(0, 10e30)
        self.register()
        self._data = weakref.ref(data)

    @property
    def data(self):
        return self._data()