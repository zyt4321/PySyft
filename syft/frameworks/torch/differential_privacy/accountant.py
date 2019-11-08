import torch as th
class PrivacyAccountant():

    def __init__(self, n_entities):
        self.n_entities = n_entities
        self.epsilons = th.zeros(self.n_entities)
