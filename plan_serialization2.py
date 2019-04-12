import random
import syft as sy
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pytest
import torch

# Import Hook
hook = sy.TorchHook(th, is_client=False)
from syft.frameworks.torch import TorchHook

from syft.serde import serialize, deserialize


sy.create_sandbox(globals(), download_data=False)

device_4 = sy.VirtualWorker(hook, id="device_4")

@sy.func2plan
def plan_mult_3(data):
    return data * 3


foo = serialize(plan_mult_3)
import pickle

#pickle.dump(bar, open('ttst.pkl', 'wb' ))


from multiprocessing import Process


def start_thing(foo):
    bar = deserialize(foo)
    print(bar(torch.Tensor([2,4,5])))

p = Process(target=start_thing, args=(foo,))
p.start()

# Use cases:
# I am bobby
# bobby has a plan, I want to run that plan on MY tensor
