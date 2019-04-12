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

# Import grids
from syft.grid import VirtualGrid

sy.create_sandbox(globals(), download_data=False)

device_4 = sy.VirtualWorker(hook, id="device_4", verbose=True)

@sy.func2plan
def plan_mult_3(data):
    return data * 3

x_ptr = th.tensor([-1, 2, 3]).send(device_4)
sent_plan = plan_mult_3.send(device_4)
sent_plan(x_ptr)

plan_copy = device_4.fetch_plan(plan_mult_3.id)
assert plan_copy.id != plan_mult_3.id
