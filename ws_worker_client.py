import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import syft
from syft.workers import WebsocketClientWorker
from syft.workers import WebsocketServerWorker
from multiprocessing import Process
import threading
import asyncio

import numpy as np
from collections import ChainMap as merge

hook = syft.TorchHook(torch)
kwargs = {"id": "bob", "host": "localhost", "port": 8765, "hook": hook}
client = WebsocketClientWorker(**kwargs)
client.ready_to_compute()

kwargs = {"id": "fed1", "host": "localhost", "port": 8765, "hook": hook}
server = start_proc(WebsocketServerWorker, kwargs)

time.sleep(0.1)
x = torch.ones(5)

socket_pipe = WebsocketClientWorker(**kwargs)

x = x.send(socket_pipe)
y = x + x
y = y.get()

assert (y == torch.ones(5) * 2).all()


