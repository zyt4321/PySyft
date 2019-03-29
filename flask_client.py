import requests
import json

import binascii
from flask import jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import syft as sy  # import the Pysyft library
from syft.serde import serialize,deserialize

sy.serde._apply_compress_scheme = sy.serde.apply_no_compression
hook = sy.TorchHook(torch)  # hook PyTorch ie add extra functionalities
server = hook.local_worker



response = requests.get('http://0.0.0.0:5000/plan_and_model')

plan_json = response.json()['plan_result']
plan = binascii.unhexlify(plan_json[2:-1])
print(deserialize(plan))
