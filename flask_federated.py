from flask import Flask
app = Flask(__name__)
from flask import jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import binascii

import syft as sy  # import the Pysyft library
from syft.serde import serialize,deserialize

sy.serde._apply_compress_scheme = sy.serde.apply_no_compression
print(sy.serde._apply_compress_scheme)

hook = sy.TorchHook(torch)  # hook PyTorch ie add extra functionalities
server = hook.local_worker

device_1 = sy.VirtualWorker(hook, id="device_1")

"""
exit()

serialized_plan = str(binascii.hexlify(serialize(plan_double_abs)))
print('serialized', serialized_plan)

unhexed_plan = str(binascii.unhexlify(serialized_plan[2:-1]))
print('unhexed', unhexed_plan)
print('unhexed', deserialize(unhexed_plan))

"""






@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/plan_and_model')
def get_plan_and_model():
    @sy.func2plan
    def plan_double_abs(x):
        x = x + x
        x = torch.abs(x)
        return x



    plan_pointer = plan_double_abs.send(device_1)
    a = torch.ones(2).tag('foobar')
    ptr = a.send(device_1)

    result = plan_pointer(ptr).get()


#    plan_serialized = str(binascii.hexlify(serialized_plan))
    payload = {
            'plan': str(binascii.hexlify(serialize(plan_double_abs))),
            'plan_result': str(binascii.hexlify(serialize(result)))
            }
    return jsonify(payload)


"""
print(plan_double_abs)
    print(serialized_plan)
    print(plan_serialized)
"""
