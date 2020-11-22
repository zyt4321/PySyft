"""
This file exists to provide one common place for all serialization to occur
regardless of framework. By default, we serialize using msgpack and compress
using lz4. If different compressions are required, the worker can override
the function apply_compress_scheme.
"""

# import numpy as np
# import pyarrow
# import syft

from typing import Callable

from syft.serde.msgpack import serialize as msgpack_serialize
from syft.serde.msgpack import deserialize as msgpack_deserialize


## SECTION:  High Level Public Functions (these are the ones you use)
def serialize(
    obj: object,
    worker=None,
    simplified: bool = False,
    force_full_simplification: bool = False,
    strategy: Callable[[object], bin] = None,
) -> bin:
    """This method can serialize any object PySyft needs to send or store.

    This is the high level function for serializing any object or collection
    of objects which PySyft needs to send over the wire. It includes three
    steps, Simplify, Serialize, and Compress as described inline below.

    Args:
        obj (object): the object to be serialized
        simplified (bool): in some cases we want to pass in data which has
            already been simplified - in which case we must skip double
            simplification - which would be bad.... so bad... so... so bad
        force_full_simplification (bool): Some objects are only partially serialized
            by default. For objects where this is the case, setting this flag to True
            will force the entire object to be serialized. For example, setting this
            flag to True will cause a VirtualWorker to be serialized WITH all of its
            tensors while by default VirtualWorker objects only serialize a small
            amount of metadata.

    Returns:
        binary: the serialized form of the object.
    """
    # print(f"serding stuff: {type(obj)}")
    # print(obj)
    # if isinstance(obj, syft.messaging.message.WorkerCommandMessage):
    #     if obj.command_name == "feed_crypto_primitive_store":
    #         # print(obj.message[0][0])
    #         if "fss_comp" in obj.message[0][0]:
    #             print(
    #                 f"Strat: {strategy}, simplified? \
    # {simplified} forcefull {force_full_simplification}"
    #             )
    # return pyarrow.serialize(obj.message[0][0]).to_buffer()
    # s = pyarrow.serialize(obj.message[0][0]).to_buffer()
    # print(s)
    # print(type(s))
    # print(pyarrow.deserialize(s))
    #     print("JACK")
    #     print(obj.message[0][0]["fss_comp"])
    # print(obj.message)
    # print(len(obj.message))
    # print()
    # print(obj[0])
    # print("ser array")
    # print(obj.shape)

    if strategy is None:

        strategy = msgpack_serialize

    return strategy(obj, worker, simplified, force_full_simplification)


def deserialize(
    binary: bin,
    worker=None,
    strategy: Callable[[bin], object] = None,
) -> object:
    """This method can deserialize any object PySyft needs to send or store.

    This is the high level function for deserializing any object or collection
    of objects which PySyft has sent over the wire or stored. It includes three
    steps, Decompress, Deserialize, and Detail as described inline below.

    Args:
        binary (bin): the serialized object to be deserialized.
        worker (AbstractWorker): the worker which is acquiring the message content,
            for example used to specify the owner of a tensor received(not obvious
            for virtual workers)
        details (bool): there are some cases where we need to perform the decompression
            and deserialization part, but we don't need to detail all the message.
            This is the case for Plan workers for instance

    Returns:
        object: the deserialized form of the binary input.
    """

    # print(len(bin))
    # print(type(bin))
    # if isinstance(bin, pyarrow.lib.Buffer):
    #     print("buff")

    if strategy is None:

        strategy = msgpack_deserialize

    return strategy(binary, worker)
