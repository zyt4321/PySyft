from typing import Optional, Dict, Any, List, Callable

import pysyft
from ...protos.message_pb2 import SyftMessage

# id: string identifier
# path: string path
# args: string array of arguments
# kwargs: dictionary of keyword arguments
# object: any type of python object or pointer


def make_message(
    id: Optional[str],
    path: Optional[str],
    args: List[str],
    kwargs: Dict[str, str],
    object: Any,
) -> SyftMessage:
    message = SyftMessage()
    if id is not None:
        message.remote = id

    if path is not None:
        message.path = path

    if args is not None:
        message.args.extend(args)

    if kwargs is not None:
        for key, value in kwargs.items():
            message.kwargs[key] = value

    return message


class RunClassMethodMessage:

    def __init__(self, path, _self, args, kwargs, id_at_location):
        self.message = make_message(
            id=id_at_location,
            path=path,
            args=args,
            kwargs=kwargs,
            object=_self)


class RunFunctionOrConstructorMessage:
    def __init__(self, path, args, kwargs):
        self.message = make_message(
            id=None,
            path=path,
            args=args,
            kwargs=kwargs,
            object=None
        )


class SaveObjectMessage:
    def __init__(self, id, obj):
        self.message = make_message(
            id=id,
            path=None,
            args=None,
            kwargs=None,
            object=obj
        )


class GetObjectMessage:
    def __init__(self, id):
        self.message = make_message(
            id=id,
            path=None,
            args=None,
            kwargs=None,
            object=None
        )


class DeleteObjectMessage:
    def __init__(self, id):
        self.message = make_message(
            id=id,
            path=None,
            args=None,
            kwargs=None,
            object=None
        )
