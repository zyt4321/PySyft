from ...protos.message_pb2 import SyftMessage as SyftMessageProto
from typing import Optional, Dict, Any, List, Callable
import pickle
import pysyft


class SyftMessageProxy:
    def __init__(
        self,
        id: Optional[str] = None,
        id_remote: Optional[str] = None,
        path: Optional[str] = None,
        args: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, str]] = None,
        obj: Optional[Any] = None,
    ):
        request = self.__create_request(
            id=id, id_remote=id_remote, path=path, args=args, kwargs=kwargs, obj=obj
        )
        self.response = self.__get_response(
            address="localhost", capability="message", request=request
        )

    def __create_request(
        self,
        id: Optional[str] = None,
        id_remote: Optional[str] = None,
        path: Optional[str] = None,
        args: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, str]] = None,
        obj: Optional[Any] = None,
    ) -> SyftMessageProto:
        request = SyftMessageProto()
        if id is not None:
            request.local_id = id
        if id_remote is not None:
            request.remote = id_remote

        if path is not None:
            request.path = path

        if args is not None:
            request.args.extend(args)

        if kwargs is not None:
            for key, value in kwargs.items():
                request.kwargs[key] = value

        if obj is not None:
            request.object = pickle.dumps(obj)

        return request

    def __get_response(
        self, address: str, capability: str, request: SyftMessageProto
    ) -> Optional[SyftMessageProto]:
        request_bytes = request.SerializeToString()
        try:
            # this is where we are calling rust
            response_bytes = pysyft.message.run_class_method_message(
                address, capability, request_bytes
            )
            response = SyftMessageProto()
            response.ParseFromString(bytes(response_bytes))
            print(f"Python got response: {response}")
            return response
        except Exception as e:
            print(f"Python failed to decode response {response_bytes}, error: {e}")
            return None


class SyftMessage(SyftMessageProxy):
    def __getattr__(self, attr):
        return getattr(self.response, attr)

