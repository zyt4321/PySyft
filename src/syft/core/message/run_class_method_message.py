from .syft_message import SyftMessage
from typing import Any


class RunClassMethodMessage(SyftMessage):
    def __init__(
        self,
        path: str,
        _self: Any,
        args: List[Any],
        kwargs: Dict[Any, Any],
        id_at_location: str,
    ):
        super.__init__(
            path=path, obj=_self, args=args, kwargs=kwargs, id_remote=id_at_location
        )
