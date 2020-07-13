from .syft_message import SyftMessage
from typing import Any, List, Dict


class RunFunctionOrConstructorMessage(SyftMessage):
    def __init__(self, path: str, args: List[Any], kwargs: Dict[Any, Any]):
        super().__init__(path=path, args=args, kwargs=kwargs)
