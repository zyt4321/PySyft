from .syft_message import SyftMessage
from typing import Any


class SaveObjectMessage(SyftMessage):
    def __init__(self, id: str, obj: Any):
        super().__init__(id=id, obj=obj)
