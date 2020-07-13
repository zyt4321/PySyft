from .syft_message import SyftMessage


class DeleteObjectMessage(SyftMessage):
    def __init__(self, id: str):
        super().__init__(id=id)
