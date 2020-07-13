from .syft_message import SyftMessage


class GetObjectMessage(SyftMessage):
    def __init__(self, id: str):
        super.__init__(id)
