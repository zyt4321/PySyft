# syft relative
from ...core.common import UID


class PyPrimitive:
    def __init__(self, uid: UID) -> None:
        self._id: UID = uid
