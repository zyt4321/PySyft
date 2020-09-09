# stdlib
from typing import Optional

# syft relative
from ...core.common import UID
from .primitive_interface import PyPrimitive


class SyNone(PyPrimitive):
    def __init__(self, uid: Optional[UID] = None):
        PyPrimitive.__init__(self, uid if uid is not None else UID())
