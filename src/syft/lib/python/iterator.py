# stdlib
from typing import Any
from typing import Sequence
from typing import Iterator as IteratorType
from typing import Type
from typing import Union

# syft relative
from ...core.common.uid import UID
from ...decorators import syft_decorator
from .primitive_interface import PyPrimitive

dict_keyiterator: Type = type(iter({}.keys()))
dict_valueiterator: Type = type(iter({}.values()))
generator: Type = type((lambda: (yield))())


class Iterator(PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(
        self,
        _ref: Union[  # type: ignore
            Sequence, dict_keyiterator, dict_valueiterator, generator
        ],
    ):
        super().__init__()
        self._obj_ref = _ref
        self._index = 0
        self._id = UID()
        self._exhausted = False

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iter__(self) -> "Iterator":
        return self

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __next__(self) -> Any:
        if self._exhausted:
            raise StopIteration

        # https://docs.python.org/3/library/collections.abc.html
        # types that have __len__ and __getitem__
        if issubclass(type(self._obj_ref), Sequence) or (
            hasattr(self._obj_ref, "__len__") and hasattr(self._obj_ref, "__getitem__")
        ):
            if self._index >= len(self._obj_ref):
                self._exhausted = True
                raise StopIteration

            obj = self._obj_ref[self._index]  # type: ignore
            self._index += 1
            return obj

        # https://docs.python.org/3/library/collections.abc.html
        # types that have __next__
        elif issubclass(type(self._obj_ref), IteratorType) or hasattr(
            self._obj_ref, "__next__"
        ):
            try:
                obj = self._obj_ref.__next__()  # type: ignore
                self._index += 1
                return obj
            except StopIteration:
                raise StopIteration
