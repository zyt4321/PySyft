# whatever you want to be automatically compiled, import here
from syft.generic.tensor import restricted
from syft.generic.tensor import abstract
from syft.generic.tensor import precision
from syft.generic.tensor import constructors

do_not_compile_modules = set(
    [
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "compile_classes",
        "do_not_compile_modules",
    ]
)



