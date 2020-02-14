from syft.generic.tensor import abstract, precision

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
