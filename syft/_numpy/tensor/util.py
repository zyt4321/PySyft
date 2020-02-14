import functools


def override_numpy_function(torch_function, HANDLED_FUNCTIONS_DICT):
    """Register a torch function override.

    PyTorch's functions are type sensitive. This means that it expects to run
    different functionality for any method depending on the type of tensor
    that you pass into tha method. This function allows us to register
    a new type for a torch function (torch_function) for a specific
    new tensor type which we are creating. The list of functions we are
    overriding is stored in a global dictionary HANDLED_FUNCTIONS_DICT
    which is passed to our new tensor type when the class is defined.

    The overrides then occur within the method __torch_function__ within
    our new custom tensor type.

    Args:
        torch_function (func): the function we want to override
        HANDLED_FUNCTIONS_DICT (dict): the global torch_function->new_function
            mapping.
    """

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_DICT[torch_function] = func
        return func

    return decorator


def torch_only(func):
    """Tells compiler that this function should only be included in the torch
    generated code.
    """

    @functools.wraps(func)
    def decorator(func):
        return func

    return decorator


def numpy_only(func):
    """Tells compiler that this function should only be included in the torch
    generated code.
    """

    @functools.wraps(func)
    def decorator(func):
        return func

    return decorator


def tensorflow_only(func):
    """Tells compiler that this function should only be included in the torch
    generated code.
    """

    @functools.wraps(func)
    def decorator(func):
        return func

    return decorator


def jax_only(func):
    """Tells compiler that this function should only be included in the torch
    generated code.
    """

    @functools.wraps(func)
    def decorator(func):
        return func

    return decorator
