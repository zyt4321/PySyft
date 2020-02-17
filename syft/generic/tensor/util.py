import functools
import torch as th
import numpy as np


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

framework="Syft"

HANDLED_FUNCTIONS_ABSTRACT = {}

def BaseTensor(framework):
    if framework == 'Torch':
        return th.Tensor
    elif framework == 'Numpy':
        return np.ndarray
    else:
        return object

@torch_only
def method_argument_pre_process(x):
    return x.child

@numpy_only
def method_argument_pre_process(x):
    return np.asarray(x)

def method_return_post_process(result, out=None, obj_type=BaseTensor(framework)):

    if out is None:
        return obj_type.Constructor(result)
    else:
        out.data.set_(result)

    return out

def execute_default_function_on_child_and_wrap(self, func, args, kwargs, types=None):

    if kwargs is None:
        kwargs = {}

    if func not in HANDLED_FUNCTIONS_ABSTRACT:

        new_args = list()
        for arg in args:
            new_args.append(method_argument_pre_process(arg))

        result = func(*new_args, **kwargs)  # TODO: pre_process kwargs too

        result = method_return_post_process(result, obj_type=type(args[0]))

        return result

    if types is not None:
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

    return HANDLED_FUNCTIONS_ABSTRACT[func](*args, **kwargs)


def override_syft_function(torch_function, HANDLED_FUNCTIONS_DICT):
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

        def exec(*args, **kwargs):

            new_args = list()
            for arg in args:
                new_args.append(method_argument_pre_process(arg))

            result = func(*new_args, **kwargs)  # TODO: pre_process kwargs too

            result = method_return_post_process(result, obj_type=type(args[0]))

            return result

        HANDLED_FUNCTIONS_DICT[torch_function] = exec
        return func

    return decorator
