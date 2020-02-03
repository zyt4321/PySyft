def args2child(self, *args, **kwargs):

    new_args = list()
    new_kwargs = {}

    self = self.child
    self.set_attr("end", False)
    for arg in args:
        if hasattr(arg, "child"):
            new_args.append(arg.child)
        else:
            new_args.append(arg)

    for key, arg in kwargs.items():
        if hasattr(arg, "child"):
            new_kwargs[key] = arg.child
        else:
            new_kwargs[key] = arg

    return self, new_args, new_kwargs


def args2data(self, *args, **kwargs):

    new_args = list()
    new_kwargs = {}

    self = self.data

    self.set_attr("end", True)
    for arg in args:
        if hasattr(arg, "data"):
            new_args.append(arg.data)
        else:
            new_args.appends(arg)

    for key, arg in kwargs.items():
        if hasattr(arg, "data"):
            new_kwargs[key] = arg.data
        else:
            new_kwargs[key] = arg

    return self, new_args, new_kwargs


def chain_method(func):
    def inner(self, *args, **kwargs):

        chain_end = False

        # if self is in the middle of a chain,
        # assume all arguments are either at the
        # same position or are not part of a tensor
        # chain at all (such as booleans)
        if self.child is not None:

            self, args, kwargs = args2child(self, *args, **kwargs)

        else:

            self, args, kwargs = args2data(self, *args, **kwargs)

            chain_end = True

        result = func(self, *args, **kwargs)

        if chain_end:
            self.set_attr("end", False)

        return result

    return inner
