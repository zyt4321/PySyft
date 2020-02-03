def chain_method(func):
    def inner(self, *args, **kwargs):

        new_args = list()

        new_kwargs = {}

        # if self is in the middle of a chain,
        # assume all arguments are either at the
        # same position or are not part of a tensor
        # chain at all (such as booleans)
        if self.child is not None:

            self = self.child

            for arg in args:
                if (hasattr(arg, 'child')):
                    new_args.append(arg.child)
                else:
                    new_args.append(arg)

            for key, arg in kwargs.items():
                if (hasattr(arg, 'child')):
                    new_kwargs[key] = arg.child
                else:
                    new_kwargs[key] = arg

        else:

            self = self.data

            for arg in args:
                if (hasattr(arg, 'data')):
                    new_args.append(arg.data)
                else:
                    new_args.append(arg)

            for key, arg in kwargs.items():
                if (hasattr(arg, 'child')):
                    new_kwargs[key] = arg.data
                else:
                    new_kwargs[key] = arg

        result = func(self, *new_args, **kwargs)

        return result

    return inner