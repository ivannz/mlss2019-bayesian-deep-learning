import torch

from torch.nn import Module
from inspect import signature


class FreezableWeight(Module):
    def __init__(self):
        super().__init__()
        self.unfreeze()

    def unfreeze(self):
        self.register_buffer("frozen_weight", None)

    def is_frozen(self):
        """Check if a frozen weight is available."""
        return isinstance(self.frozen_weight, torch.Tensor)

    def freeze(self):
        """Sample from the distribution and freeze."""
        raise NotImplementedError()


def freeze(module):
    for mod in module.modules():
        if isinstance(mod, FreezableWeight):
            mod.freeze()

    return module  # return self


def unfreeze(module):
    for mod in module.modules():
        if isinstance(mod, FreezableWeight):
            mod.unfreeze()

    return module  # return self


def check_defaults(fn):
    arguments = [p for p in signature(fn).parameters.values()
                 if p.name != "self" and p.kind != p.VAR_KEYWORD]

    missing = [p.name for p in arguments if p.default is p.empty]
    if missing:
        fn_name = getattr(fn, "__qualname__", fn.__name__)
        raise TypeError(f"`{fn_name}`: no default(s) for `{missing}`.")


class PenalizedWeight(Module):
    def __init_subclass__(cls, **kwargs):
        # enforce defaults on explicit parameters of `.penalty`
        check_defaults(cls.penalty)
        super().__init_subclass__(**kwargs)

    def penalty(self):
        raise NotImplementedError()


def named_penalties(module, hyperparameters=None, prefix=""):
    if hyperparameters is None:
        hyperparameters = {}

    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, PenalizedWeight):
            par = hyperparameters.get(name, {})
            yield name, mod.penalty(**par)


def penalties(module):
    for name, penalty in named_penalties(module):
        yield penalty
