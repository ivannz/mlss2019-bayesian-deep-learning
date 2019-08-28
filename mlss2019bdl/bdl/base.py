from torch.nn import Module


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


class PenalizedWeight(Module):
    def penalty(self):
        raise NotImplementedError()


def penalties(module):
    for mod in module.modules():
        if isinstance(mod, PenalizedWeight):
            yield mod.penalty()
