import os
import random

import torch
import torch.nn as nn

from tests import PACKAGE_ROOT


def test_paths():
    assert os.path.isdir(PACKAGE_ROOT)


def get_a_text():
    length = random.randint(5, 10)
    return torch.randint(0, 100, (length,))


def get_a_mel():
    length = random.randint(10, 20)
    return torch.rand(80, length).clamp(min=1e-3).log()


def get_a_text_mel_pair():
    return get_a_text(), get_a_mel()


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
