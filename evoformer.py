import torch
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from inspect import isfunction
from functools import partial
from dataclasses import dataclass
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from utils import *
import constants as constants
from mlm import MLM

from evoformer_block import EvoformerBlock


class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(
        self,
        x,
        m,
        mask = None,
        msa_mask = None
    ):
        inp = (x, m, mask, msa_mask)
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m