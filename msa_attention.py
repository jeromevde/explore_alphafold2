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

from axial_attention import AxialAttention

class MsaAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.
    ):
        super().__init__()
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None
    ):
        x = self.row_attn(x, mask = mask, edges = pairwise_repr) + x
        x = self.col_attn(x, mask = mask) + x
        return x

# main evoformer class