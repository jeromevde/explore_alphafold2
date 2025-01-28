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

from outermean import OuterMean

from triangle import TriangleMultiplicativeModule
from axial_attention import AxialAttention

class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.,
        global_column_attn = False
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)

        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True, global_query_attn = global_column_attn)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None,
        msa_mask = None
    ):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask = msa_mask)

        x = self.triangle_multiply_outgoing(x, mask = mask) + x
        x = self.triangle_multiply_ingoing(x, mask = mask) + x
        x = self.triangle_attention_outgoing(x, edges = x, mask = mask) + x
        x = self.triangle_attention_ingoing(x, edges = x, mask = mask) + x
        return x
