#%%

import torch
from alphafold2_pytorch import Alphafold2


model = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64,
    reversible = False  # set this to True for fully reversible self / cross attention for the trunk
).cuda()

seq = torch.randint(0, 21, (1, 128)).cuda()      # AA length of 128
msa = torch.randint(0, 21, (1, 5, 120)).cuda()   # MSA doesn't have to be the same length as primary sequence
mask = torch.ones_like(seq).bool().cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

distogram = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
) # (1, 128, 128, 37)
# %%
