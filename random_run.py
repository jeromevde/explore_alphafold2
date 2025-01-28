#%%
import torch
from alphafold2.alphafold2 import Alphafold2

model = Alphafold2(
    dim = 256,
    depth = 2,
    heads = 8,
    dim_head = 64,
)

seq = torch.randint(0, 21, (1, 128))   # AA length of 128
msa = torch.randint(0, 21, (1, 5, 128))   # MSA doesn't have to be the same length as primary sequence
mask = torch.ones_like(seq).bool()
msa_mask = torch.ones_like(msa).bool()

distogram = model(
    seq,
    msa,
    mask = mask,
    msa_mask = msa_mask
) # (1, 128, 128, 37)

# %%
from torchsummary import summary
summary(model, input_size=(1, 128))

# %%
