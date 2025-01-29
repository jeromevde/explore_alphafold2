# Alhpafold2

The aim of this repo is to be a minimal exploration of the alphafold architecture, and just that, no training, inference, dataset or other code

## Install


Code adapted from [lucidrains/alhpafold2](https://github.com/lucidrains/alphafold2)

```
pip install -r .devcontainer/requirements.txt
```

or just open the repo in a [github codespace](https://github.com/features/codespaces)



## Background

<details>


# pytorch 

<detail>

https://www.blopig.com/blog/2021/07/alphafold-2-is-here-whats-behind-the-structure-prediction-miracle/



Peformed on random data to test the functionality of the code as alhpafold requires quite a bit of data for the sequence alignment

![alt text](images/model.png)



## Multiple Sequence Alignment MSA

![msa](images/MSA_wikipedia.png)

### Co-evolution
![alt text](images/coevolution.png)

## Pair representation

## Evo former

![alt text](images/evoformer.png)

## Structure module

</details>


## Code architecture

```
EvoFormer
    - EvoFormerBlock
        - PairwiseAttention
            - AxialAttention --> (Attention & LayerNorm)
            - TriangleMultiplicativeModule
        - MsaAttention
            - AxialAttention --> (Attention & LayerNorm)
Structure Module
    - InvariantPointAttention
```

### ```attention.py``` Gated attention with mask & dropout

$
\begin{aligned}
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T + \text{Bias}}{\sqrt{d_k}} + \text{Mask}\right) \cdot V \\
\text{Out} &= \text{Attention}(Q, K, V) \\
\text{Gated Out} &= \text{Out} \odot \sigma(\text{Gating}(X)) \\
\text{Final Out} &= \text{Linear}(\text{Gated Out})
\end{aligned}
$

- ``` inner_dim = dim_head * heads ``` as the q,k,v are passed through a linear layer, the dimension of the heads can be chosen as long as the linear layers project to that specific dimension. The inner dimension is then derived from that.
- ``` self.scale = dim_head ** -0.5. ```
- gating applied to the attention output based on input sequence
- ``` q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1)) ```. The key and value (k,v) are treated together in the general context and q being the input sequence, but in the case of self attention they all derive form a transformation of the input sequence.
- ```tie_dim```
- ```attn_bias```
- ```attn = self.dropout(attn)``` applied to the attention scores
- masking



### AxialAttention


### PairwiseAttention

### MsaAttention

### InvariantPointAttention