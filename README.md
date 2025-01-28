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

### AxialAttention


### PairwiseAttention

### MsaAttention

### InvariantPointAttention