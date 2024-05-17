import torch as tch
import torch.nn as nn


class UniteEmbedding(nn.Module):
    def __init__(self, emb, grad_cnt):
        super(UniteEmbedding, self).__init__( )
        weight = emb.weight
        self.fixed_weight = nn.Parameter(weight[:-grad_cnt])
        self.grad_weight = nn.Parameter(weight[-grad_cnt:])
        self.fixed_weight.requires_grad = False
        self.grad_weight.requires_grad = True

    def forward(self, x):
        weight = tch.cat([self.fixed_weight, self.grad_weight], dim=0)
        return nn.functional.embedding(x, weight)

