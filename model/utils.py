import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .config import TransConfig

class FeedForwardGELU(nn.Module):
    def __init__(self, config:TransConfig):
        super().__init__()
        
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        
    def forward(self, x:Tensor) -> Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, config:TransConfig):
        super().__init__()
        
        self.eps = config.eps
        self.alpha = nn.Parameter(torch.ones(config.d_model))
        self.bias = nn.Parameter(torch.zeros(config.d_model))
        
    def forward(self, x:Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias