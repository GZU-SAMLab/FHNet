from abc import ABC
import torch
import math
from torch import nn, einsum
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F


class PreNorm(nn.Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        device = self.norm.weight.device
        x = x.to(device)
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module, ABC):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    

class Share_Attention(nn.Module, ABC):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., alpha=0.5):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.alpha = alpha

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2):
        h = self.heads
        h1 = int(h * self.alpha)
        h2 = int(h * (1 - self.alpha))

        b, n1, _ = x1.shape
        b2, n2, _ = x2.shape  
        assert b == b2, "need same batch size for x1 and x2"

        qkv1 = self.to_qkv(x1).chunk(3, dim=-1)
        qkv2 = self.to_qkv(x2).chunk(3, dim=-1)
        
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h1), qkv1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h2), qkv2)

        dots1 = torch.einsum('b h i d, b h j d -> b h i j', q1, k1) * self.scale
        dots2 = torch.einsum('b h i d, b h j d -> b h i j', q2, k2) * self.scale

        attn1 = F.softmax(dots1, dim=-1)
        attn2 = F.softmax(dots2, dim=-1)
        
        out1 = torch.einsum('b h i j, b h j d -> b h i d', attn1, v1)
        out2 = torch.einsum('b h i j, b h j d -> b h i d', attn2, v2)

        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        
        out1 = self.to_out(out1)
        out2 = self.to_out(out2)

        return out1, out2
    