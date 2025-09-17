from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

class embedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim = 512, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    
    def forward(self, x):
        B, N, D = x.shape
        Q = self.Q(x).reshape(B, N, self.heads, self.dim_head).transpose(1,2)
        K = self.K(x).reshape(B, N, self.heads, self.dim_head).transpose(1,2)
        V = self.V(x).reshape(B, N, self.heads, self.dim_head).transpose(1,2)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / sqrt(self.dim_head)
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p = self.dropout, training=self.training)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        out = self.proj(out)
        return out

class FFN(nn.Module):
    def __init__(self, dim = 512, mlp_dim=2048, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim), 
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, dim = 512, dim_head = 64, mlp_dim = 2048, dropout = 0.0, heads = 8):
        super().__init__()
        self.attn = MultiHeadAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FFN(dim=dim, mlp_dim=mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn = self.attn(x)
        x = x + attn
        x = self.norm1(x)
        ff = self.ff(x)
        x = x + ff
        x = self.norm2(x)
        return x

class ViT(nn.Module):
    def __init__(self, dim = 512, heads = 8, mlp_dim = 2048, num_classes = 100):
        super().__init__()
        self.embedding = embedding()
        self.cls = nn.Parameter(torch.randn((1, 1, dim)))
        self.pos = nn.Parameter(torch.randn((1, 65, dim)))
        self.blocks = nn.ModuleList([TransformerBlock(dim = dim, heads = heads, mlp_dim = mlp_dim) for i in range(12)])
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)
        

    def forward(self, x):
        x = self.embedding(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 0]
        return self.mlp_head(x) 







        
    



    