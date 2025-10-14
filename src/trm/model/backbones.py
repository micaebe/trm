import torch.nn as nn
import torch.nn.functional as F
from trm.model.utils import RotaryEmbedding, apply_rotary_pos_emb


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden_dim = dim * mult
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head

        self.rotary_emb = RotaryEmbedding(dim_head)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        B, N, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)

        cos, sin = self.rotary_emb(q)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class MlpMixer(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Linear(seq_len, seq_len)
        ) 
        # channel mixing/ffn is applied inside recursive block inside trm.py

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        return self.token_mixer(x.transpose(1, 2)).transpose(1, 2)