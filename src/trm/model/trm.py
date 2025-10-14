import torch
import torch.nn as nn
from trm.model.backbones import Attention, MlpMixer, FeedForward


class RecursiveBlock(nn.Module):
    def __init__(self, dim: int, seq_len: int, use_attention: bool = True, heads: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        if use_attention:
            self.mixer = Attention(dim, heads=heads)
        else:
            self.mixer = MlpMixer(seq_len)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        x = self.norm1(x + self.mixer(x))
        x = self.norm2(x + self.ffn(x))
        return x


class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        vocab_size: int,
        n_layers: int = 2,
        n_recursions: int = 6,
        T_recursions: int = 3,
        use_attention: bool = True
    ):
        super().__init__()
        assert n_recursions >= 1, "n_recursion must be at least 1"
        assert T_recursions > 1, "T_recursion must be greater than 1"
        self.dim = dim
        self.n = n_recursions
        self.T = T_recursions

        self.input_embedding = nn.Embedding(vocab_size, dim)
        self.output_head = nn.Linear(dim, vocab_size)
        self.net = nn.Sequential(
            *[RecursiveBlock(dim, seq_len, use_attention) for _ in range(n_layers)]
        )
        self.q_head = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def latent_recursion(self, x, y, z):
        for _ in range(self.n):
            z = self.net(x + y + z)
        y = self.net(y + z)
        return y, z

    def forward(self, x_embed, y, z):
        # x_embed, y, z shape: (batch, seq_len, dim)
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.latent_recursion(x_embed, y, z)
        y, z = self.latent_recursion(x_embed, y, z)

        y_logits = self.output_head(y)
        q = self.q_head(y.mean(dim=1))
        return y, z, y_logits, q