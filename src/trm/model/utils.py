import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    def update(self, model: nn.Module):
        with torch.no_grad():
            for shadow_p, current_p in zip(self.shadow_params, [p for p in model.parameters() if p.requires_grad]):
                shadow_p.data = self.decay * shadow_p.data + (1.0 - self.decay) * current_p.data

    def apply_shadow(self, model: nn.Module):
        original_params = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
        for shadow_p, (_, p) in zip(self.shadow_params, [(n, p) for n, p in model.named_parameters() if p.requires_grad]):
            p.data.copy_(shadow_p.data)
        return original_params

    def restore_original(self, model: nn.Module, original_params: dict):
        for name, p in model.named_parameters():
            if name in original_params:
                p.data.copy_(original_params[name])


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)