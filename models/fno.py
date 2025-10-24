# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
# Fourier Neural Operator (FNO) code forked from Cheng et al.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuralop.models import FNO as _FNO

from ._base import register_model


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from Ho et al., 2020.
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = np.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


@register_model('fno')
class FNO(nn.Module):
    def __init__(self, n_modes, emb_channels=16, hidden_channels=32, proj_channels=128, n_layers=4):
        super().__init__()
        self.n_modes = n_modes
        self.emb_channels = emb_channels
        self.hidden_channels = hidden_channels
        self.proj_channels = proj_channels
        self.n_layers = n_layers
        self.model = _FNO(
            n_modes=n_modes, hidden_channels=hidden_channels,
            in_channels=1 + len(n_modes) + emb_channels, out_channels=1,
            lifting_channels=proj_channels, proj_channels=proj_channels, n_layers=n_layers,
        )

    def forward(self, t, u):
        # u: (B, ...)
        # t: either scalar or (B,)

        batch_size, *dims = u.size()
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(batch_size, device=t.device) * t

        t_emb = get_time_embedding(t, self.emb_channels).view(
            [batch_size, -1] + [1] * len(dims)
        ).expand(-1, -1, *dims)
        pos_emb = torch.stack(torch.meshgrid(
            [torch.linspace(0, 1, n, device=u.device) for n in dims], indexing='ij'
        ), dim=0).unsqueeze(0).expand(batch_size, -1, *dims)
        u = torch.cat([u.unsqueeze(1), t_emb, pos_emb], dim=1)
        out = self.model(u).squeeze(1)
        return out

