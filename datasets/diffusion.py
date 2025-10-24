# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
# heat equation dataset

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from ._base import register_dataset


@register_dataset('diffusion')
class DiffusionDataset(Dataset):
    def __init__(self, split, nx=100, nt=100, visc_range=(1., 5.),
                 phi_range=(0., np.pi), t_range=(0., 1.), n_data=5000):
        super().__init__()
        self.split = split
        self.nx = nx
        self.nt = nt
        self.visc_range = visc_range
        self.phi_range = phi_range
        self.n_data = n_data

        self.xs = torch.linspace(0., 2 * np.pi, nx + 1)[:-1].view(-1, 1)
        self.ts = torch.linspace(*t_range, nt).view(1, -1)

    def __getitem__(self, index):
        v = random.uniform(*self.visc_range)
        phi = random.uniform(*self.phi_range)
        u = torch.sin(self.xs + phi) * torch.exp(-self.ts * v)  # (nx, nt)
        return u

    def __len__(self):
        return self.n_data
