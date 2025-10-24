# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
# Navier-Stokes dataset

import os

import h5py
import torch
from torch.utils.data import Dataset

from ._base import register_dataset


@register_dataset('ns')
class NavierStokesDataset(Dataset):
    def __init__(self, root, split, data_file):
        self.root = root
        self.split = split
        self.data_file = data_file
        self.file = h5py.File(os.path.join(root, data_file), 'r')

        self.data = self.file['u']  # (nw, nf, s, s, t)
        self.nw, self.nf, self.s, _, self.t = self.data.shape
        self.n_data = self.nw * self.nf

    def __del__(self):
        self.file.close()

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        w, f = divmod(index, self.nf)
        data = torch.from_numpy(self.data[w, f])
        return data
