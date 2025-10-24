# Reaction-Diffusion dataset

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from ._base import register_dataset

@register_dataset('rd1d')
class RD1DDataset(Dataset):
    def __init__(self, root, split, data_file):
        self.root = root
        self.split = split
        self.file = h5py.File(os.path.join(root, data_file), 'r')

        # u has shape (N_ic, N_bc, Nx, nt)
        self.u = self.file['u']
        self.N_ic, self.N_bc, self.Nx, self.nt = self.u.shape
        self.n_data = self.N_ic * self.N_bc

    def __del__(self):
        self.file.close()

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        i_ic, i_bc = divmod(index, self.N_bc)
        arr = self.u[i_ic, i_bc]
        return torch.from_numpy(arr.astype(np.float32))