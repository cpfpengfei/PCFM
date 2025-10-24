# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

import io

import torch
from torchvision.transforms import ToTensor
from PIL import Image
from matplotlib import pyplot as plt

plt.switch_backend('agg')


def plt2tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image


def draw_2d(x, vmin=0, vmax=1, cmap='bwr'):
    # x: (nx, nt)
    plt.figure(figsize=(4, 4))
    plt.imshow(x.detach().cpu().numpy(), cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.axis('off')
    return plt2tensor(plt.gcf())


def draw_3d(x, downsample=2, nrow=5, vmin=-2.5, vmax=2.5, cmap='bwr'):
    # x: (nx, ny, nt)
    if downsample > 1:
        idx = torch.tensor(list(range(0, x.size(-1), downsample)), device=x.device, dtype=torch.long)
        x = x[..., idx]
    ncol = x.size(-1) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8))
    fig.subplots_adjust(hspace=0., wspace=0.)
    for ax, im in zip(axes.flat, x.detach().cpu().permute(2, 0, 1)):
        ax.imshow(im, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.axis('off')
    return plt2tensor(plt.gcf())


def draw(x, **kwargs):
    if x.dim() == 2:
        return draw_2d(x, **kwargs)
    if x.dim() == 3:
        return draw_3d(x, **kwargs)
    raise ValueError(f'Unsupported dimension {x.dim()}')
