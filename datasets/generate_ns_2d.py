# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

"""
Original code from Cheng et al and also adapted from Li et al, 2020.

Solve Navier-Stokes equations with the Crank-Nicolson method.

"""

import os
import math
import argparse

import h5py
import numpy as np
import torch
from einops import rearrange, repeat
from tqdm import tqdm

from datasets.random_fields import GaussianRF


def solve_navier_stokes_2d(w0, f, visc=1e-3, T=49, delta_t=1e-3, record_steps=50):
    """Solve Navier-Stokes equations in 2D using Crank-Nicolson method.

    Parameters
    ----------
    w0 : torch.Tensor
        Initial vorticity field.

    f : torch.Tensor
        Forcing term.

    visc : float
        Viscosity (1/Re).

    T : float
        Final time.

    delta_t : float
        Internal time-step for solve (descrease if blow-up).

    record_steps : int
        Number of in-time snapshots to record.

    """
    # Grid size - must be power of 2
    N = w0.shape[-1]

    # Maximum frequency
    k_max = math.floor(N / 2)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2], norm='backward')

    # Forcing to Fourier space
    f_h = torch.fft.fftn(f, dim=[-2, -1], norm='backward')

    # If same forcing for the whole batch
    if len(f_h.shape) < len(w_h.shape):
        f_h = rearrange(f_h, '... -> 1 ...')

    # Record solution every this number of steps
    # Also record t = 0
    record_time = math.floor(steps / (record_steps - 1))

    # Wavenumbers in y-direction
    k_y = torch.cat((
        torch.arange(start=0, end=k_max, step=1, device=w0.device),
        torch.arange(start=-k_max, end=0, step=1, device=w0.device)),
        0).repeat(N, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0

    if isinstance(visc, np.ndarray):
        visc = torch.from_numpy(visc).to(w0.device)
        visc = repeat(visc, 'b -> b m n', m=N, n=N)
        lap = repeat(lap, 'm n -> b m n', b=w0.shape[0])

    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(
            torch.abs(k_y) <= (2.0 / 3.0) * k_max,
            torch.abs(k_x) <= (2.0 / 3.0) * k_max
        ).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)
    sol[..., 0] = w0

    # Record counter
    c = 1
    # Physical time
    t = 0.0

    for j in tqdm(range(steps)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        q_real_temp = q.real.clone()
        q.real = -2 * math.pi * k_y * q.imag
        q.imag = 2 * math.pi * k_y * q_real_temp
        q = torch.fft.ifftn(q, dim=[1, 2], norm='backward').real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        v_real_temp = v.real.clone()
        v.real = 2 * math.pi * k_x * v.imag
        v.imag = -2 * math.pi * k_x * v_real_temp
        v = torch.fft.ifftn(v, dim=[1, 2], norm='backward').real

        # Partial x of vorticity
        w_x = w_h.clone()
        w_x_temp = w_x.real.clone()
        w_x.real = -2 * math.pi * k_x * w_x.imag
        w_x.imag = 2 * math.pi * k_x * w_x_temp
        w_x = torch.fft.ifftn(w_x, dim=[1, 2], norm='backward').real

        # Partial y of vorticity
        w_y = w_h.clone()
        w_y_temp = w_y.real.clone()
        w_y.real = -2 * math.pi * k_y * w_y.imag
        w_y.imag = 2 * math.pi * k_y * w_y_temp
        w_y = torch.fft.ifftn(w_y, dim=[1, 2], norm='backward').real

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(q * w_x + v * w_y,
                             dim=[1, 2], norm='backward')

        # Dealias
        F_h *= dealias

        # Cranck-Nicholson update
        factor = 0.5 * delta_t * visc * lap
        num = -delta_t * F_h + delta_t * f_h + (1.0 - factor) * w_h
        w_h = num / (1.0 + factor)

        # Update real time (used only for recording)
        t += delta_t

        if (j + 1) % record_time == 0:
            # Solution in physical space
            w = torch.fft.ifftn(w_h, dim=[1, 2], norm='backward').real
            if w.isnan().any().item():
                raise ValueError('NaN values found.')

            # Record solution and time
            sol[..., c] = w
            sol_t[c] = t

            c += 1

    return sol.cpu().numpy()


def get_random_force(b, s, device, cycles=1, scaling=0.1, t=0., t_scaling=0.2, seed=None):
    ft = torch.linspace(0, 1, s + 1).to(device)
    ft = ft[0:-1]
    X, Y = torch.meshgrid(ft, ft, indexing='ij')
    X = repeat(X, 'x y -> b x y', b=b)
    Y = repeat(Y, 'x y -> b x y', b=b)

    gen = torch.Generator(device)
    gen.manual_seed(seed)

    f = 0
    for p in range(1, cycles + 1):
        k = 2 * math.pi * p

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.sin(k * X + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.cos(k * X + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.sin(k * Y + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.cos(k * Y + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.sin(k * (X + Y) + t_scaling * t)

        alpha = torch.rand(b, 1, 1, generator=gen, device=device)
        f += alpha * torch.cos(k * (X + Y) + t_scaling * t)

    f = f * scaling

    return f


@torch.no_grad()
def navier_stokes(
        root: str,
        nw: int = 100,
        nf: int = 100,
        s: int = 64,
        t: int = 49,
        steps: int = 50,
        mu: float = 1e-3,
        batch_size: int = 1024,
        seed: int = 42,
        delta: float = 1e-3,
):
    device = torch.device('cuda')
    torch.manual_seed(seed)
    np.random.seed(seed + 1234)

    if os.path.dirname(root):
        os.makedirs(os.path.dirname(root), exist_ok=True)
    path = os.path.join(root, f'ns_nw{nw}_nf{nf}_s{s}_t{steps}_mu{mu}.h5')

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

    data_f = h5py.File(path, 'a')

    data_f.create_dataset('a', (nw, s, s), np.float32)
    data_f.create_dataset('f', (nf, s, s), np.float32)
    data_f.create_dataset('u', (nw, nf, s, s, steps), np.float32)

    w0 = GRF.sample(nw)
    ft = torch.linspace(0, 1, s + 1, device=w0.device)
    ft = ft[0:-1]
    X, Y = torch.meshgrid(ft, ft, indexing='ij')
    phi = np.pi / 2 * torch.linspace(0, 1, nf, dtype=torch.float, device=device)
    fs = 0.1 * np.sqrt(2) * torch.sin(
        2 * math.pi * (X + Y).unsqueeze(0) + phi.view(-1, 1, 1)
    )
    # fs = get_random_force(nf, s, device, seed=seed)
    data_f['a'][:] = w0.cpu().numpy()
    data_f['f'][:] = fs.cpu().numpy()

    sols = []
    w0 = w0.unsqueeze(1).repeat(1, nf, 1, 1).view(-1, s, s)
    fs = fs.unsqueeze(0).repeat(nw, 1, 1, 1).view(-1, s, s)
    n_batch = math.ceil(w0.size(0) / batch_size)
    i = 0
    for w, f in zip(torch.split(w0, batch_size, dim=0), torch.split(fs, batch_size, dim=0)):
        i += 1
        print(f'Batch {i} / {n_batch}')
        sol = solve_navier_stokes_2d(
            w, f, mu, t, delta, steps
        )
        sols.append(sol)
    sols = np.concatenate(sols, axis=0).reshape(nw, nf, s, s, steps)
    data_f['u'][:] = sols
    print(f'Done. Dataset saved to {path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incompressible Navier-Stokes data generation script.")
    parser.add_argument('--root', type=str, default='.', help='root to save the data')
    parser.add_argument('--nw', type=int, default=100,
                        help='Number of initial vorticity fields to generate')
    parser.add_argument('--nf', type=int, default=100,
                        help='Number of forcing fields to generate')
    parser.add_argument('--s', type=int, default=64,
                        help='Width of the solution grid')
    parser.add_argument('--t', type=int, default=49,
                        help='Final time step')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of snapshots from solution')
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='Viscosity')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Batch size for solver')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed value for reproducibility')
    parser.add_argument('--delta', type=float, default=1e-3,
                        help='Internal time step for solver')
    args = parser.parse_args()
    navier_stokes(
        args.root, args.nw, args.nf, args.s, args.t,
        args.steps, args.mu, args.batch_size, args.seed, args.delta
    )
