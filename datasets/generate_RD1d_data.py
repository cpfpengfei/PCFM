# script to solve the Reaction-Diffusion equation numerically to construct the PDE solution datasets for training and sampling

import numpy as np
import h5py
import os
import multiprocessing as mp
from functools import partial

rho, nu = 0.01, 0.005
Nx = 128
xL, xR = 0.0, 1.0
dt_save = 0.01
ini_time, fin_time = 0.0, 1.0 - dt_save
CFL = 0.25

x = np.linspace(xL, xR, Nx, endpoint=False) + (xR - xL) / Nx * 0.5
dx = x[1] - x[0]
nt = int(np.ceil((fin_time - ini_time) / dt_save)) + 1
t_grid = np.linspace(ini_time, fin_time, nt)

def solve_single(u0, gL, gR):
    u = u0.copy()
    sol = np.zeros((nt, Nx))
    sol[0] = u
    t = ini_time
    save_idx = 1

    def Courant_diff(dx, epsilon=1.0e-3):
        return 0.5 * dx**2 / (epsilon + 1e-8)

    def flux_numpy(u):
        N = u.size
        _u = np.zeros(N + 4)
        _u[2:2 + N] = u
        f = -nu * (_u[2:N + 3] - _u[1:N + 2]) / dx
        f[0] = gL
        f[-1] = gR
        return f

    def update(u, u_tmp, dt):
        stiff = 1.0 / (1.0 + np.exp(-rho * dt) * (1.0 - u) / (u + 1e-12))
        f = flux_numpy(u_tmp)
        return stiff - dt * (f[1:u.size + 1] - f[0:u.size]) / dx

    while t < fin_time and save_idx < nt:
        dt = min(Courant_diff(dx, nu) * CFL, fin_time - t, t_grid[save_idx] - t)
        if dt <= 1e-8:
            break
        u_tmp = update(u, u, 0.5 * dt)
        u = update(u, u_tmp, dt)
        t += dt
        if t >= t_grid[save_idx] - 1e-12:
            sol[save_idx] = u
            save_idx += 1
    sol[-1] = u
    return sol

def generate_ic(xc, k_tot=3, num_choice_k=2):
    selected = np.random.choice(k_tot, size=(num_choice_k,), replace=True)
    onehot = np.zeros(k_tot, dtype=int)
    for j in selected:
        onehot[j] += 1
    kk = 2 * np.pi * np.arange(1, k_tot + 1) * onehot / (xc[-1] - xc[0])
    amp = np.random.rand(k_tot, 1)
    phs = 2 * np.pi * np.random.rand(k_tot, 1)
    u = (amp * np.sin(kk[:, None] * xc[None, :] + phs)).sum(axis=0)
    if np.random.rand() < 0.1:
        u = np.abs(u)
    u *= np.random.choice([1, -1])
    if np.random.rand() < 0.1:
        xL = np.random.uniform(0.1, 0.45)
        xR = np.random.uniform(0.55, 0.9)
        trns = 0.01
        mask = 0.5 * (np.tanh((xc - xL) / trns) - np.tanh((xc - xR) / trns))
        u *= mask
    u -= u.min()
    if u.max() > 0:
        u /= u.max()
    return u

def worker(args):
    i_ic, i_bc, ic_array, bc_array = args
    u0 = ic_array[i_ic]
    gL, gR = bc_array[i_bc]
    sol = solve_single(u0, gL, gR)
    print(f"Finished: IC #{i_ic}, BC #{i_bc}, gL={gL:.3f}, gR={gR:.3f}")
    return i_ic, i_bc, sol.T

def run_parallel(root, N_ic=80, N_bc=80, nproc=48, seed=42, filename="RD_neumann_train"):
    np.random.seed(seed)
    ic_array = [generate_ic(x) for _ in range(N_ic)]
    bc_array = [(np.round(0.05 * np.random.rand(), 3), np.round(-0.05 * np.random.rand(), 3)) for _ in range(N_bc)]

    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'{filename}_nIC{N_ic}_nBC{N_bc}.h5')

    with h5py.File(path, 'w') as f:
        f.create_dataset('u', shape=(N_ic, N_bc, Nx, nt), dtype=np.float32)
        f.create_dataset('ic', data=np.stack(ic_array), dtype=np.float32)
        f.create_dataset('bc', data=np.array(bc_array), dtype=np.float32)
        f.create_dataset('x', data=x, dtype=np.float32)
        f.create_dataset('t', data=t_grid, dtype=np.float32)
        f.attrs['rho'] = rho
        f.attrs['nu'] = nu

        tasks = [(i, j, ic_array, bc_array) for i in range(N_ic) for j in range(N_bc)]
        with mp.Pool(processes=nproc) as pool:
            for i_ic, i_bc, sol in pool.imap_unordered(worker, tasks):
                f['u'][i_ic, i_bc] = sol.astype(np.float32)
    return path


# training data: vary IC and BC 
run_parallel(root='datasets/data/', N_ic=80, N_bc=80, nproc=48, seed=42, filename="RD_neumann_train")
run_parallel(root='datasets/data/', N_ic=30, N_bc=30, nproc=48, seed=0, filename="RD_neumann_test")

# sampling data for fixed ICs
run_parallel(root='datasets/data/', N_ic=20, N_bc=512, nproc=48, seed=42, filename="RD_sampling_diffICs")

