# script to solve Burger's equation numerically to construct the PDE solution datasets for training and sampling

import numpy as np
import h5py
import os
import multiprocessing as mp

def solve_burgers(p_loc, u_bc, u_L=1.0, u_R=0.0, L=1.0, T=1.0, Nx=100, Nt=100, eps=0.02):
    dx, dt = L / Nx, T / Nt
    x = np.linspace(0, L, Nx + 1)
    u = np.zeros((Nt + 1, Nx + 1))
    u[0] = 1.0 / (1 + np.exp((x - p_loc) / eps))  # IC

    def godunov_flux(uL, uR):
        flux = np.zeros_like(uL)
        rarefaction = uL <= uR
        shock = uL > uR
        flux[rarefaction] = np.minimum(0.5 * uL[rarefaction] ** 2, 0.5 * uR[rarefaction] ** 2)
        flux[shock] = np.where((uL[shock] + uR[shock]) / 2 > 0,
                               0.5 * uL[shock] ** 2,
                               0.5 * uR[shock] ** 2)
        return flux

    for n in range(Nt):
        if n > 0:
            u[n, 0] = u_bc 
        flux = godunov_flux(u[n, :-1], u[n, 1:])
        u[n+1, 1:-1] = u[n, 1:-1] - (dt / dx) * (flux[1:] - flux[:-1])
        u[n+1, 0] = u_bc
        u[n+1, -1] = u[n+1, -2]  
    return u.T


def generate_sample(args):
    i_ic, i_bc, p_locs, u_bcs, Nx, Nt = args
    u = solve_burgers(p_locs[i_ic], u_bcs[i_bc], Nx=Nx, Nt=Nt)
    return i_ic, i_bc, u

def generate_burgers_dataset(path, N_ic, N_bc, Nx=100, Nt=100, T=1.0, nproc=48, seed=42, filename="burgers_train"):
    np.random.seed(seed)
    p_locs = np.random.uniform(0.2, 0.8, N_ic)
    u_bcs = np.random.uniform(0.0, 1.0, N_bc)
    x = np.linspace(0, 1.0, Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{filename}_nIC{N_ic}_nBC{N_bc}.h5")

    with h5py.File(full_path, 'w') as f:
        f.create_dataset("u", shape=(N_ic, N_bc, Nx + 1, Nt + 1), dtype=np.float32)
        f.create_dataset("ic", data=p_locs.astype(np.float32))
        f.create_dataset("bc", data=u_bcs.astype(np.float32))
        f.create_dataset("x", data=x.astype(np.float32))
        f.create_dataset("t", data=t.astype(np.float32))

        tasks = [(i, j, p_locs, u_bcs, Nx, Nt) for i in range(N_ic) for j in range(N_bc)]
        with mp.Pool(nproc) as pool:
            for i_ic, i_bc, sol in pool.imap_unordered(generate_sample, tasks):
                f["u"][i_ic, i_bc] = sol.astype(np.float32)

    return full_path


def generate_burgers_dataset_diffBCs(
    path, N_bc=20, N_ic=512, Nx=100, Nt=100, T=1.0, nproc=48, seed=42, filename="burgers_sampling_diffBCs"
):
    np.random.seed(seed)
    u_bcs = np.random.uniform(0.0, 1.0, N_bc)                 
    p_locs = np.random.uniform(0.2, 0.8, N_ic)                  
    x = np.linspace(0, 1.0, Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{filename}_nBC{N_bc}_nIC{N_ic}.h5")

    with h5py.File(full_path, 'w') as f:
        f.create_dataset("u", shape=(N_bc, N_ic, Nx + 1, Nt + 1), dtype=np.float32)
        f.create_dataset("bc", data=u_bcs.astype(np.float32))    
        f.create_dataset("ic", data=p_locs.astype(np.float32))   
        f.create_dataset("x", data=x.astype(np.float32))
        f.create_dataset("t", data=t.astype(np.float32))

        tasks = [(i_ic, i_bc, p_locs, u_bcs, Nx, Nt) for i_bc in range(N_bc) for i_ic in range(N_ic)]

        with mp.Pool(nproc) as pool:
            for i_ic, i_bc, sol in pool.imap_unordered(generate_sample, tasks):
                f["u"][i_bc, i_ic] = sol.astype(np.float32)

    print(f"Saved to {full_path}")
    return full_path

# training data: vary IC and BC 
generate_burgers_dataset(
    path="datasets/data/", N_ic=80, N_bc=80, seed = 42, filename="burgers_train"
)
generate_burgers_dataset(
    path="datasets/data/", N_ic=30, N_bc=30, seed = 0, filename="burgers_test"
)

# sampling data for fixed ICs
generate_burgers_dataset(
    path="datasets/data/", N_ic=20, N_bc=512, Nx=100, Nt=100, seed=42, filename="burgers_sampling_diffICs"
)

# sampling data for fixed BCs
generate_burgers_dataset_diffBCs(
    path="datasets/data/", N_bc=20, N_ic=512, Nx=100, Nt=100, seed=42, filename="burgers_sampling_diffBCs"
)
