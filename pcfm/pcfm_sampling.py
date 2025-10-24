# Key algorithms for Physics-Constrained Flow Matching (PCFM)

import torch
from torch.func import vmap, jacrev
import gc
from typing import Callable, Sequence

def compute_jacobian(fn: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    def fn_flat(x: torch.Tensor) -> torch.Tensor:
        return fn(x).flatten()
    J = jacrev(fn_flat)(inputs)
    m = J.shape[0]
    n = inputs.numel()
    return J.reshape(m, n)


def fast_project_batched(xi_batch: torch.Tensor, h_func: Callable[[torch.Tensor], torch.Tensor], max_iter: int = 1) -> torch.Tensor:
    """
    Final projection step in PCFM
    """
    B, n = xi_batch.shape

    def newton_step(u, xi):
        h_val = h_func(u)
        if h_val.ndim == 1:
            h_val = h_val.unsqueeze(-1)
        J = jacrev(h_func,chunk_size=len(h_val)//4)(u)
        if J.ndim == 1:
            J = J.unsqueeze(0)
        delta = (xi - u).unsqueeze(-1)
        JJt = J @ J.transpose(-2, -1)
        rhs = J @ delta + h_val
        lambda_ = torch.linalg.solve(JJt, rhs)
        du = delta - J.transpose(-2, -1) @ lambda_
        return u + du.squeeze(-1)

    def loop(xi):
        u = xi.clone()
        gc.collect()
        torch.cuda.empty_cache()
        for _ in range(max_iter):
            u = newton_step(u, xi)
        return u

    return vmap(loop)(xi_batch)

def fast_project_batched_chunk(xi_batch, h_func, max_iter=1, chunk_size=16):
    B, n = xi_batch.shape
    results = []
    for start in range(0, B, chunk_size):
        xi_chunk = xi_batch[start:start + chunk_size]

        def newton_step(u, xi):
            h_val = h_func(u)
            if h_val.ndim == 1:
                h_val = h_val.unsqueeze(-1)
            J = jacrev(h_func, chunk_size=max(1, len(h_val)//4))(u)
            if J.ndim == 1:
                J = J.unsqueeze(0)
            delta = (xi - u).unsqueeze(-1)
            JJt = J @ J.transpose(-2, -1)
            rhs = J @ delta + h_val
            lambda_ = torch.linalg.solve(JJt, rhs)
            du = delta - J.transpose(-2, -1) @ lambda_
            return u + du.squeeze(-1)

        def loop(xi):
            u = xi.clone()
            for _ in range(max_iter):
                u = newton_step(u, xi)
            return u

        results.append(vmap(loop)(xi_chunk))
        del xi_chunk
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)


def make_grid(dims: tuple[int], device='cpu', start: float | tuple[float] = 0., end: float | tuple[float] = 1.):
    ndim = len(dims)
    if not isinstance(start, (tuple, list)):
        start = [start] * ndim
    if not isinstance(end, (tuple, list)):
        end = [end] * ndim
    if ndim == 1:
        return torch.linspace(start[0], end[0], dims[0], dtype=torch.float, device=device).unsqueeze(-1)
    xs = torch.meshgrid([
        torch.linspace(start[i], end[i], dims[i], dtype=torch.float, device=device)
        for i in range(ndim)
    ], indexing='ij')
    grid = torch.stack(xs, dim=-1).view(-1, ndim)
    return grid


def relaxed_penalty_constraint_interp_linear_detached(
    u0, u1_proj, v_flat, t, dt, hfunc, lam=1e-2, step_size=1e-2, num_steps=10, safe_clamp=1e-3
):
    """
    Relaxed constraint correction step in PCFM algorithm
    Solves:
        min_u ||u - hat_u(t')||^2 + lam * ||h(u + gamma * v_flat)||^2
    Args:
        u0: Tensor (n,)
        u1_proj: Tensor (n,)
        v_flat: Tensor (n,), vector field at current state
        t: scalar float (flow matching time t)
        dt: scalar float 
        hfunc: constraint residual
        lam: penalty coefficient
        step_size: gradient descent step size
        num_steps: gradient descent iterations
        safe_clamp: minimum value for gamma
    Returns:
        u_corr: Tensor (n,) 
    """
    t_prime = t + dt
    gamma = max(1 - t_prime, safe_clamp)
    hat_u = (1 - t_prime) * u0 + t_prime * u1_proj
    u = hat_u.detach().clone().requires_grad_(True)

    for _ in range(num_steps):
        u_ext = u + gamma * v_flat
        penalty = hfunc(u_ext).pow(2).sum()
        loss = (u - hat_u).pow(2).sum() + lam * penalty
        grad = torch.autograd.grad(loss, u)[0]
        u = (u - step_size * grad).detach().clone().requires_grad_(True)

    return u.detach()


# 
def pcfm_sample(
    u_flat, v_flat, t, u0_flat, dt, hfunc,
    mode='root', newtonsteps=1, eps=1e-6,
    guided_interpolation = False, interpolation_params = {}
):
    """
    PCFM sampling on a pretrained flow-based model to satisify hard constraints defined in hfunc 
    """
    ut1 = u_flat + (1.0 - t) * v_flat
    u_corr = ut1.clone()

    for _ in range(newtonsteps):
        res = hfunc(u_corr)
        J = compute_jacobian(hfunc, u_corr)
        JJt = J @ J.T
        rhs = res

        if mode == 'least_squares':
            delta = (ut1 - u_corr).unsqueeze(-1)
            rhs = J @ delta + res.unsqueeze(-1)
            rhs = rhs.squeeze(-1)

        lam = torch.linalg.solve(
            JJt + eps * torch.eye(JJt.shape[0], device=u_flat.device), rhs
        )
        u_corr = u_corr - J.T @ lam

    t_next = t + dt

    if guided_interpolation:
        if interpolation_params != {}:
            custom_lam = interpolation_params['custom_lam']
            step_size = interpolation_params['step_size']
            num_steps = interpolation_params['num_steps']
        else:
            custom_lam = 1e0
            step_size = 1e-2
            num_steps = 20
            
        ut_interp = relaxed_penalty_constraint_interp_linear_detached(
            u0=u0_flat,
            u1_proj=u_corr,
            v_flat=v_flat,
            t=t.item(),
            dt=dt,
            hfunc=hfunc,
            lam=custom_lam,
            step_size=step_size,
            num_steps=num_steps
        )
    else:
        ut_interp = (1.0 - t_next) * u0_flat + t_next * u_corr
    proj_vf = ((ut_interp - u_flat) / dt).detach()
    return proj_vf


def pcfm_batched(ut, vf, t, u0, dt, hfunc, use_vmap=False, mode='root', newtonsteps=1, guided_interpolation=False, interpolation_params={}, eps=1e-6):
    """
    Batched PCFM projection for 1D problems (nx, nt)
    """
    B, nx, nt = ut.shape
    n = nx * nt

    def wrapped_project(u_flat, v_flat, u0_flat):
        return pcfm_sample(
            u_flat, v_flat, t, u0_flat, dt,
            hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
            guided_interpolation=guided_interpolation,
            interpolation_params=interpolation_params,
            eps=eps
        )

    u_flat = ut.view(B, n).detach().clone().requires_grad_(True)
    v_flat = vf.view(B, n)
    u0_flat = u0.view(B, n)

    if use_vmap:
        v_proj_flat = vmap(wrapped_project)(u_flat, v_flat, u0_flat)
    else:
        v_proj_list = []
        for i in range(B):
            v_proj = wrapped_project(u_flat[i], v_flat[i], u0_flat[i])
            v_proj_list.append(v_proj)
        v_proj_flat = torch.stack(v_proj_list, dim=0)

    return v_proj_flat.view(B, nx, nt)


def pcfm_2d_batched(ut, vf, t, u0, dt, hfunc, mode='root', newtonsteps=1, guided_interpolation = True, interpolation_params={}, eps=1e-6):
    """
    Batched PCFM projection for 2D problems (nx, ny, nt)
    """
    B, nx, ny, nt = ut.shape
    n = nx * ny * nt

    gc.collect()
    torch.cuda.empty_cache()

    def wrapped_project(u_flat, v_flat, u0_flat):
        return pcfm_sample(
            u_flat, v_flat, t, u0_flat, dt,
            hfunc=hfunc, mode=mode, newtonsteps=newtonsteps, 
            guided_interpolation=guided_interpolation, 
            interpolation_params=interpolation_params, 
            eps=eps
        )

    u_flat = ut.view(B, n).detach().clone().requires_grad_(True)
    v_flat = vf.view(B, n)
    u0_flat = u0.view(B, n)

    # v_proj_flat = vmap(wrapped_project)(u_flat, v_flat, u0_flat) 
    # prevent OOM: 
    v_proj_list = []
    for i in range(u_flat.shape[0]):
        v_proj = wrapped_project(u_flat[i], v_flat[i], u0_flat[i])
        v_proj_list.append(v_proj)
    v_proj_flat = torch.stack(v_proj_list, dim=0)
    return v_proj_flat.view(B, nx, ny, nt)

