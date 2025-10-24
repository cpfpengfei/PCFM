# Core sampler module to implement PCFM, vanilla flow matching, ECI, DiffusionPDE's guided sample, and D-Flow sampling

import gc
import torch
from tqdm import tqdm
from torchdiffeq import odeint
from .pcfm_sampling import make_grid, pcfm_batched, pcfm_2d_batched


class FFM_sampler:
    """
    Collection of samplers using a pretrained functional flow matching model 
    """
    def __init__(self, model, gp):
        self.model = model
        self.gp = gp

    def pcfm_sample(self, u0, n_step, hfunc, mode='root', newtonsteps=1, eps=1e-6,
                    guided_interpolation=True, interpolation_params={}, use_vmap=False):
        """
        PCFM sampler
        """
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="PCFM sampling"):
            vf = self.model(t, u)
            v_proj = pcfm_batched(
                ut=u, vf=vf, t=t, u0=u0, dt=dt,
                hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
                guided_interpolation=guided_interpolation,
                interpolation_params=interpolation_params,
                eps=eps,
                use_vmap=use_vmap
            )
            u = u + dt * v_proj
        return u.detach()

    @torch.no_grad()
    def vanilla_sample(self, u0, n_step):
        """
        Vanilla FFM
        """
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="Vanilla"):
            vf = self.model(t, u)
            u = u + dt * vf
        return u.detach()

    @torch.no_grad()
    def eci_sample(self, u0, n_step, n_mix, resample_step, constraint):
        """
        ECI sampling
        """
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        cnt = 0
        dt = 1 / n_step
        grid = make_grid(u.shape[-2:], u.device)

        if resample_step == 0 or resample_step is None:
            resample_step = n_step * n_mix + 1

        for t in tqdm(ts[:-1], desc='ECI sampling'):
            for mix in range(n_mix):
                cnt += 1
                if cnt % resample_step == 0:
                    u0 = self.gp.sample(grid, u.shape[-2:], n_samples=u.shape[0])
                vf = self.model(t, u)
                u1 = u + vf * (1 - t)
                u1 = constraint.adjust(u1)
                if mix < n_mix - 1:
                    u = u1 * t + u0 * (1 - t)
                else:
                    u = u1 * (t + dt) + u0 * (1 - t - dt)
        return u.detach()


    def guided_sample(self, u0, u1_true, mask, n_step, loss_fn, eta=2e2):
        """
        DiffusionPDE: takes an IC and PINN loss (if known) on the extrapolated sample and updates the vector field 
        """
        device = u0.device
        u = u0.clone().to(device)
        u1_true = u1_true.to(device)
        mask = mask.to(device)
        ts = torch.linspace(0, 1, n_step + 1, device=device)

        for t in tqdm(ts[:-1], desc='DiffusionPDE sampling'):
            vf = self.model(t, u).detach()
            if t < ts[-2]:
                vf2 = self.model(t + 1 / n_step, u).detach()
                vf = (vf + vf2) / 2

            u.requires_grad_(True)
            u1_pred = u + vf * (1 - t)
            loss = loss_fn(u1_pred, u1_true, mask)
            loss.backward()
            grad = u.grad
            u = u.detach() + vf / n_step - eta * grad
        return u.detach()

    #
    def dflow_sample(self, u1_true, mask, n_sample, n_step, n_iter=20, lr=1e-1, loss_fn=None):
        """
        D-Flow: optimizes the noise by differentiating through the flow matching ODE steps 
        """
        device = u1_true.device
        mask = mask.to(device)
        grid = make_grid(u1_true.size()[1:], device)

        noise = self.gp.sample(grid, u1_true.size()[1:], n_samples=n_sample).to(device)
        noise.requires_grad_(True)

        ts = torch.linspace(0, 1, n_step + 1, device=device)

        def default_loss_fn(u_pred, u_true, mask):
            return ((u_pred - u_true) * mask).square().sum()
        loss_fn = loss_fn or default_loss_fn

        def euler_ffm(u):
            print("DFlow sampling...")
            tspan = torch.tensor([0, 1.], device=device)
            u = odeint(self.model, u, tspan, method="euler", options = {"step_size":ts[1]-ts[0]})[-1]
            return u 

        def closure():
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            u_pred = euler_ffm(noise)
            loss = loss_fn(u_pred, u1_true, mask)
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS([noise], max_iter=n_iter, lr=lr)
        optimizer.step(closure)

        with torch.no_grad():
            u_final = euler_ffm(noise)
        return u_final.detach()


class FFM_NS_sampler:
    """
    Collection of samplers using a pretrained functional flow matching model for Navier Stokes (2d)
    """
    def __init__(self, model):
        self.model = model

    # our method, PCFM
    def pcfm_sample(self, u0, n_step, hfunc, mode='root', newtonsteps=1, eps=1e-6,
                    guided_interpolation=True, interpolation_params={}):
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="PCFM sampling"):
            vf = self.model(t, u)
            v_proj = pcfm_2d_batched(
                ut=u, vf=vf, t=t, u0=u0, dt=dt,
                hfunc=hfunc, mode=mode, newtonsteps=newtonsteps,
                guided_interpolation=guided_interpolation,
                interpolation_params=interpolation_params,
                eps=eps
            )
            u = u + dt * v_proj
        return u.detach()

    @torch.no_grad()
    def vanilla_sample(self, u0, n_step):
        dt = 1.0 / n_step
        u = u0.clone()
        for t in tqdm(torch.linspace(0, 1, n_step + 1, device=u0.device)[:-1], desc="Vanilla"):
            vf = self.model(t, u)
            u = u + dt * vf
        return u.detach()

    @torch.no_grad()
    def eci_sample(self, u0, n_step, n_mix, resample_step, constraint):
        u = u0.clone()
        ts = torch.linspace(0, 1, n_step + 1, device=u0.device)
        cnt = 0
        dt = 1 / n_step
        grid = make_grid(u.shape[-2:], u.device)

        if resample_step == 0 or resample_step is None:
            resample_step = n_step * n_mix + 1

        for t in tqdm(ts[:-1], desc='ECI sampling'):
            for mix in range(n_mix):
                cnt += 1
                if cnt % resample_step == 0:
                    u0 = torch.randn_like(u)
                vf = self.model(t, u)
                u1 = u + vf * (1 - t)
                u1 = constraint.adjust(u1)
                if mix < n_mix - 1:
                    u = u1 * t + u0 * (1 - t)
                else:
                    u = u1 * (t + dt) + u0 * (1 - t - dt)
        return u.detach()

    def guided_sample(self, u0, u1_true, mask, n_step, loss_fn, eta=2e2):
        device = u0.device
        u = u0.clone().to(device)
        u1_true = u1_true.to(device)
        mask = mask.to(device)
        ts = torch.linspace(0, 1, n_step + 1, device=device)

        for t in tqdm(ts[:-1], desc='DiffusionPDE sampling'):
            vf = self.model(t, u).detach()
            if t < ts[-2]:
                vf2 = self.model(t + 1 / n_step, u).detach()
                vf = (vf + vf2) / 2

            u.requires_grad_(True)
            u1_pred = u + vf * (1 - t)
            loss = loss_fn(u1_pred, u1_true, mask)
            loss.backward()
            grad = u.grad
            u = u.detach() + vf / n_step - eta * grad
        return u.detach()

    def dflow_sample(self, u1_true, mask, n_sample, n_step, n_iter=20, lr=1e-1, loss_fn=None):
        device = u1_true.device
        mask = mask.to(device)
        grid = make_grid(u1_true.size()[1:], device)

        noise = torch.randn_like(u1_true) # randn for NS 
        noise.requires_grad_(True)
        ts = torch.linspace(0, 1, n_step + 1, device=device)
        
        def default_loss_fn(u_pred, u_true, mask):
            return ((u_pred - u_true) * mask).square().sum()
        loss_fn = loss_fn or default_loss_fn

        def euler_ffm(u):
            print("DFlow sampling...")
            tspan = torch.tensor([0, 1.], device=device)
            u = odeint(self.model, u, tspan, method="euler", options = {"step_size":ts[1]-ts[0]})[-1]
            return u 

        def closure():
            gc.collect()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            u_pred = euler_ffm(noise)
            loss = loss_fn(u_pred, u1_true, mask)
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS([noise], max_iter=n_iter, lr=lr)
        optimizer.step(closure)

        with torch.no_grad():
            u_final = euler_ffm(noise)
        return u_final.detach()

