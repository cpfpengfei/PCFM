# pcfm/__init__.py
from .constraints import Residuals, Residuals2D
from .ffm_sampler import FFM_sampler, FFM_NS_sampler
from .pcfm_sampling import (
    compute_jacobian,
    fast_project_batched,
    fast_project_batched_chunk,
    make_grid,
    pcfm_batched,
    pcfm_2d_batched,
)

__all__ = [
    "Residuals", "Residuals2D",
    "FFM_sampler", "FFM_NS_sampler",
    "compute_jacobian", "fast_project_batched", "fast_project_batched_chunk",
    "make_grid", "pcfm_batched", "pcfm_2d_batched",
]

__version__ = "0.1.0"
