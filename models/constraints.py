# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
# the following PDE constraints are used in ECI sampling only (Cheng et al.)

"""
class hierarchy for PDE constraints
- Constraint (abstract)
    - ChainCondition
    - NoneCondition
    - DirichletCondition
    - DirichletXtCondition
        - DirichletX0Condition
            - InitialCondition
            - BoundaryCondition
        - DirichletXnCondition
    - PeriodicCondition
    - ConservationLaw
"""

from abc import ABC, abstractmethod


def expand_dims(t, ndim):
    """
    Expand dimensions of a tensor.
    :param t: input tensor, size (B, ...)
    :param ndim: number of dimensions to expand
    :return: expanded tensor, size (B, ...)
    """
    return t.view(-1, *([1] * (ndim - 1)))


#################################################
# Base Class for Constraints                    #
#################################################
class Constraint(ABC):
    """
    Base class for PDE constraints.
    :param value: value for the constraint, if any
    :param mask: mask for the constrained region, if any
    """

    def __init__(self, value=None, mask=None):
        self.value = value
        self.mask = mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def to(self, device):
        if self.value is not None:
            self.value = self.value.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self

    @abstractmethod
    def adjust(self, x1):
        """
        Adjust the solution x1 such that it satisfies the constraints.
        :param x1: input to adjust, size (B, ...)
        :return: adjusted input, size (B, ...)
        """
        pass


#################################################
# General Constraints                           #
#################################################

class ChainConstraint(Constraint):
    """
    Chain multiple constraints.
    """

    def __init__(self, *constraints):
        super().__init__()
        self.constraints = constraints

    def to(self, device):
        for constraint in self.constraints:
            constraint.to(device)
        return self

    def adjust(self, x1):
        for cond in self.constraints:
            x1 = cond.adjust(x1)
        return x1


class NoneConstraint(Constraint):
    """
    Dummy constraint.
    """

    def adjust(self, x1):
        return x1


#################################################
# Dirichlet Boundary Conditions                 #
#################################################

class DirichletCondition(Constraint):
    """
    Dirichlet boundary condition. Specify u(x) = value at the boundary.
    """

    def adjust(self, x1):
        mask = self.mask.expand_as(x1)
        value = self.value.expand_as(x1)
        x1[mask] = value[mask]
        return x1


class DirichletXtCondition(Constraint):
    """
    Dirichlet boundary condition at timestep t (some frame in the middle). Specify u(x, t) = value.
    :param value: value at timestep t
    :param t: time frame to apply the boundary condition
    :param tdim: time dimension, default is the last dimension
    """

    def __init__(self, value, t: int, tdim: int = -1):
        super().__init__(value)
        self.t = t
        self.ndim = value.dim()  # value dimension, input should have an additional dimension
        self.tdim = tdim % (self.ndim + 1)
        self.slice = (slice(None),) * self.tdim + (t,)

    def adjust(self, x1):
        x1[self.slice] = self.value.expand_as(x1[self.slice])
        return x1


class DirichletX0Condition(DirichletXtCondition):
    """
    Dirichlet boundary condition at t=0 (first frame). Specify u(x, 0) = value.
    :param value: value at t=0
    :param tdim: time dimension, default is the last dimension
    """

    def __init__(self, value, tdim=-1):
        super().__init__(value, 0, tdim)


class DirichletXnCondition(DirichletXtCondition):
    """
    Dirichlet boundary condition at t=n (last frame). Specify u(x, n) = value.
    :param value: value at t=n
    :param tdim: time dimension, default is the last dimension
    """

    def __init__(self, value, tdim=-1):
        super().__init__(value, -1, tdim)


class InitialCondition(DirichletX0Condition):
    pass


class BoundaryCondition(DirichletX0Condition):
    def __init__(self, value, tdim=-2):
        super().__init__(value, tdim)


#################################################
# Periodic Boundary Conditions (PBC)            #
#################################################


class PeriodicCondition(Constraint):
    """
    Periodic boundary condition. Specify u(x, 0) = u(x, -1) for all x.
    :param ndim: number of dimensions of the input
    :param dims: dimensions to apply the boundary condition, default is the first (non-batch) dimension
    """

    def __init__(self, ndim, dims=(1,)):
        super().__init__()
        self.ndim = ndim
        self.dims = [i % ndim for i in dims]

    def adjust(self, x1):
        for dim in self.dims:
            slice_start = (slice(None),) * dim + (0,)
            slice_end = (slice(None),) * dim + (-1,)
            value = (x1[slice_end] + x1[slice_start]) / 2
            x1[slice_start] = value
            x1[slice_end] = value
        return x1


#################################################
# Conservation Laws                             #
#################################################

class RegionConservationLaw(Constraint):
    r"""
    Conservation law over the region. Specify the conservation values along certain dimensions.

    .. math:: \int_\Omega u(x,t) dx = b(t)

    :param value: the conservation values b(t)
    :param dims: dimensions to apply the conservation law, default is the first (non-batch) dimension
    :param area: constrained region area :math:`\int_\Omega dx`, default is 1
    """

    def __init__(self, value, dims=(1,), area=1.):
        super().__init__(value)
        self.ndim = value.dim() + len(dims)
        self.dims = [i % self.ndim for i in dims]
        self.area = area

    def adjust(self, x1):
        diff = self.value - x1.mean(dim=self.dims) * self.area
        x1 = x1 + diff.view([1 if i in self.dims else sz for i, sz in enumerate(x1.size())]) / self.area
        return x1
