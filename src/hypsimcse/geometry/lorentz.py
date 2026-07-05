"""Lorentz (hyperboloid) model of hyperbolic space. Appendix A formulas.

Points are stored as full (d+1) vectors [x_0, x_1..x_d]; x_0 is the time axis.
Curvature c > 0; manifold is <x,x>_L = -1/c, x_0 > 0.
"""
import math
import torch

EPS = 1e-6


def _sqrt(c):
    """sqrt that keeps the autograd graph when c is a tensor (for learnable
    curvature), and stays a plain float otherwise."""
    if isinstance(c, torch.Tensor):
        return torch.sqrt(c)
    return math.sqrt(c)


def lorentz_inner(x, y, keepdim=False):
    """<x,y>_L = -x0*y0 + sum_{i>=1} xi*yi  over the last dimension."""
    prod = x * y
    inner = prod[..., 1:].sum(dim=-1) - prod[..., 0]
    return inner.unsqueeze(-1) if keepdim else inner


def time_component(x_space, c=1.0):
    """x_0 = sqrt(1/c + ||x_space||^2)."""
    sq = (x_space * x_space).sum(dim=-1, keepdim=True)
    return torch.sqrt(torch.clamp(1.0 / c + sq, min=EPS))


def to_hyperboloid(x_space, c=1.0):
    """Prepend the time component to space coords, yielding a manifold point."""
    x0 = time_component(x_space, c=c)
    return torch.cat([x0, x_space], dim=-1)


def origin(d, c=1.0, dtype=torch.float32, device=None):
    """Origin o = (1/sqrt(c), 0, ..., 0) in R^{d+1}."""
    sqrt_c = _sqrt(c)
    if isinstance(sqrt_c, torch.Tensor):
        head = (1.0 / sqrt_c).reshape(1).to(dtype)
        tail = torch.zeros(d, dtype=dtype, device=sqrt_c.device)
        return torch.cat([head, tail])
    o = torch.zeros(d + 1, dtype=dtype, device=device)
    o[0] = 1.0 / sqrt_c
    return o


def expmap0(v_space, c=1.0, max_norm=None, eps=EPS):
    """Exp map at the origin of a tangent (space-only) vector v_space.

    x_space = sinh(sqrt(c)|v|)/(sqrt(c)|v|) * v ;  x0 = (1/sqrt(c)) cosh(sqrt(c)|v|).
    """
    sqrt_c = _sqrt(c)
    norm = torch.norm(v_space, dim=-1, keepdim=True)
    if max_norm is not None:
        scale = torch.clamp(max_norm / torch.clamp(norm, min=eps), max=1.0)
        v_space = v_space * scale
        norm = torch.norm(v_space, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    space = torch.sinh(sqrt_c * norm) / (sqrt_c * norm) * v_space
    x0 = torch.cosh(sqrt_c * norm) / sqrt_c
    return torch.cat([x0, space], dim=-1)


def dist(x, y, c=1.0, eps=EPS):
    """d_H(x,y) = (1/sqrt(c)) arccosh(-c <x,y>_L)."""
    sqrt_c = _sqrt(c)
    arg = torch.clamp(-c * lorentz_inner(x, y), min=1.0 + eps)
    return torch.arccosh(arg) / sqrt_c
