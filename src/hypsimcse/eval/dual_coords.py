"""Theory-implementation bridge: the softmax-weighted Lorentz mean the model
uses IS the dual coordinate eta = grad psi (Einstein midpoint)."""
import math
import torch
from ..geometry import lorentz as L


def lorentz_centroid(points, weights, c=1.0, eps=1e-6):
    """Weighted Lorentz mean projected back to the hyperboloid."""
    w = weights.reshape(-1, 1)
    m = (w * points).sum(dim=0)
    denom = torch.sqrt(torch.clamp(-c * L.lorentz_inner(m, m), min=eps))
    return m / (denom * math.sqrt(c))


def softmax_lorentz_mean(query, points, tau, c=1.0):
    inner = L.lorentz_inner(query.expand_as(points), points)  # (N,)
    weights = torch.softmax(inner / tau, dim=0)
    return lorentz_centroid(points, weights, c=c), weights


def dual_coordinate_gap(query, points, tau, c=1.0):
    mean, weights = softmax_lorentz_mean(query[0], points, tau, c=c)
    explicit = lorentz_centroid(points, weights, c=c)
    return L.dist(mean.unsqueeze(0), explicit.unsqueeze(0), c=c).item()
