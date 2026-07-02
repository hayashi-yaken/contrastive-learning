"""H3 collapse diagnostics: empirical Fisher (batch covariance in tangent coords),
split into radial vs tangential variance, condition number vs temperature."""
import math
import numpy as np
import torch
from ..geometry import lorentz as L


def logmap0(z, c=1.0, eps=1e-6):
    """Inverse exp map at origin: hyperboloid point -> tangent space vector.
    v = (arccosh(sqrt(c) x0) / sqrt(c)) * x_space / ||x_space||."""
    sqrt_c = math.sqrt(c)
    x0 = z[..., :1]
    x_space = z[..., 1:]
    norm = torch.norm(x_space, dim=-1, keepdim=True).clamp(min=eps)
    dist = torch.arccosh(torch.clamp(sqrt_c * x0, min=1.0 + eps)) / sqrt_c
    return dist * x_space / norm


def _covariance(x):
    x = x - x.mean(dim=0, keepdim=True)
    n = max(x.shape[0] - 1, 1)
    return (x.T @ x) / n


def _looks_hyperboloid(z, c):
    ip = L.lorentz_inner(z, z)
    return bool(torch.allclose(ip, torch.full_like(ip, -1.0 / c), atol=1e-2))


def fisher_eigenvalues(embeddings, c=1.0):
    if embeddings.shape[-1] >= 2 and _looks_hyperboloid(embeddings, c):
        tang = logmap0(embeddings, c=c)
    else:
        tang = embeddings
    cov = _covariance(tang).detach().cpu().numpy()
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    mean_vec = tang.mean(dim=0).detach().cpu().numpy()
    mean_norm = np.linalg.norm(mean_vec) + 1e-12
    u = mean_vec / mean_norm
    radial_var = float(u @ cov @ u)                       # variance along mean dir
    total = float(np.trace(cov))
    tangential_var = float((total - radial_var) / max(len(eigvals) - 1, 1))
    max_e = float(eigvals.max())
    min_e = float(max(eigvals.min(), 1e-12))
    return {
        "radial_var": radial_var,
        "tangential_var": tangential_var,
        "condition_number": max_e / min_e,
        "eigenvalues": [float(e) for e in sorted(eigvals, reverse=True)],
    }


def collapse_vs_temperature(results_by_tau):
    taus = sorted(results_by_tau)
    return {
        "tau": taus,
        "condition_number": [results_by_tau[t]["condition_number"] for t in taus],
        "radial_var": [results_by_tau[t]["radial_var"] for t in taus],
        "tangential_var": [results_by_tau[t]["tangential_var"] for t in taus],
    }
