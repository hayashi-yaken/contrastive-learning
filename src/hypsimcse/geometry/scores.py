"""Pairwise similarity scores that plug into InfoNCE. Only the score differs
across experimental conditions (EUC / HYP-dist / HYP-dist2 / HYP-inner)."""
import math
import torch
import torch.nn.functional as Fn
from . import lorentz as L

SCORE_LABELS = ("EUC", "HYP-dist", "HYP-dist2", "HYP-inner")


def is_hyperbolic(label):
    return label != "EUC"


def _pairwise_lorentz_inner(u, v):
    """(N,d+1),(M,d+1) -> (N,M) matrix of <u_i, v_j>_L."""
    time = -u[:, :1] @ v[:, :1].T
    space = u[:, 1:] @ v[:, 1:].T
    return time + space


def _pairwise_dist(u, v, c=1.0, eps=L.EPS):
    inner = _pairwise_lorentz_inner(u, v)
    arg = torch.clamp(-c * inner, min=1.0 + eps)
    return torch.arccosh(arg) / math.sqrt(c)


def pairwise_scores(u, v, label, c=1.0, eps=L.EPS):
    if label == "EUC":
        un = Fn.normalize(u, dim=-1)
        vn = Fn.normalize(v, dim=-1)
        return un @ vn.T
    if label == "HYP-dist":
        return -_pairwise_dist(u, v, c=c, eps=eps)
    if label == "HYP-dist2":
        return -_pairwise_dist(u, v, c=c, eps=eps) ** 2
    if label == "HYP-inner":
        return c * _pairwise_lorentz_inner(u, v)
    raise ValueError(f"unknown score label: {label!r}")
