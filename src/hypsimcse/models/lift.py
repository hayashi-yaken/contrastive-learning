"""Shared head: project a Euclidean feature to either an L2-normalizable
tangent vector (EUC) or a hyperboloid point via exp map (HYP)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from ..geometry import lorentz as L


class HyperbolicHead(nn.Module):
    def __init__(self, in_dim, out_dim, geometry, c=1.0,
                 learnable_c=False, max_tangent_norm=None):
        super().__init__()
        assert geometry in ("EUC", "HYP")
        self.geometry = geometry
        self.out_dim = out_dim
        self.max_tangent_norm = max_tangent_norm
        self.proj = nn.Linear(in_dim, out_dim)
        self.learnable_c = learnable_c
        if learnable_c:
            # softplus(raw_c) ~ c ; init so curvature ~= c
            self.raw_c = nn.Parameter(torch.tensor(math.log(math.exp(c) - 1.0)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

    def curvature(self):
        """Python float — for reporting / evaluation."""
        if self.learnable_c:
            return Fn.softplus(self.raw_c).item()
        return float(self._c)

    def curvature_tensor(self):
        """Differentiable curvature for the forward/training path (keeps the
        autograd graph so a learnable curvature actually receives gradients)."""
        if self.learnable_c:
            return Fn.softplus(self.raw_c)
        return self._c

    def forward(self, h):
        v = self.proj(h)
        if self.geometry == "EUC":
            return v
        return L.expmap0(v, c=self.curvature_tensor(), max_norm=self.max_tangent_norm)
