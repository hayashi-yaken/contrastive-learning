"""Pure-graph embedding (Nickel-Kiela style): one learnable tangent vector per
synset, lifted to the manifold. No encoder — isolates the geometry effect."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from ..geometry import lorentz as L


class GraphEmbedding(nn.Module):
    def __init__(self, num_nodes, dim, geometry, c=1.0, learnable_c=False,
                 max_tangent_norm=None, init_scale=1e-3):
        super().__init__()
        assert geometry in ("EUC", "HYP")
        self.geometry = geometry
        self.dim = dim
        self.max_tangent_norm = max_tangent_norm
        self.table = nn.Embedding(num_nodes, dim)
        nn.init.normal_(self.table.weight, std=init_scale)
        self.learnable_c = learnable_c
        if learnable_c:
            self.raw_c = nn.Parameter(torch.tensor(math.log(math.exp(c) - 1.0)))
        else:
            self.register_buffer("_c", torch.tensor(float(c)))

    def curvature(self):
        if self.learnable_c:
            return Fn.softplus(self.raw_c).item()
        return float(self._c)

    def forward(self, node_ids):
        v = self.table(node_ids)
        if self.geometry == "EUC":
            return v
        return L.expmap0(v, c=self.curvature(), max_norm=self.max_tangent_norm)
