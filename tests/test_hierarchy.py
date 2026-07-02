import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import hierarchy as H


class _Data:
    def __init__(self, n):
        self.synsets = list(range(n))
        self.direct_edges = [(1, 0), (2, 1), (3, 2)]  # chain 0<-1<-2<-3
        self.roots = [0]


def test_embedding_norms_monotone():
    v = torch.zeros(3, 4); v[0] = 0.0; v[1, 0] = 1.0; v[2, 0] = 2.0
    z = L.expmap0(v)
    norms = H.embedding_norms(z)
    assert norms[0] < norms[1] < norms[2]


def test_norm_depth_correlation_positive_when_aligned():
    v = torch.zeros(4, 4)
    for i in range(4):
        v[i, 0] = float(i)
    z = L.expmap0(v)
    data = _Data(4)
    out = H.norm_depth_correlation(z, data)
    assert out["spearman"] > 0.9


def test_embedding_delta_nonneg():
    z = L.expmap0(torch.randn(20, 4))
    d = H.embedding_delta_hyperbolicity(z, num_samples=50, seed=0)
    assert d >= 0.0
