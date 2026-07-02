import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import dual_coords as D


def test_centroid_on_manifold():
    pts = L.expmap0(torch.randn(5, 4))
    w = torch.softmax(torch.randn(5), dim=0)
    m = D.lorentz_centroid(pts, w)
    assert torch.allclose(L.lorentz_inner(m, m), torch.tensor(-1.0), atol=1e-3)


def test_centroid_of_identical_points_is_the_point():
    p = L.expmap0(torch.randn(1, 4)).repeat(5, 1)
    w = torch.softmax(torch.randn(5), dim=0)
    m = D.lorentz_centroid(p, w)
    assert L.dist(m.unsqueeze(0), p[0:1]).item() < 1e-2


def test_dual_gap_small():
    pts = L.expmap0(torch.randn(6, 4))
    q = L.expmap0(torch.randn(1, 4))
    gap = D.dual_coordinate_gap(q, pts, tau=0.1)
    assert gap < 1e-2
