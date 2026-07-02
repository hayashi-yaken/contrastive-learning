import math, torch
from hypsimcse.geometry import lorentz as L


def test_inner_product_signature():
    x = torch.tensor([1.0, 0.0, 0.0]); y = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(L.lorentz_inner(x, y), torch.tensor(-1.0))


def test_origin_on_manifold():
    o = L.origin(2, c=1.0)
    assert torch.allclose(L.lorentz_inner(o, o), torch.tensor(-1.0), atol=1e-5)


def test_expmap0_lands_on_manifold():
    v = torch.randn(8, 5)
    z = L.expmap0(v, c=1.0)
    assert z.shape == (8, 6)
    ip = L.lorentz_inner(z, z)
    assert torch.allclose(ip, torch.full((8,), -1.0), atol=1e-4)


def test_expmap0_zero_is_origin():
    z = L.expmap0(torch.zeros(3, 5), c=1.0)
    assert torch.allclose(z[:, 0], torch.ones(3), atol=1e-5)
    assert torch.allclose(z[:, 1:], torch.zeros(3, 5), atol=1e-5)


def test_dist_self_is_zero():
    z = L.expmap0(torch.randn(4, 5))
    d = L.dist(z, z)
    # float32 imprecision in the exp-map lift dominates; 1e-2 is the honest
    # tolerance for arccosh near its (numerically unstable) argument = 1.
    assert torch.allclose(d, torch.zeros(4), atol=1e-2)


def test_dist_matches_arccosh_inner_c1():
    z1 = L.expmap0(torch.randn(4, 5)); z2 = L.expmap0(torch.randn(4, 5))
    d = L.dist(z1, z2, c=1.0)
    inner = L.lorentz_inner(z1, z2)
    assert torch.allclose(d, torch.arccosh(torch.clamp(-inner, min=1.0 + L.EPS)), atol=1e-4)


def test_dist_triangle_inequality():
    a = L.expmap0(torch.randn(1, 5)); b = L.expmap0(torch.randn(1, 5)); c = L.expmap0(torch.randn(1, 5))
    dab = L.dist(a, b); dbc = L.dist(b, c); dac = L.dist(a, c)
    assert (dac <= dab + dbc + 1e-3).all()


def test_max_norm_clamps():
    v = torch.randn(4, 5) * 1e3
    z = L.expmap0(v, max_norm=5.0)
    assert torch.isfinite(z).all()
