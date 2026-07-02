import math, torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import fisher as Fi


def test_logmap_inverts_expmap():
    v = torch.randn(6, 4)
    z = L.expmap0(v)
    v2 = Fi.logmap0(z)
    assert torch.allclose(v, v2, atol=1e-3)


def test_fisher_keys():
    z = L.expmap0(torch.randn(200, 5))
    out = Fi.fisher_eigenvalues(z)
    assert set(out) >= {"radial_var", "tangential_var", "condition_number", "eigenvalues"}
    assert out["condition_number"] >= 1.0


def test_collapse_detects_low_rank():
    v = torch.zeros(200, 5); v[:, 0] = torch.randn(200) * 2.0; v[:, 1:] = torch.randn(200, 4) * 1e-3
    z = L.expmap0(v)
    out = Fi.fisher_eigenvalues(z)
    assert out["condition_number"] > 100.0


def test_collapse_vs_temperature_sorted():
    a = Fi.fisher_eigenvalues(L.expmap0(torch.randn(100, 4)))
    b = Fi.fisher_eigenvalues(L.expmap0(torch.randn(100, 4)))
    out = Fi.collapse_vs_temperature({0.5: a, 0.05: b})
    assert out["tau"] == [0.05, 0.5]
