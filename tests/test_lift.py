import torch
from hypsimcse.models.lift import HyperbolicHead
from hypsimcse.geometry import lorentz as L


def test_hyp_head_output_on_manifold():
    head = HyperbolicHead(in_dim=8, out_dim=5, geometry="HYP", c=1.0)
    z = head(torch.randn(4, 8))
    assert z.shape == (4, 6)
    assert torch.allclose(L.lorentz_inner(z, z), torch.full((4,), -1.0), atol=1e-3)


def test_euc_head_returns_tangent():
    head = HyperbolicHead(in_dim=8, out_dim=5, geometry="EUC")
    z = head(torch.randn(4, 8))
    assert z.shape == (4, 5)


def test_learnable_c_positive():
    head = HyperbolicHead(in_dim=8, out_dim=5, geometry="HYP", learnable_c=True)
    assert head.curvature() > 0
    assert any(p.requires_grad for p in head.parameters())


def test_learnable_curvature_receives_gradient():
    # Regression test: the learnable curvature must actually get a gradient
    # through the exp map (previously detached via .item(), so it never learned).
    head = HyperbolicHead(in_dim=8, out_dim=5, geometry="HYP", learnable_c=True)
    z = head(torch.randn(4, 8))
    z.sum().backward()
    assert head.raw_c.grad is not None
    assert head.raw_c.grad.abs().item() > 0
