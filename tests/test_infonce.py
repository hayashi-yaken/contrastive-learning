import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.losses import infonce as I


def test_info_nce_perfect_alignment_low_loss():
    z = L.expmap0(torch.randn(8, 4))
    loss = I.info_nce(z, z.clone(), "HYP-inner", tau=0.05)
    assert loss.item() < 0.5


def test_info_nce_gradients_flow():
    v = torch.randn(6, 4, requires_grad=True)
    z = L.expmap0(v)
    loss = I.info_nce(z, L.expmap0(v.detach() + 0.01), "HYP-dist", tau=0.1)
    loss.backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_info_nce_euc_runs():
    u = torch.randn(8, 16); up = u + 0.01 * torch.randn(8, 16)
    loss = I.info_nce(u, up, "EUC", tau=0.05)
    assert torch.isfinite(loss)


def test_root_anchor_reg_rewards_spread():
    near = L.expmap0(torch.randn(8, 4) * 0.1)
    far = L.expmap0(torch.randn(8, 4) * 2.0)
    assert I.root_anchor_reg(far) < I.root_anchor_reg(near)


def test_total_loss_parts():
    z = L.expmap0(torch.randn(8, 4))
    loss, parts = I.total_loss(z, z.clone(), "HYP-inner", tau=0.05, anchor_weight=0.1)
    assert {"infonce", "anchor"} <= set(parts)
    assert torch.isfinite(loss)
