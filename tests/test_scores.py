import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.geometry import scores as S


def test_labels_registered():
    assert S.SCORE_LABELS == ("EUC", "HYP-dist", "HYP-dist2", "HYP-inner")


def test_euc_is_cosine():
    u = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    v = torch.tensor([[1.0, 0.0]])
    out = S.pairwise_scores(u, v, "EUC")
    assert out.shape == (2, 1)
    assert torch.allclose(out[:, 0], torch.tensor([1.0, 0.0]), atol=1e-5)


def test_hyp_dist_diagonal_zero():
    z = L.expmap0(torch.randn(3, 4))
    out = S.pairwise_scores(z, z, "HYP-dist")
    assert out.shape == (3, 3)
    assert torch.allclose(torch.diagonal(out), torch.zeros(3), atol=1e-2)


def test_hyp_dist2_is_negative_square():
    z1 = L.expmap0(torch.randn(3, 4)); z2 = L.expmap0(torch.randn(2, 4))
    d = S.pairwise_scores(z1, z2, "HYP-dist")
    d2 = S.pairwise_scores(z1, z2, "HYP-dist2")
    assert torch.allclose(d2, -(d ** 2), atol=1e-4)


def test_hyp_inner_equals_neg_cosh_dist_c1():
    z1 = L.expmap0(torch.randn(4, 4)); z2 = L.expmap0(torch.randn(5, 4))
    inner = S.pairwise_scores(z1, z2, "HYP-inner", c=1.0)
    d = S.pairwise_scores(z1, z2, "HYP-dist", c=1.0)  # = -d_H
    assert torch.allclose(inner, -torch.cosh(-d), atol=1e-3)


def test_hyp_inner_diagonal_is_minus_one_c1():
    z = L.expmap0(torch.randn(4, 4))
    inner = S.pairwise_scores(z, z, "HYP-inner", c=1.0)
    assert torch.allclose(torch.diagonal(inner), torch.full((4,), -1.0), atol=1e-3)
