import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import reconstruction as R


class _Data:  # minimal stand-in
    def __init__(self, n):
        self.synsets = list(range(n))
        self.closure_edges = [(0, 1), (2, 3)]


def test_rank_of_positive_best_is_one():
    q = L.expmap0(torch.randn(1, 4))
    negs = L.expmap0(torch.randn(5, 4))
    r = R.rank_of_positive(q, q.clone(), negs, "HYP-inner")
    assert r == 1


def test_reconstruction_metrics_keys():
    emb = L.expmap0(torch.randn(4, 4))
    data = _Data(4)
    m = R.reconstruction_metrics(emb, data.closure_edges, data,
                                 num_negatives=2, score="HYP-inner", seed=0)
    assert set(m) >= {"MAP", "mean_rank"}
    assert 0.0 <= m["MAP"] <= 1.0
    assert m["mean_rank"] >= 1.0


def test_map_perfect_when_positives_closest():
    emb = torch.stack([
        L.expmap0(torch.tensor([0.0, 0.0, 0.0, 0.0])),   # 0
        L.expmap0(torch.tensor([0.01, 0.0, 0.0, 0.0])),  # 1 (near 0)
        L.expmap0(torch.tensor([5.0, 0.0, 0.0, 0.0])),   # 2
        L.expmap0(torch.tensor([5.01, 0.0, 0.0, 0.0])),  # 3 (near 2)
    ])
    data = _Data(4)
    m = R.reconstruction_metrics(emb, data.closure_edges, data,
                                 num_negatives=2, score="HYP-inner", seed=0)
    assert m["MAP"] > 0.6
