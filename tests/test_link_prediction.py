import torch
from hypsimcse.geometry import lorentz as L
from hypsimcse.eval import link_prediction as LP


class _Data:
    def __init__(self, n):
        self.synsets = list(range(n))
        self.closure_edges = [(0, 1), (2, 3)]


def test_lp_keys_and_ranges():
    emb = L.expmap0(torch.randn(4, 4))
    data = _Data(4)
    m = LP.link_prediction_metrics(emb, data.closure_edges, data,
                                   num_negatives=2, score="HYP-inner", seed=0)
    assert set(m) == {"MAP", "AUC"}
    assert 0.0 <= m["AUC"] <= 1.0 and 0.0 <= m["MAP"] <= 1.0


def test_auc_high_when_positive_closest():
    emb = torch.stack([
        L.expmap0(torch.tensor([0.0, 0.0, 0.0])),
        L.expmap0(torch.tensor([0.01, 0.0, 0.0])),
        L.expmap0(torch.tensor([5.0, 0.0, 0.0])),
        L.expmap0(torch.tensor([5.01, 0.0, 0.0])),
    ])
    data = _Data(4)
    m = LP.link_prediction_metrics(emb, data.closure_edges, data,
                                   num_negatives=2, score="HYP-inner", seed=0)
    assert m["AUC"] > 0.6
