import torch, pytest
from hypsimcse.eval import inductive as IND

pytest.importorskip("transformers")
MODEL = "prajjwal1/bert-tiny"


class _Data:
    def __init__(self):
        self.synsets = ["a.n.01", "b.n.01", "c.n.01", "d.n.01"]
        self.glosses = ["alpha thing", "beta thing", "gamma item", "delta item"]
        self.closure_edges = [(0, 1), (2, 3)]


def test_inductive_runs_on_heldout():
    from hypsimcse.models.encoder_model import SentenceEncoder
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="HYP", c=1.0)
    data = _Data()
    isplit = {"test_synsets": {3}, "eval_edges": [(2, 3)],
              "train_synsets": {0, 1, 2}, "train_edges": [(0, 1)]}
    m = IND.inductive_metrics(model, tok, data, isplit, num_negatives=1,
                              score="HYP-inner", device="cpu", seed=0)
    assert set(m) == {"MAP", "mean_rank"}
    assert 0.0 <= m["MAP"] <= 1.0
