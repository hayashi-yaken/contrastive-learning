import torch, pytest
from hypsimcse.eval import sts as STS

pytest.importorskip("transformers")
MODEL = "prajjwal1/bert-tiny"


def test_sts_spearman_returns_float():
    from hypsimcse.models.encoder_model import SentenceEncoder
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="HYP", c=1.0)
    pairs = [("a cat", "a dog"), ("hello world", "hello world"), ("x", "completely different y")]
    gold = [3.0, 5.0, 1.0]
    r = STS.sts_spearman(model, tok, pairs, gold, "HYP-inner", device="cpu")
    assert isinstance(r, float) and -1.0 <= r <= 1.0
