import torch, pytest
from hypsimcse.geometry import lorentz as L

pytest.importorskip("transformers")
MODEL = "prajjwal1/bert-tiny"


def test_encoder_hyp_on_manifold():
    from hypsimcse.models.encoder_model import SentenceEncoder
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="HYP", c=1.0)
    batch = tok(["a dog", "a cat"], return_tensors="pt", padding=True)
    z = model(batch["input_ids"], batch["attention_mask"])
    assert z.shape == (2, 6)
    assert torch.allclose(L.lorentz_inner(z, z), torch.full((2,), -1.0), atol=1e-3)


def test_encoder_euc_shape():
    from hypsimcse.models.encoder_model import SentenceEncoder
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="EUC")
    batch = tok(["hi"], return_tensors="pt", padding=True)
    z = model(batch["input_ids"], batch["attention_mask"])
    assert z.shape == (1, 5)


def test_encode_texts_helper():
    from hypsimcse.models.encoder_model import SentenceEncoder, encode_texts
    model, tok = SentenceEncoder.build(MODEL, out_dim=5, geometry="HYP")
    z = encode_texts(model, tok, ["a", "b", "c"], device="cpu", batch_size=2)
    assert z.shape == (3, 6)
