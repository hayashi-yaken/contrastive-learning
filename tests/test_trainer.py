import torch
from hypsimcse.training.config import ExperimentConfig
from hypsimcse.training.seeding import set_seed
from hypsimcse.training import trainer as T
from hypsimcse.data import wordnet as W


def _tiny_cfg(**kw):
    base = dict(track="graph", geometry="HYP", score="HYP-inner", dim=5, tau=0.1,
                c=1.0, learnable_c=False, max_tangent_norm=5.0, anchor_weight=0.0,
                lr=0.05, weight_decay=0.0, grad_clip=1.0, epochs=1, batch_size=64,
                num_negatives=10, model_name="prajjwal1/bert-tiny", pooling="cls",
                max_synsets=800, optimizer="adamw", seed=0, max_length=32)
    base.update(kw)
    return ExperimentConfig(**base)


def test_set_seed_reproducible():
    set_seed(0); a = torch.randn(3)
    set_seed(0); b = torch.randn(3)
    assert torch.allclose(a, b)


def test_config_yaml_roundtrip(tmp_path):
    import yaml
    cfg = _tiny_cfg()
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg.to_dict()))
    loaded = ExperimentConfig.from_yaml(str(p))
    assert loaded.to_dict() == cfg.to_dict()


def test_trainer_graph_runs_and_loss_finite():
    W.ensure_wordnet()
    data = W.load_noun_hypernymy(max_synsets=800)
    cfg = _tiny_cfg(epochs=1)
    set_seed(cfg.seed)
    model = T.build_model(cfg, data, "cpu")
    tr = T.Trainer(cfg, data, model, "cpu")
    logs = tr.train()
    assert len(logs) == 1
    assert all(l["loss"] == l["loss"] for l in logs)


def test_trainer_loss_decreases_graph():
    W.ensure_wordnet()
    data = W.load_noun_hypernymy(max_synsets=800)
    cfg = _tiny_cfg(epochs=3)
    set_seed(cfg.seed)
    model = T.build_model(cfg, data, "cpu")
    logs = T.Trainer(cfg, data, model, "cpu").train()
    assert logs[-1]["loss"] < logs[0]["loss"]
