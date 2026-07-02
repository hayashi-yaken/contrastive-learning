from hypsimcse.data import wordnet as W
from hypsimcse.data import splits as SP
import pytest


@pytest.fixture(scope="module")
def data():
    W.ensure_wordnet()
    return W.load_noun_hypernymy(max_synsets=2000)


def test_reconstruction_uses_all_edges(data):
    r = SP.reconstruction_split(data)
    assert set(r["edges"]) == set(data.closure_edges)


def test_link_prediction_partitions(data):
    sp = SP.link_prediction_split(data, seed=0)
    total = set(map(tuple, sp["train"])) | set(map(tuple, sp["valid"])) | set(map(tuple, sp["test"]))
    assert total == set(data.closure_edges)
    assert not (set(map(tuple, sp["train"])) & set(map(tuple, sp["test"])))
    assert len(sp["valid"]) > 0 and len(sp["test"]) > 0


def test_link_prediction_deterministic(data):
    a = SP.link_prediction_split(data, seed=1)
    b = SP.link_prediction_split(data, seed=1)
    assert a["test"] == b["test"]


def test_inductive_eval_touches_heldout(data):
    sp = SP.inductive_split(data, test_synset_frac=0.1, seed=0)
    ts = sp["test_synsets"]
    for h, hyper in sp["eval_edges"]:
        assert h in ts or hyper in ts
    for h, hyper in sp["train_edges"]:
        assert h not in ts and hyper not in ts


def test_negatives_exclude_true_hypernyms(data):
    import random
    rng = random.Random(0)
    edge = data.closure_edges[0]
    true_hypers = {hyper for h, hyper in data.closure_edges if h == edge[0]}
    negs = SP.negatives_for(edge, data, num=10, rng=rng)
    assert len(negs) == 10
    assert not (set(negs) & true_hypers)
