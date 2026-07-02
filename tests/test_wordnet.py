import pytest
from hypsimcse.data import wordnet as W


@pytest.fixture(scope="module")
def data():
    W.ensure_wordnet()
    return W.load_noun_hypernymy(max_synsets=2000)


def test_synsets_and_index_consistent(data):
    assert len(data.synsets) == len(data.index)
    for i, name in enumerate(data.synsets):
        assert data.index[name] == i


def test_edges_reference_valid_indices(data):
    n = len(data.synsets)
    for h, hyper in data.direct_edges:
        assert 0 <= h < n and 0 <= hyper < n


def test_closure_superset_of_direct(data):
    direct = set(data.direct_edges)
    closure = set(data.closure_edges)
    assert direct <= closure
    assert len(closure) >= len(direct)


def test_glosses_nonempty_and_clean(data):
    assert len(data.glosses) == len(data.synsets)
    g = data.glosses[0]
    assert g == g.lower() and "  " not in g


def test_gloss_text_format():
    from nltk.corpus import wordnet as wn
    s = wn.synset("dog.n.01")
    txt = W.gloss_text(s)
    assert " : " in txt and txt == txt.lower()


def test_roots_have_no_hypernym(data):
    hypo_with_parent = {h for h, _ in data.direct_edges}
    for r in data.roots:
        assert r not in hypo_with_parent
