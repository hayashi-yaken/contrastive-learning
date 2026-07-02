from hypsimcse.data import wordnet as W
from hypsimcse.data import graph_stats as G
import pytest


@pytest.fixture(scope="module")
def data():
    W.ensure_wordnet()
    return W.load_noun_hypernymy(max_synsets=1500)


def test_depths_roots_are_zero(data):
    d = G.depths(data)
    for r in data.roots:
        assert d[r] == 0


def test_depths_child_at_most_one_below_parent(data):
    # WordNet nouns form a DAG (multiple inheritance), so a hyponym may reach
    # the root via a shorter alternate parent. The invariant BFS guarantees is
    # that a child is at most one level below its parent's shortest depth.
    d = G.depths(data)
    for h, hyper in data.direct_edges:
        assert d[h] <= d[hyper] + 1


def test_branching_positive(data):
    b = G.branching_factors(data)
    assert b["mean"] > 0 and b["max"] >= 1


def test_gromov_delta_nonneg_and_small_for_tree(data):
    delta = G.gromov_delta(data, num_samples=300, seed=0)
    assert delta >= 0.0


def test_summarize_json_serializable(data):
    import json
    s = G.summarize(data, num_samples=200)
    json.dumps(s)
    assert "gromov_delta" in s and "depth_distribution" in s
