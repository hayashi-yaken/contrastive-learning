"""Task splits: reconstruction (all edges), link prediction (edge hold-out),
inductive (synset hold-out)."""
import random
from collections import defaultdict


def reconstruction_split(data):
    return {"edges": list(data.closure_edges)}


def link_prediction_split(data, val_frac=0.05, test_frac=0.05, seed=0):
    edges = list(data.closure_edges)
    rng = random.Random(seed)
    rng.shuffle(edges)
    n = len(edges)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    valid = edges[:n_val]
    test = edges[n_val:n_val + n_test]
    train = edges[n_val + n_test:]
    return {"train": train, "valid": valid, "test": test}


def inductive_split(data, test_synset_frac=0.1, seed=0):
    rng = random.Random(seed)
    n = len(data.synsets)
    idx = list(range(n))
    rng.shuffle(idx)
    n_test = int(n * test_synset_frac)
    test_synsets = set(idx[:n_test])
    train_edges, eval_edges = [], []
    for h, hyper in data.closure_edges:
        if h in test_synsets or hyper in test_synsets:
            eval_edges.append((h, hyper))
        else:
            train_edges.append((h, hyper))
    return {
        "train_synsets": set(idx[n_test:]),
        "test_synsets": test_synsets,
        "train_edges": train_edges,
        "eval_edges": eval_edges,
    }


def _hypernyms_by_hypo(data):
    d = defaultdict(set)
    for h, hyper in data.closure_edges:
        d[h].add(hyper)
    return d


def negatives_for(edge, data, num, rng, _cache={}):
    hypo = edge[0]
    key = id(data)
    hyp_map = _cache.get(key)
    if hyp_map is None:
        hyp_map = _hypernyms_by_hypo(data)
        _cache[key] = hyp_map
    true_h = hyp_map[hypo] | {hypo}
    n = len(data.synsets)
    out = []
    while len(out) < num:
        cand = rng.randrange(n)
        if cand not in true_h:
            out.append(cand)
    return out
