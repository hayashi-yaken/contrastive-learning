"""Data-side priors: how tree-like is WordNet noun hypernymy? Gromov delta,
depth, branching. Establishes the precondition for hyperbolic advantage (H1)
and the depth baseline for norm-vs-depth (H4)."""
import random
from collections import defaultdict, deque


def _children_map(data):
    ch = defaultdict(list)
    for h, hyper in data.direct_edges:
        ch[hyper].append(h)
    return ch


def _parents_map(data):
    pa = defaultdict(list)
    for h, hyper in data.direct_edges:
        pa[h].append(hyper)
    return pa


def depths(data):
    """Hops up to nearest root via direct hypernym edges (BFS from roots down)."""
    ch = _children_map(data)
    depth = {r: 0 for r in data.roots}
    dq = deque(data.roots)
    while dq:
        u = dq.popleft()
        for c in ch[u]:
            if c not in depth or depth[c] > depth[u] + 1:
                depth[c] = depth[u] + 1
                dq.append(c)
    for i in range(len(data.synsets)):  # disconnected fallback
        depth.setdefault(i, 0)
    return depth


def branching_factors(data):
    ch = _children_map(data)
    counts = [len(v) for v in ch.values()]
    hist = defaultdict(int)
    for c in counts:
        hist[c] += 1
    mean = sum(counts) / len(counts) if counts else 0.0
    return {"mean": mean, "max": max(counts) if counts else 0, "hist": dict(hist)}


def depth_distribution(data):
    d = depths(data)
    hist = defaultdict(int)
    for v in d.values():
        hist[v] += 1
    return dict(hist)


def _undirected_adj(data):
    adj = defaultdict(set)
    for h, hyper in data.direct_edges:
        adj[h].add(hyper)
        adj[hyper].add(h)
    return adj


def _bfs_dist(adj, src, targets):
    dist = {src: 0}
    dq = deque([src])
    remaining = set(targets)
    remaining.discard(src)
    while dq and remaining:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                remaining.discard(v)
                dq.append(v)
    return dist


def gromov_delta(data, num_samples=2000, seed=0):
    """Sampled 4-point Gromov delta on the undirected shortest-path metric.
    delta = max over sampled quadruples of the gap between the two largest of
    the three pairwise-sum matchings, halved. 0 => tree metric."""
    adj = _undirected_adj(data)
    rng = random.Random(seed)
    n = len(data.synsets)
    nodes = [i for i in range(n) if adj[i]]
    if len(nodes) < 4:
        return 0.0
    delta = 0.0
    dist_cache = {}

    def dist(a, b):
        if a == b:
            return 0
        if a not in dist_cache:
            dist_cache[a] = _bfs_dist(adj, a, nodes)
        return dist_cache[a].get(b, float("inf"))

    for _ in range(num_samples):
        w, x, y, z = rng.sample(nodes, 4)
        d = {(w, x): dist(w, x), (w, y): dist(w, y), (w, z): dist(w, z),
             (x, y): dist(x, y), (x, z): dist(x, z), (y, z): dist(y, z)}
        if any(v == float("inf") for v in d.values()):
            continue
        s1 = d[(w, x)] + d[(y, z)]
        s2 = d[(w, y)] + d[(x, z)]
        s3 = d[(w, z)] + d[(x, y)]
        s = sorted([s1, s2, s3])
        delta = max(delta, (s[2] - s[1]) / 2.0)
    return delta


def summarize(data, num_samples=2000, seed=0):
    b = branching_factors(data)
    return {
        "num_synsets": len(data.synsets),
        "num_direct_edges": len(data.direct_edges),
        "num_closure_edges": len(data.closure_edges),
        "num_roots": len(data.roots),
        "branching_mean": b["mean"],
        "branching_max": b["max"],
        "depth_distribution": depth_distribution(data),
        "gromov_delta": gromov_delta(data, num_samples=num_samples, seed=seed),
    }
