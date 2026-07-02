"""H4 signal: does hierarchy organize as root-anchored scatter? Norm-vs-depth
correlation + delta-hyperbolicity of the learned embedding metric."""
import random
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from ..geometry import lorentz as L
from ..data.graph_stats import depths


def _is_hyperboloid(z, c):
    ip = L.lorentz_inner(z, z)
    return bool(torch.allclose(ip, torch.full_like(ip, -1.0 / c), atol=1e-2))


def embedding_norms(embeddings, c=1.0):
    if embeddings.shape[-1] >= 2 and _is_hyperboloid(embeddings, c):
        o = L.origin(embeddings.shape[-1] - 1, c=c, dtype=embeddings.dtype,
                     device=embeddings.device).expand_as(embeddings)
        return L.dist(embeddings, o, c=c)
    return torch.norm(embeddings, dim=-1)


def norm_depth_correlation(embeddings, data, c=1.0):
    norms = embedding_norms(embeddings, c=c).detach().cpu().numpy()
    depth = depths(data)
    d = np.array([depth[i] for i in range(len(data.synsets))], dtype=float)
    n = norms[:len(d)]
    rho, _ = spearmanr(n, d)
    r, _ = pearsonr(n, d)
    return {"spearman": float(rho), "pearson": float(r),
            "norms": n.tolist(), "depths": d.tolist()}


def embedding_delta_hyperbolicity(embeddings, num_samples=1000, c=1.0, seed=0):
    rng = random.Random(seed)
    n = embeddings.shape[0]
    if n < 4:
        return 0.0
    is_hyp = _is_hyperboloid(embeddings, c)

    def dist(a, b):
        if is_hyp:
            return L.dist(embeddings[a:a + 1], embeddings[b:b + 1], c=c).item()
        return torch.norm(embeddings[a] - embeddings[b]).item()

    delta = 0.0
    for _ in range(num_samples):
        w, x, y, z = rng.sample(range(n), 4)
        s = sorted([dist(w, x) + dist(y, z),
                    dist(w, y) + dist(x, z),
                    dist(w, z) + dist(x, y)])
        delta = max(delta, (s[2] - s[1]) / 2.0)
    return delta
