"""Tree-reconstruction metrics: rank true hypernym among negatives.
MAP (mean reciprocal), mean rank, average distortion."""
import random
import torch
from ..geometry import scores as S
from ..data.splits import negatives_for


def rank_of_positive(query_emb, pos_emb, neg_embs, score, c=1.0):
    cand = torch.cat([pos_emb, neg_embs], dim=0)          # (1+N, ·)
    sims = S.pairwise_scores(query_emb, cand, score, c=c)[0]  # (1+N,)
    order = torch.argsort(sims, descending=True)
    pos_position = (order == 0).nonzero(as_tuple=True)[0].item()
    return pos_position + 1


def reconstruction_metrics(embeddings, edges, data, num_negatives, score,
                           c=1.0, seed=0):
    rng = random.Random(seed)
    ranks = []
    for (hypo, hyper) in edges:
        negs = negatives_for((hypo, hyper), data, num_negatives, rng)
        q = embeddings[hypo:hypo + 1]
        pos = embeddings[hyper:hyper + 1]
        neg = embeddings[torch.tensor(negs)]
        ranks.append(rank_of_positive(q, pos, neg, score, c=c))
    ranks = torch.tensor(ranks, dtype=torch.float)
    return {
        "MAP": (1.0 / ranks).mean().item(),
        "mean_rank": ranks.mean().item(),
        "distortion": float("nan"),  # populated by caller when graph metric given
    }
