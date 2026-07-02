"""Link prediction on held-out hypernymy edges: MAP + ranking AUC."""
import random
import torch
from ..geometry import scores as S
from ..data.splits import negatives_for


def link_prediction_metrics(embeddings, test_edges, data, num_negatives, score,
                            c=1.0, seed=0):
    rng = random.Random(seed)
    recips, aucs = [], []
    for (hypo, hyper) in test_edges:
        negs = negatives_for((hypo, hyper), data, num_negatives, rng)
        q = embeddings[hypo:hypo + 1]
        cand = torch.cat([embeddings[hyper:hyper + 1], embeddings[torch.tensor(negs)]], dim=0)
        sims = S.pairwise_scores(q, cand, score, c=c)[0]
        pos_score = sims[0]
        neg_scores = sims[1:]
        rank = 1 + int((neg_scores > pos_score).sum().item())
        recips.append(1.0 / rank)
        aucs.append((neg_scores < pos_score).float().mean().item())
    return {
        "MAP": sum(recips) / max(len(recips), 1),
        "AUC": sum(aucs) / max(len(aucs), 1),
    }
