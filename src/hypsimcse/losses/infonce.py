"""InfoNCE with a pluggable score, plus the H4 root-anchor regularizer."""
import torch
import torch.nn.functional as Fn
from ..geometry import scores as S
from ..geometry import lorentz as L


def info_nce(z, z_pos, label, tau, c=1.0):
    """Standard SimCSE/NT-Xent InfoNCE. Row i's positive is z_pos[i];
    z_pos[j != i] are in-batch negatives. Identical loss across conditions;
    only `label` changes the similarity."""
    logits = S.pairwise_scores(z, z_pos, label, c=c) / tau
    targets = torch.arange(z.shape[0], device=logits.device)
    return Fn.cross_entropy(logits, targets)


def root_anchor_reg(z, c=1.0):
    """Mean hyperbolic distance from the origin, negated. Minimizing this
    spreads embeddings outward from the root (H4 replacement for uniformity)."""
    o = L.origin(z.shape[-1] - 1, c=c, dtype=z.dtype, device=z.device)
    o = o.expand_as(z)
    return -L.dist(z, o, c=c).mean()


def total_loss(z, z_pos, label, tau, c=1.0, anchor_weight=0.0):
    nce = info_nce(z, z_pos, label, tau, c=c)
    parts = {"infonce": nce.detach().item(), "anchor": 0.0}
    loss = nce
    if anchor_weight > 0.0 and S.is_hyperbolic(label):
        anchor = root_anchor_reg(z, c=c)
        parts["anchor"] = anchor.detach().item()
        loss = loss + anchor_weight * anchor
    return loss, parts
