"""Inductive generalization: encode UNSEEN synsets' glosses and check they land
in the right place. Only the encoder track can do this (N-K is transductive)."""
from ..models.encoder_model import encode_texts
from .reconstruction import reconstruction_metrics


def inductive_metrics(model, tokenizer, data, inductive_split, num_negatives,
                      score, device, c=1.0, seed=0):
    emb = encode_texts(model, tokenizer, data.glosses, device=device)
    m = reconstruction_metrics(emb, inductive_split["eval_edges"], data,
                               num_negatives=num_negatives, score=score,
                               c=c, seed=seed)
    return {"MAP": m["MAP"], "mean_rank": m["mean_rank"]}
