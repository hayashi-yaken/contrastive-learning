"""STS-B Spearman quality guard: does the gloss encoder preserve general
semantic similarity while specializing to hierarchy?"""
import torch
from scipy.stats import spearmanr
from ..geometry import scores as S
from ..models.encoder_model import encode_texts


def sts_spearman(model, tokenizer, pairs, gold, score, device, c=1.0):
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    za = encode_texts(model, tokenizer, a, device=device)
    zb = encode_texts(model, tokenizer, b, device=device)
    sims = torch.stack([
        S.pairwise_scores(za[i:i + 1], zb[i:i + 1], score, c=c)[0, 0]
        for i in range(len(pairs))
    ]).tolist()
    rho, _ = spearmanr(sims, gold)
    return float(rho)


def load_stsb_validation():
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "stsb", split="validation")
        pairs = list(zip(ds["sentence1"], ds["sentence2"]))
        gold = list(ds["label"])
        return pairs, gold
    except Exception as e:  # noqa: BLE001
        print(f"STS-B unavailable: {e}")
        return [], []
