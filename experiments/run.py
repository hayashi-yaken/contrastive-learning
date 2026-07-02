"""Run one ExperimentConfig end-to-end: train + eval -> JSON result."""
import argparse
import json
import random
import torch
from hypsimcse.training.config import ExperimentConfig
from hypsimcse.training.seeding import set_seed
from hypsimcse.training.device import get_device
from hypsimcse.training import trainer as T
from hypsimcse.data import wordnet as W
from hypsimcse.data import splits as SP
from hypsimcse.data import graph_stats as GS
from hypsimcse.eval import reconstruction as R
from hypsimcse.eval import link_prediction as LP
from hypsimcse.eval import inductive as IND
from hypsimcse.eval import sts as STS
from hypsimcse.eval import fisher as FI
from hypsimcse.eval import hierarchy as HI
from hypsimcse.eval import dual_coords as DC
from hypsimcse.models.encoder_model import encode_texts


def _all_embeddings(model, data, cfg, device):
    if cfg.track == "graph":
        with torch.no_grad():
            ids = torch.arange(len(data.synsets), device=device)
            return model(ids).cpu()
    return encode_texts(model, model.tokenizer, data.glosses, device=device)


def run_experiment(config, device=None, full_hierarchy=False):
    device = device or get_device()
    set_seed(config.seed)
    W.ensure_wordnet()
    data = W.load_noun_hypernymy(max_synsets=(config.max_synsets or None))

    model = T.build_model(config, data, device)
    tr = T.Trainer(config, data, model, device)
    train_logs = tr.train()

    emb = _all_embeddings(model, data, config, device)
    c = model.curvature()
    result = {"config": config.to_dict(), "train_logs": train_logs,
              "curvature": c, "device": device}

    result["data_stats"] = GS.summarize(data, num_samples=1000, seed=config.seed)

    rng = random.Random(config.seed)
    recon_edges = SP.reconstruction_split(data)["edges"]
    sample = recon_edges if len(recon_edges) <= 20000 else rng.sample(recon_edges, 20000)
    result["reconstruction"] = R.reconstruction_metrics(
        emb, sample, data, config.num_negatives, config.score, c=c, seed=config.seed)

    lp = SP.link_prediction_split(data, seed=config.seed)
    test = lp["test"] if len(lp["test"]) <= 20000 else rng.sample(lp["test"], 20000)
    result["link_prediction"] = LP.link_prediction_metrics(
        emb, test, data, config.num_negatives, config.score, c=c, seed=config.seed)

    result["fisher"] = FI.fisher_eigenvalues(emb, c=c)
    nd = HI.norm_depth_correlation(emb, data, c=c)
    if full_hierarchy:
        result["norm_depth"] = nd
    else:
        result["norm_depth"] = {k: nd[k] for k in ("spearman", "pearson")}
    result["embedding_delta"] = HI.embedding_delta_hyperbolicity(
        emb, num_samples=1000, c=c, seed=config.seed)

    if emb.shape[0] >= 8 and config.score.startswith("HYP"):
        q = emb[0:1]
        result["dual_gap"] = DC.dual_coordinate_gap(q, emb[1:65], config.tau, c=c)
    else:
        result["dual_gap"] = None

    if config.track == "encoder":
        isplit = SP.inductive_split(data, seed=config.seed)
        result["inductive"] = IND.inductive_metrics(
            model, model.tokenizer, data, isplit, config.num_negatives,
            config.score, device=device, c=c, seed=config.seed)
        pairs, gold = STS.load_stsb_validation()
        if pairs:
            result["sts"] = STS.sts_spearman(model, model.tokenizer, pairs, gold,
                                             config.score, device=device, c=c)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-synsets", type=int, default=None)
    ap.add_argument("--full-hierarchy", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_synsets is not None:
        cfg.max_synsets = args.max_synsets
    result = run_experiment(cfg, full_hierarchy=args.full_hierarchy)
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
