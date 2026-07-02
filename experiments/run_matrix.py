"""Expand a base config over dims/scores/seeds/taus and dump one JSON per cell."""
import argparse
import json
import os
from hypsimcse.training.config import ExperimentConfig
from run import run_experiment  # same directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--scores", nargs="*", default=None)
    ap.add_argument("--taus", type=float, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--max-synsets", type=int, default=None)
    ap.add_argument("--full-hierarchy", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    base = ExperimentConfig.from_yaml(args.config)
    if args.max_synsets is not None:
        base.max_synsets = args.max_synsets
    dims = args.dims or [base.dim]
    scores = args.scores or [base.score]
    taus = args.taus or [base.tau]
    for dim in dims:
        for score in scores:
            for tau in taus:
                for seed in args.seeds:
                    cfg = ExperimentConfig(**base.to_dict())
                    cfg.dim, cfg.tau, cfg.seed = dim, tau, seed
                    cfg.score = score
                    cfg.geometry = "EUC" if score == "EUC" else "HYP"
                    tag = f"d{dim}_{score}_tau{tau}_s{seed}"
                    res = run_experiment(cfg, full_hierarchy=args.full_hierarchy)
                    path = os.path.join(args.outdir, f"{tag}.json")
                    with open(path, "w") as f:
                        json.dump(res, f, indent=2)
                    print(f"wrote {path}")


if __name__ == "__main__":
    main()
